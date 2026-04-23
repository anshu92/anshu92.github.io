"""Read and structure primary-paper evidence from PDF text."""

from __future__ import annotations

import io
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

import httpx

from . import config, lint, openrouter_client
from .models import Item, Quote, is_allowed_url

LOG = logging.getLogger(__name__)
_TIMEOUT = httpx.Timeout(60.0, connect=15.0)


@dataclass
class PaperEvidence:
    pdf_url: str = ""
    cleaned_text: str = ""
    outline: list[str] = field(default_factory=list)
    section_evidence: dict[str, str] = field(default_factory=dict)
    result_notes: list[str] = field(default_factory=list)
    limitation_notes: list[str] = field(default_factory=list)
    reproducibility_notes: list[str] = field(default_factory=list)
    quotes: list[Quote] = field(default_factory=list)
    llm_chunks_used: int = 0


def _client() -> httpx.Client:
    return httpx.Client(
        verify=True,
        timeout=_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": "blogpipe/0.1 (+https://anshu92.github.io)"},
    )


def _extract_arxiv_id(text: str) -> str:
    s = text or ""
    patterns = (
        r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})(?:v\d+)?",
        r"\b(?:hf|arxiv|ss)_(\d{4}\.\d{4,5})\b",
        r"\barXiv:(\d{4}\.\d{4,5})\b",
    )
    for pat in patterns:
        m = re.search(pat, s, re.I)
        if m:
            return m.group(1)
    return ""


def candidate_pdf_urls(primary: Item) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    def add(url: str) -> None:
        u = (url or "").strip()
        if not u or u in seen:
            return
        seen.add(u)
        urls.append(u)

    if primary.url:
        if "/pdf/" in primary.url:
            add(primary.url if primary.url.endswith(".pdf") else primary.url + ".pdf")
        if "/abs/" in primary.url or "/html/" in primary.url:
            arxiv_id = _extract_arxiv_id(primary.url)
            if arxiv_id:
                add(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
    arxiv_id = _extract_arxiv_id(primary.id) or _extract_arxiv_id(primary.url)
    if arxiv_id:
        add(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
    paper_url = str(primary.extra.get("paper_url") or primary.extra.get("pdf_url") or "").strip()
    if paper_url:
        add(paper_url)
    return [u for u in urls if is_allowed_url(u)]


def _pdf_bytes_to_text_pypdf(data: bytes) -> str:
    from pypdf import PdfReader  # lazy import: tests do not require the dependency installed locally

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as e:  # noqa: BLE001
            LOG.debug("paper_reader page extract: %s", e)
    return "\n".join(pages)


def _pdf_bytes_to_text_pdftotext(data: bytes) -> str:
    exe = shutil.which("pdftotext")
    if not exe:
        return ""
    try:
        proc = subprocess.run(
            [exe, "-nopgbrk", "-", "-"],
            input=data,
            capture_output=True,
            check=True,
        )
    except Exception as e:  # noqa: BLE001
        LOG.debug("paper_reader pdftotext: %s", e)
        return ""
    return proc.stdout.decode("utf-8", errors="ignore")


def _pdf_bytes_to_text(data: bytes) -> str:
    text = _pdf_bytes_to_text_pdftotext(data)
    if len(text.split()) >= 400:
        return text
    alt = ""
    try:
        alt = _pdf_bytes_to_text_pypdf(data)
    except Exception as e:  # noqa: BLE001
        LOG.debug("paper_reader pypdf unavailable: %s", e)
    return alt if len(alt.split()) > len(text.split()) else text


def _is_heading(line: str) -> bool:
    s = (line or "").strip()
    if not s or len(s) > 120:
        return False
    if s in {"Abstract", "References", "Conclusion", "Discussion", "Methodology", "Introduction"}:
        return True
    if re.match(
        r"^(?:[A-Z][A-Za-z0-9()/-]*)(?:\s+[A-Z][A-Za-z0-9()/-]*){0,7}$",
        s,
    ):
        return True
    return bool(
        re.match(r"^(?:[A-Z]\.\s+)?\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9][^\n]{0,100}$", s)
    )


def clean_pdf_text(text: str, title: str = "") -> str:
    lines = (text or "").replace("\x0c", "\n").splitlines()
    cleaned: list[str] = []
    buf = ""
    for raw in lines:
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            if buf:
                cleaned.append(buf.strip())
                buf = ""
            continue
        if title and line == title:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if "Correspondence to:" in line or "@" in line:
            continue
        if line.startswith("arXiv:") or line.startswith("Preprint."):
            continue
        if _is_heading(line):
            if buf:
                cleaned.append(buf.strip())
                buf = ""
            cleaned.append(line)
            continue
        if not buf:
            buf = line
            continue
        if buf.endswith("-") and line[:1].islower():
            buf = buf[:-1] + line
        else:
            buf = buf + " " + line
    if buf:
        cleaned.append(buf.strip())
    out = "\n\n".join(x for x in cleaned if x).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def extract_named_sections(text: str) -> dict[str, str]:
    src = text or ""
    pat = re.compile(
        r"(?m)^(Abstract|(?:[A-Z]\.\s+)?\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9][^\n]{0,100}|"
        r"(?:[A-Z][A-Za-z0-9()/-]*)(?:\s+[A-Z][A-Za-z0-9()/-]*){0,7})$"
    )
    matches = list(pat.finditer(src))
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(src)
        body = src[body_start:body_end].strip()
        key = re.sub(r"^(?:[A-Z]\.\s+)?\d+(?:\.\d+)*\.?\s+", "", heading).strip().lower()
        if key:
            sections[key] = body[:4000]
    return sections


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [x.strip() for x in parts if x and x.strip()]


def _pick_section(sections: dict[str, str], *keywords: str) -> str:
    for key, body in sections.items():
        if any(k in key for k in keywords) and body.strip():
            return body.strip()
    return ""


def _pick_sentences(text: str, patterns: tuple[str, ...], limit: int = 4, min_words: int = 4) -> list[str]:
    out: list[str] = []
    for sent in _sentences(text):
        if len(sent.split()) < min_words:
            continue
        if "Correspondence to:" in sent or "@" in sent:
            continue
        if any(re.search(pat, sent, re.I) for pat in patterns):
            out.append(sent)
        if len(out) >= limit:
            break
    return out


def _rank_notes(
    notes: list[str],
    preferred: tuple[str, ...] = (),
    penalized: tuple[str, ...] = (),
) -> list[str]:
    scored: list[tuple[int, int, str]] = []
    for idx, sent in enumerate(notes):
        score = 0
        for pat in preferred:
            if re.search(pat, sent, re.I):
                score += 2
        for pat in penalized:
            if re.search(pat, sent, re.I):
                score -= 2
        scored.append((-score, idx, sent))
    return [sent for _, _, sent in sorted(scored)]


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_author_voice(text: str) -> str:
    out = _normalize_space(text)
    replacements = (
        (
            r"\bWe (introduce|present|propose|show|find|observe|use|model|evaluate|analyze|"
            r"study|train|fine-tune|adapt|apply|achieve|compare|extend|demonstrate|report|focus|"
            r"design|identify|derive|select|benchmark|measure|test)\b",
            r"The authors \1",
        ),
        (
            r"\bwe (introduce|present|propose|show|find|observe|use|model|evaluate|analyze|"
            r"study|train|fine-tune|adapt|apply|achieve|compare|extend|demonstrate|report|focus|"
            r"design|identify|derive|select|benchmark|measure|test)\b",
            r"the authors \1",
        ),
        (
            r"\bOur ((?:[A-Za-z-]+\s+){0,3}(?:methodology|method|methods|approach|approaches|"
            r"model|models|results|experiments|study|paper|framework|analysis|system|systems|"
            r"technique|techniques|baseline|baselines|llms?|μlms?))\b",
            r"The authors' \1",
        ),
        (
            r"\bour ((?:[A-Za-z-]+\s+){0,3}(?:methodology|method|methods|approach|approaches|"
            r"model|models|results|experiments|study|paper|framework|analysis|system|systems|"
            r"technique|techniques|baseline|baselines|llms?|μlms?))\b",
            r"the authors' \1",
        ),
    )
    for pat, repl in replacements:
        out = re.sub(pat, repl, out)
    out = re.sub(r"\b([Tt]he authors [a-z]+) the authors'\s+", r"\1 their ", out)
    return out


def _normalize_claim(text: str) -> str:
    t = (text or "").lower().replace("×", "x")
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"\bparameters?\b", "param", t)
    t = re.sub(r"\bparams?\b", "param", t)
    t = re.sub(r"\bwords?\b", "word", t)
    t = re.sub(r"\blayers?\b", "layer", t)
    t = re.sub(r"\btokens?\b", "token", t)
    t = re.sub(r"\bhours?\b", "hour", t)
    t = re.sub(r"\bdays?\b", "day", t)
    t = re.sub(r"\s+", "", t)
    return t


def _claims_supported(summary: str, source_text: str) -> bool:
    claims = {_normalize_claim(x) for x in lint.numeric_claims(summary)}
    if not claims:
        return True
    source_claims = {_normalize_claim(x) for x in lint.numeric_claims(source_text)}
    return claims.issubset(source_claims)


def _support_in_chunk(support: str, chunk_text: str) -> bool:
    norm_support = _normalize_space(support)
    norm_chunk = _normalize_space(chunk_text)
    return bool(norm_support and norm_support in norm_chunk)


def _dedupe_texts(items: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _normalize_space(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


_SETUP_SIGNAL = re.compile(
    r"\b(qwen|llama|gemma|deepseek|mmlu|orcamath|dataset|benchmark|hardware|gpu|"
    r"batch size|learning rate|sequence length|checkpoint)\b",
    re.I,
)
_LIMITATION_SIGNAL = re.compile(
    r"\b(limit|future work|single benchmark|single run|variance|seed|generaliz|"
    r"reproduc|public code|not exhaustively|scope)\b",
    re.I,
)
_RESULT_SIGNAL = re.compile(
    r"\b(\d+(?:\.\d+)?%|outperform|accuracy|baseline|mmlu|f1|bleu|rouge|score)\b",
    re.I,
)


def _looks_like_category(text: str, kind: str) -> bool:
    src = _normalize_space(text)
    if not src:
        return False
    if kind == "setup":
        return bool(_SETUP_SIGNAL.search(src))
    if kind == "limitation":
        return bool(_LIMITATION_SIGNAL.search(src))
    if kind == "result":
        return bool(_RESULT_SIGNAL.search(src))
    if kind == "reproducibility":
        return bool(_SETUP_SIGNAL.search(src))
    return True


def _section_chunks(cleaned: str, sections: dict[str, str], max_chars: int = 5000, max_chunks: int = 4) -> list[tuple[list[str], str]]:
    entries = [(heading, body.strip()) for heading, body in sections.items() if body.strip()]
    if entries:
        chunks: list[tuple[list[str], str]] = []
        cur_heads: list[str] = []
        cur_parts: list[str] = []
        cur_len = 0
        for heading, body in entries:
            part = f"## {heading.title()}\n{body}"
            if cur_parts and cur_len + len(part) > max_chars and len(chunks) < max_chunks - 1:
                chunks.append((cur_heads, "\n\n".join(cur_parts)))
                cur_heads = [heading.title()]
                cur_parts = [part]
                cur_len = len(part)
            else:
                cur_heads.append(heading.title())
                cur_parts.append(part)
                cur_len += len(part)
        if cur_parts:
            chunks.append((cur_heads, "\n\n".join(cur_parts)))
        return chunks[:max_chunks]

    paras = [p.strip() for p in re.split(r"\n{2,}", cleaned) if p.strip()]
    chunks: list[tuple[list[str], str]] = []
    cur: list[str] = []
    cur_len = 0
    for para in paras:
        if cur and cur_len + len(para) > max_chars and len(chunks) < max_chunks - 1:
            chunks.append(([], "\n\n".join(cur)))
            cur = [para]
            cur_len = len(para)
        else:
            cur.append(para)
            cur_len += len(para)
    if cur:
        chunks.append(([], "\n\n".join(cur)))
    return chunks[:max_chunks]


def _parse_json_obj(raw: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", raw or "")
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _llm_chunk_summary(primary: Item, headings: list[str], chunk_text: str) -> dict[str, Any]:
    system = (
        "You read one chunk from a machine learning research paper and extract grounded notes. "
        "Output JSON only with this shape: "
        '{"problem":"","method":"","setup":"","results":[{"summary":"","support":""}],'
        '"limitations":[{"summary":"","support":""}],"reproducibility":[{"summary":"","support":""}],'
        '"quotes":[""]}. '
        "Rules: use only facts present in CHUNK_TEXT; write summaries in neutral third-person, not "
        "'we' or 'our'; support fields must be short verbatim spans copied from CHUNK_TEXT; "
        "prefer benchmark results, named baselines, datasets, models, hardware, and explicit "
        "limitations; if a field is absent in this chunk, return '' or []. Keep each summary to 1-2 "
        "sentences. Max 3 results, 3 limitations, 3 reproducibility items, and 3 quotes."
    )
    user = (
        f"TITLE: {primary.title}\n"
        f"CHUNK_HEADINGS: {', '.join(headings) if headings else '(none)'}\n\n"
        f"CHUNK_TEXT:\n{chunk_text[:12000]}"
    )
    return _parse_json_obj(
        openrouter_client.llm_text(
            system,
            user,
            mode="smart",
            max_tokens=min(1600, config.max_tokens_smart()),
        )
    )


def _validated_scalar_summary(raw: object, chunk_text: str, kind: str = "") -> str:
    text = _normalize_author_voice(_normalize_space(str(raw or "")))[:700]
    if len(text.split()) < 5:
        return ""
    if kind and not _looks_like_category(text, kind):
        return ""
    if not _claims_supported(text, chunk_text):
        return ""
    return text


def _validated_note_items(
    raw: object,
    chunk_text: str,
    *,
    kind: str = "",
    limit: int = 4,
) -> tuple[list[str], list[str]]:
    if not isinstance(raw, list):
        return [], []
    notes: list[str] = []
    supports: list[str] = []
    for item in raw:
        summary = ""
        support = ""
        if isinstance(item, str):
            summary = str(item)
        elif isinstance(item, dict):
            summary = str(item.get("summary") or item.get("claim") or item.get("text") or "")
            support = str(item.get("support") or item.get("quote") or "")
        summary = _normalize_author_voice(_normalize_space(summary))[:500]
        support = _normalize_space(support)[:280]
        if len(summary.split()) < 4:
            continue
        if kind and not _looks_like_category(summary, kind):
            continue
        if not _claims_supported(summary, chunk_text):
            continue
        notes.append(summary)
        if support and _support_in_chunk(support, chunk_text):
            supports.append(support)
        if len(notes) >= limit:
            break
    return _dedupe_texts(notes, limit), _dedupe_texts(supports, limit)


def _validated_quotes(raw: object, chunk_text: str, limit: int = 4) -> list[str]:
    if not isinstance(raw, list):
        return []
    quotes: list[str] = []
    for item in raw:
        quote = _normalize_space(str(item or ""))[:280]
        if not quote or not _support_in_chunk(quote, chunk_text):
            continue
        quotes.append(quote)
        if len(quotes) >= limit:
            break
    return _dedupe_texts(quotes, limit)


def _llm_paper_summary(primary: Item, cleaned: str, sections: dict[str, str]) -> dict[str, Any]:
    if config.dry_run() or not config.llm_configured():
        return {}
    chunks = _section_chunks(cleaned, sections)
    if not chunks:
        return {}

    problem_candidates: list[str] = []
    method_candidates: list[str] = []
    setup_candidates: list[str] = []
    result_notes: list[str] = []
    limitation_notes: list[str] = []
    reproducibility_notes: list[str] = []
    quote_spans: list[str] = []
    llm_chunks_used = 0

    for headings, chunk_text in chunks:
        data = _llm_chunk_summary(primary, headings, chunk_text)
        if not data:
            continue
        llm_chunks_used += 1
        problem = _validated_scalar_summary(data.get("problem"), chunk_text)
        method = _validated_scalar_summary(data.get("method"), chunk_text)
        setup = _validated_scalar_summary(data.get("setup"), chunk_text, kind="setup")
        res_notes, res_supports = _validated_note_items(
            data.get("results"),
            chunk_text,
            kind="result",
            limit=4,
        )
        lim_notes, lim_supports = _validated_note_items(
            data.get("limitations"),
            chunk_text,
            kind="limitation",
            limit=4,
        )
        repro_notes, repro_supports = _validated_note_items(
            data.get("reproducibility"),
            chunk_text,
            kind="reproducibility",
            limit=4,
        )
        extra_quotes = _validated_quotes(data.get("quotes"), chunk_text, limit=4)

        if problem:
            problem_candidates.append(problem)
        if method:
            method_candidates.append(method)
        if setup:
            setup_candidates.append(setup)
        result_notes.extend(res_notes)
        limitation_notes.extend(lim_notes)
        reproducibility_notes.extend(repro_notes)
        quote_spans.extend(res_supports + lim_supports + repro_supports + extra_quotes)

    if llm_chunks_used == 0:
        return {}

    problem = next((x for x in problem_candidates if not lint.numeric_claims(x)), "") or (
        problem_candidates[0] if problem_candidates else ""
    )
    return {
        "problem": problem,
        "method": method_candidates[0] if method_candidates else "",
        "setup": setup_candidates[0] if setup_candidates else "",
        "result_notes": _dedupe_texts(result_notes, 6),
        "limitation_notes": _dedupe_texts(limitation_notes, 5),
        "reproducibility_notes": _dedupe_texts(reproducibility_notes, 6),
        "quote_spans": _dedupe_texts(quote_spans, 8),
        "llm_chunks_used": llm_chunks_used,
    }


def build_paper_evidence_from_text(primary: Item, raw_text: str, pdf_url: str = "") -> PaperEvidence:
    cleaned = clean_pdf_text(raw_text, primary.title)
    sections = extract_named_sections(cleaned)
    if len(cleaned.split()) < 120 and len(sections) < 4:
        return PaperEvidence()
    intro = _pick_section(sections, "introduction") or _pick_section(sections, "abstract")
    method = _pick_section(sections, "method", "approach", "framework")
    experiments = (
        _pick_section(sections, "core experiments", "experiments", "results", "evaluation")
        or _pick_section(sections, "discussion")
    )
    limitations = _pick_section(sections, "limitations")
    future = _pick_section(sections, "future work")
    conclusion = _pick_section(sections, "conclusion")
    setup = _pick_section(sections, "training configuration", "hardware", "cluster environment", "software stack")
    llm_summary = _llm_paper_summary(primary, cleaned, sections)

    result_notes = _pick_sentences(
        cleaned,
        (
            r"\b(outperform|speedup|throughput|accuracy|wer|f1|bleu|rouge|reaches|yields|achieve)\b",
            r"\b\d+(?:\.\d+)?\s*%\b",
        ),
        limit=6,
    )
    result_notes = _rank_notes(
        result_notes,
        preferred=(r"\bqwen", r"\bmmlu", r"\boutperform", r"\bfull\b", r"\brandom\b", r"\b\d+(?:\.\d+)?%"),
        penalized=(r"\bet al\.", r"\bcorrespondence\b"),
    )
    limitation_notes = _pick_sentences(
        limitations or cleaned,
        (
            r"\blimit",
            r"\bfuture work\b",
            r"\bnot exhaustively\b",
            r"\bsingle benchmark\b",
            r"\bvariance analysis\b",
            r"\bleft for future work\b",
            r"\bgeneraliz",
        ),
        limit=5,
    )
    limitation_notes = _rank_notes(
        limitation_notes,
        preferred=(
            r"\bsingle run\b",
            r"\bsingle benchmark\b",
            r"\bfuture work\b",
            r"\bvariance\b",
            r"\bseed",
            r"\bgeneraliz",
            r"\bnot exhaustively\b",
            r"\breported results\b",
        ),
        penalized=(r"\bthis property\b", r"\bthis structural pattern\b", r"\bet al\."),
    )
    limitation_notes = [
        x
        for x in limitation_notes
        if not re.search(
            r"^\(%\)|\bto address this limitation\b|\bthis property\b|\bthis structural pattern\b",
            x,
            re.I,
        )
    ]
    reproducibility_notes = _pick_sentences(
        setup or experiments or cleaned,
        (
            r"\bdataset\b",
            r"\btraining configuration\b",
            r"\bidentical training and evaluation settings\b",
            r"\bqwen|llama|gemma|deepseek|whisper\b",
            r"\bgpu|node|cluster|batch size|sequence length|learning rate\b",
        ),
        limit=6,
        min_words=2,
    )
    reproducibility_notes = _rank_notes(
        reproducibility_notes,
        preferred=(r"\bmodel:\b", r"\bdataset:\b", r"\bbenchmark:\b", r"\bqwen", r"\bgemma", r"\bdeepseek", r"\borcamath"),
    )
    if llm_summary.get("problem"):
        intro = str(llm_summary["problem"])
    if llm_summary.get("method"):
        method = str(llm_summary["method"])
    if llm_summary.get("setup"):
        setup = str(llm_summary["setup"])
    if llm_summary.get("result_notes"):
        result_notes = list(llm_summary["result_notes"])
    if llm_summary.get("limitation_notes"):
        limitation_notes = list(llm_summary["limitation_notes"])
    if llm_summary.get("reproducibility_notes"):
        reproducibility_notes = list(llm_summary["reproducibility_notes"])
    outline = [k.title() for k in sections.keys()][:12]
    section_evidence: dict[str, str] = {}
    if outline:
        section_evidence["paper_outline"] = "\n".join(f"- {x}" for x in outline)
    if intro:
        section_evidence["paper_problem"] = intro[:1400]
    if method:
        section_evidence["paper_method"] = method[:1400]
    if setup:
        section_evidence["paper_setup"] = setup[:1400]
    if result_notes:
        section_evidence["paper_experiments"] = "\n".join(f"- {x}" for x in result_notes[:4])[:1800]
    elif experiments:
        section_evidence["paper_experiments"] = experiments[:1800]
    if limitation_notes or limitations:
        text = "\n".join(f"- {x}" for x in limitation_notes[:4]) if limitation_notes else limitations[:1400]
        section_evidence["paper_limitations"] = text[:1400]
    if future:
        section_evidence["paper_future_work"] = future[:1000]
    if conclusion:
        section_evidence["paper_conclusion"] = conclusion[:1000]
    if reproducibility_notes:
        section_evidence["paper_reproducibility"] = "\n".join(f"- {x}" for x in reproducibility_notes)[:1200]

    quotes: list[Quote] = []
    quote_spans = list(llm_summary.get("quote_spans") or [])
    for sent in (quote_spans[:4] + result_notes[:3] + limitation_notes[:2]):
        quotes.append(Quote(source_id=primary.id, text=sent[:500], url=pdf_url or primary.url))
    return PaperEvidence(
        pdf_url=pdf_url,
        cleaned_text=cleaned[:24000],
        outline=outline,
        section_evidence=section_evidence,
        result_notes=result_notes,
        limitation_notes=limitation_notes,
        reproducibility_notes=reproducibility_notes,
        quotes=quotes,
        llm_chunks_used=int(llm_summary.get("llm_chunks_used", 0) or 0),
    )


def fetch_primary_paper_evidence(primary: Item) -> PaperEvidence:
    for url in candidate_pdf_urls(primary):
        try:
            r = _client().get(url)
            r.raise_for_status()
            text = _pdf_bytes_to_text(r.content)
        except Exception as e:  # noqa: BLE001
            LOG.warning("paper_reader fetch %s: %s", url, e)
            continue
        evidence = build_paper_evidence_from_text(primary, text, pdf_url=url)
        if evidence.cleaned_text:
            return evidence
    return PaperEvidence()
