from __future__ import annotations

import html
import re
from pathlib import Path

from . import memory
from .models import ComponentSpec, TableSpec, VisualAsset, VisualPlan

UNSAFE_HTML_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<\s*script\b", re.I),
    re.compile(r"<\s*iframe\b", re.I),
    re.compile(r"\son[a-z]+\s*=", re.I),
    re.compile(r"\b(?:src|href)\s*=\s*['\"]\s*https?://", re.I),
    re.compile(r"<\s*link\b", re.I),
    re.compile(r"<\s*style\b", re.I),
)
SHORTCODE_RE = re.compile(r"{{\s*[<%].*?[>%]\s*}}", re.S)
ALLOWED_SHORTCODES = {"quote"}


def render_visual_assets(plan: VisualPlan, *, slug: str) -> list[VisualAsset]:
    rendered: list[VisualAsset] = []
    out_dir = memory.post_asset_dir(slug)
    for asset in plan.assets:
        if asset.artifact_type != "svg":
            rendered.append(asset)
            continue
        filename = f"{memory.slugify(asset.asset_id, limit=48)}.svg"
        path = out_dir / filename
        _write_svg(asset, path)
        rendered.append(asset.model_copy(update={"path": f"/img/posts/{slug}/{filename}"}))
    return rendered


def validate_visual_plan(plan: VisualPlan, *, evidence_ids: set[str]) -> list[str]:
    errors: list[str] = []
    if not (plan.assets or plan.tables or plan.components):
        errors.append("visual_plan_empty")
    for asset in plan.assets:
        errors.extend(_unknown_evidence_errors("visual_asset", asset.asset_id, asset.evidence_ids, evidence_ids))
        if not asset.title.strip() or not asset.purpose.strip():
            errors.append(f"visual_asset_missing_purpose:{asset.asset_id}")
        if "visual map" in f"{asset.title} {asset.purpose}".lower():
            errors.append(f"generic_visual_asset:{asset.asset_id}")
    for table in plan.tables:
        errors.extend(validate_table(table, evidence_ids=evidence_ids))
    for component in plan.components:
        errors.extend(validate_component(component, evidence_ids=evidence_ids))
    return errors


def validate_table(table: TableSpec, *, evidence_ids: set[str]) -> list[str]:
    errors: list[str] = []
    errors.extend(_unknown_evidence_errors("table", table.table_id, table.evidence_ids, evidence_ids))
    if len(table.headers) < 2:
        errors.append(f"table_too_few_columns:{table.table_id}")
    if not table.rows:
        errors.append(f"table_no_rows:{table.table_id}")
    for row in table.rows:
        if len(row) != len(table.headers):
            errors.append(f"table_row_width_mismatch:{table.table_id}")
            break
    blob = " ".join([" ".join(table.headers), *(" ".join(row) for row in table.rows)])
    if re.search(r"(?<![\w-])\d+(?:\.\d+)?%?(?![\w-])", blob) and not table.evidence_ids:
        errors.append(f"table_numeric_claim_without_evidence:{table.table_id}")
    if not table.purpose.strip():
        errors.append(f"table_missing_purpose:{table.table_id}")
    return errors


def validate_component(component: ComponentSpec, *, evidence_ids: set[str]) -> list[str]:
    errors: list[str] = []
    errors.extend(_unknown_evidence_errors("component", component.component_id, component.evidence_ids, evidence_ids))
    if not component.purpose.strip():
        errors.append(f"component_missing_purpose:{component.component_id}")
    html_text = component.html or ""
    for pattern in UNSAFE_HTML_PATTERNS:
        if pattern.search(html_text):
            errors.append(f"unsafe_component_html:{component.component_id}")
            break
    for shortcode in SHORTCODE_RE.findall(html_text):
        shortcode_name = _shortcode_name(shortcode)
        if shortcode_name and shortcode_name not in ALLOWED_SHORTCODES:
            errors.append(f"unsupported_shortcode:{component.component_id}:{shortcode_name}")
    return errors


def markdown_table(table: TableSpec) -> str:
    headers = [_cell(h) for h in table.headers]
    lines = [f"**{table.title}**", "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in table.rows:
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines)


def component_html(component: ComponentSpec) -> str:
    return component.html.strip()


def mermaid_block(asset: VisualAsset) -> str:
    return "```mermaid\n" + asset.content.strip() + "\n```"


def svg_markdown(asset: VisualAsset) -> str:
    if not asset.path:
        return ""
    alt = asset.title.replace('"', "'")
    return f"![{alt}]({asset.path})"


def _write_svg(asset: VisualAsset, path: Path) -> None:
    if asset.asset_id == "matmul-output-cell":
        _write_matmul_cell_svg(asset, path)
        return
    title = asset.title or "Technical visual"
    purpose = asset.purpose or "Explain the lesson mechanism"
    labels = _svg_labels(asset)
    width = 860
    height = 220 + 58 * max(1, len(labels))
    rows: list[str] = []
    for idx, label in enumerate(labels, start=1):
        y = 120 + (idx - 1) * 58
        rows.append(f'<circle cx="56" cy="{y}" r="18" fill="#2563eb"/>')
        rows.append(
            f'<text x="56" y="{y + 6}" text-anchor="middle" font-size="16" fill="#fff" '
            f'font-family="Arial" font-weight="700">{idx}</text>'
        )
        rows.append(
            f'<text x="92" y="{y + 6}" font-size="18" fill="#111827" '
            f'font-family="Arial">{html.escape(label[:88])}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img" '
        f'aria-labelledby="title desc">'
        f'<title id="title">{html.escape(title)}</title>'
        f'<desc id="desc">{html.escape(purpose)}</desc>'
        '<rect width="100%" height="100%" rx="0" fill="#ffffff"/>'
        f'<text x="32" y="46" font-size="26" font-family="Arial" font-weight="700" '
        f'fill="#111827">{html.escape(title[:70])}</text>'
        f'<text x="32" y="78" font-size="16" font-family="Arial" fill="#4b5563">'
        f'{html.escape(purpose[:110])}</text>'
        f'<g>{"".join(rows)}</g></svg>'
    )
    path.write_text(svg, encoding="utf-8")


def _write_matmul_cell_svg(asset: VisualAsset, path: Path) -> None:
    title = asset.title or "How one C cell is produced"
    purpose = asset.purpose or "Show the row-column dot product behind one output cell."
    width = 920
    height = 360
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(title)}</title>
<desc id="desc">{html.escape(purpose)}</desc>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="32" y="44" font-family="Arial" font-size="26" font-weight="700" fill="#111827">{html.escape(title[:70])}</text>
<text x="32" y="74" font-family="Arial" font-size="15" fill="#4b5563">{html.escape(purpose[:120])}</text>
<g font-family="Arial" font-size="16" fill="#111827">
  <text x="70" y="118" font-weight="700">A row i</text>
  <text x="368" y="118" font-weight="700">B column j</text>
  <text x="676" y="118" font-weight="700">Output C[i,j]</text>
</g>
<g stroke="#111827" stroke-width="2" fill="none">
  <rect x="52" y="145" width="250" height="54" rx="0"/>
  <rect x="350" y="130" width="74" height="202" rx="0"/>
  <rect x="682" y="176" width="92" height="76" rx="0"/>
</g>
<g font-family="Arial" font-size="15" text-anchor="middle">
  <rect x="60" y="153" width="50" height="38" fill="#dbeafe"/><text x="85" y="177">a0</text>
  <rect x="118" y="153" width="50" height="38" fill="#dbeafe"/><text x="143" y="177">a1</text>
  <rect x="176" y="153" width="50" height="38" fill="#dbeafe"/><text x="201" y="177">...</text>
  <rect x="234" y="153" width="50" height="38" fill="#dbeafe"/><text x="259" y="177">ak</text>
  <rect x="362" y="140" width="50" height="38" fill="#dcfce7"/><text x="387" y="164">b0</text>
  <rect x="362" y="186" width="50" height="38" fill="#dcfce7"/><text x="387" y="210">b1</text>
  <rect x="362" y="232" width="50" height="38" fill="#dcfce7"/><text x="387" y="256">...</text>
  <rect x="362" y="278" width="50" height="38" fill="#dcfce7"/><text x="387" y="302">bk</text>
  <rect x="700" y="194" width="56" height="40" fill="#fef3c7"/><text x="728" y="219">sum</text>
</g>
<g stroke="#6b7280" stroke-width="2" fill="none" marker-end="url(#arrow)">
  <path d="M 302 172 C 370 172, 454 172, 520 198"/>
  <path d="M 424 232 C 468 232, 492 222, 520 208"/>
  <path d="M 590 204 C 626 204, 650 204, 682 204"/>
</g>
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#6b7280"/></marker></defs>
<g font-family="Arial" font-size="16" fill="#111827">
  <rect x="500" y="176" width="100" height="56" fill="#f3f4f6" stroke="#111827"/>
  <text x="550" y="198" text-anchor="middle">multiply</text>
  <text x="550" y="218" text-anchor="middle">and add</text>
  <text x="52" y="235" fill="#374151">Loop invariant: one accumulator belongs to one output cell.</text>
  <text x="52" y="264" fill="#374151">Validation: direct loop, trusted library, then benchmark the bottleneck.</text>
</g>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def _svg_labels(asset: VisualAsset) -> list[str]:
    raw = [part.strip() for part in re.split(r"\n|;", asset.content or "") if part.strip()]
    if raw:
        return raw[:6]
    return ["Define shapes and invariants", "Run the core computation", "Check failure modes", "Choose the next validation gate"]


def _unknown_evidence_errors(kind: str, object_id: str, ids: list[str], known: set[str]) -> list[str]:
    return [f"{kind}_unknown_evidence:{object_id}:{evidence_id}" for evidence_id in ids if evidence_id not in known]


def _cell(value: str) -> str:
    return " ".join((value or "").split()).replace("|", "\\|")


def _shortcode_name(shortcode: str) -> str:
    match = re.search(r"{{\s*[<%]\s*([A-Za-z0-9_-]+)", shortcode)
    return match.group(1) if match else ""
