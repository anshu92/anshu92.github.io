from __future__ import annotations

import os
import unittest

from blogpipe import draft as draft_mod
from blogpipe.formats import FORMATS
from blogpipe.models import EditorialBrief, EvidenceBundle, Item
from blogpipe.voice import ANCHORS, DEFAULT_ANCHOR, get_anchor


def _bundle() -> EvidenceBundle:
    return EvidenceBundle(
        primary=Item(
            id="primary",
            title="A Sample Paper on Layer Selection",
            url="https://arxiv.org/abs/0000.0001",
            abstract="The paper proposes a layer-selection method.",
            source="arxiv",
        )
    )


def _brief(anchor) -> EditorialBrief:
    return EditorialBrief(
        voice_guide=anchor.voice_guide,
        opener_hook=anchor.opener_hook,
        format_name="deep_dive",
        format_rationale="test",
    )


class VoiceAnchorTests(unittest.TestCase):
    def test_default_is_d_hybrid(self) -> None:
        self.assertEqual(DEFAULT_ANCHOR.key, "D")

    def test_get_anchor_explicit_key(self) -> None:
        for k in ("A", "B", "C", "D"):
            self.assertEqual(get_anchor(k).key, k)

    def test_get_anchor_unknown_falls_back_to_default(self) -> None:
        self.assertEqual(get_anchor("Z").key, DEFAULT_ANCHOR.key)

    def test_env_overrides_default(self) -> None:
        prev = os.environ.get("BLOGPIPE_VOICE_ANCHOR")
        try:
            os.environ["BLOGPIPE_VOICE_ANCHOR"] = "B"
            self.assertEqual(get_anchor().key, "B")
        finally:
            if prev is None:
                os.environ.pop("BLOGPIPE_VOICE_ANCHOR", None)
            else:
                os.environ["BLOGPIPE_VOICE_ANCHOR"] = prev

    def test_all_anchors_have_exemplar_block_and_voice_guide(self) -> None:
        for key, anchor in ANCHORS.items():
            self.assertTrue(
                anchor.voice_guide and anchor.voice_guide.strip(),
                f"voice_guide missing for {key}",
            )
            self.assertTrue(
                "WRITING EXEMPLAR" in anchor.exemplar_block,
                f"exemplar block missing for {key}",
            )
            self.assertTrue(
                "WEAK" in anchor.exemplar_block,
                f"weak exemplar missing for {key}",
            )

    def test_voice_guide_lands_in_writer_system_prompt(self) -> None:
        anchor = get_anchor("D")
        bundle = _bundle()
        brief = _brief(anchor)
        fmt = FORMATS["deep_dive"]
        system, _user = draft_mod.build_prompt(bundle, brief, fmt)
        # voice_guide is rendered into the system prompt verbatim
        self.assertIn(anchor.voice_guide.split(".")[0], system)

    def test_writing_exemplar_block_is_present_in_system_prompt(self) -> None:
        anchor = get_anchor("D")
        bundle = _bundle()
        brief = _brief(anchor)
        fmt = FORMATS["deep_dive"]
        system, _user = draft_mod.build_prompt(bundle, brief, fmt)
        self.assertIn("WRITING EXEMPLAR", system)


if __name__ == "__main__":
    unittest.main()
