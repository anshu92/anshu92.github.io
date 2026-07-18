from pathlib import Path
import json
import re
import yaml

root = Path(__file__).resolve().parents[1]
md = (root / "index.md").read_text()
assert md.startswith("---\n")
assert md.count("```") % 2 == 0
assert "\x08" not in md and "\x0c" not in md
assert md.count("<!-- BEGIN AUTO-GENERATED TERMINAL OUTPUT:") == 5
for p in re.findall(r"\]\(([^)]+)\)", md):
    if p.startswith(("http://", "https://", "#")):
        continue
    if p.startswith("/images/"):
        assert (root.parents[2] / "static" / p.lstrip("/")).exists(), p
        continue
    assert (root / p).exists(), p
manifest = yaml.safe_load((root / "asset-manifest.yaml").read_text())
for asset in manifest["assets"]:
    assert (root / asset["path"]).exists(), asset["path"]
for required in [
    "pre-RMSNorm",
    "SwiGLU",
    "CopyToTensorParallelRegion",
    "ReduceFromTensorParallelRegion",
    '"forward_max_abs_diff": 1.2665987014770508e-07',
    '"forward_max_abs_diff": 1.1021764278411865',
    '"input_grad_max_abs_diff": 0.13864503800868988',
    "4 passed in 26.94s",
    "Megatron-style concept",
]:
    assert required in md, required
for svg in (root / "figures").glob("*.svg"):
    text = svg.read_text()
    assert text.startswith("<svg") and "</svg>" in text
print(json.dumps({
    "status": "PASS",
    "markdown": True,
    "links": True,
    "yaml": True,
    "latex_controls": True,
    "svgs": len(list((root / "figures").glob("*.svg"))),
    "terminal_outputs": 5,
}, indent=2))
