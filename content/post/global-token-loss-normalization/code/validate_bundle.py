from pathlib import Path
import re, yaml, xml.etree.ElementTree as ET, json, sys
root=Path(__file__).parents[1]
md=(root/'index.md').read_text()
errors=[]
for i,ch in enumerate(md):
    if ord(ch)<32 and ch not in '\n\r\t': errors.append(f'control:{i}:{ord(ch)}')
if md.count('\\[')!=md.count('\\]'): errors.append('unbalanced display math')
if not md.startswith('---\n'): errors.append('frontmatter missing')
else:
    parts=md.split('---',2)
    try: yaml.safe_load(parts[1])
    except Exception as e: errors.append(f'yaml:{e}')
for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',md):
    if link.startswith(('http://','https://','#')): continue
    if not (root/link).exists(): errors.append(f'missing link:{link}')
for svg in [root/'cover.svg',*sorted((root/'figures').glob('*.svg'))]:
    try: ET.parse(svg)
    except Exception as e: errors.append(f'svg:{svg.name}:{e}')
status={'status':'PASS' if not errors else 'FAIL','errors':errors,'svg_count':1+len(list((root/'figures').glob('*.svg')))}
(root/'data'/'bundle-validation.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status,indent=2))
sys.exit(1 if errors else 0)
