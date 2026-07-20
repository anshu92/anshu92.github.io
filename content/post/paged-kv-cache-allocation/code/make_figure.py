#!/usr/bin/env python3
import argparse,json
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--results',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
r=json.loads(Path(a.results).read_text()); rows=[('Reserve max',r['reserve_max']),('Grow contiguous',r['grow_contiguous']),('Paged B=4',r['paged_4']),('Paged B=8',r['paged_8']),('Paged B=16',r['paged_16'])]
bars=[]
for i,(name,x) in enumerate(rows):
    y=65+i*50; res=x['peak_reserved']; waste=x['peak_waste']
    bars += [f'<text x="20" y="{y+18}" font-size="15">{name}</text>',f'<rect x="180" y="{y}" width="{res*8}" height="24" fill="#888"/>',f'<rect x="{180+(res-waste)*8}" y="{y}" width="{waste*8}" height="24" fill="#ddd"/>',f'<text x="{190+res*8}" y="{y+18}" font-size="13">peak reserved {res}, waste {waste}</text>']
svg='<svg xmlns="http://www.w3.org/2000/svg" width="900" height="330"><rect width="100%" height="100%" fill="white"/><text x="20" y="30" font-size="21" font-weight="bold">Same request trace, different allocation policies</text>'+''.join(bars)+'</svg>'
Path(a.output).write_text(svg)
