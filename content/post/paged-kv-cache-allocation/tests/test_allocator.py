#!/usr/bin/env python3
import json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from kv_allocator import simulate
spec=json.loads((Path(__file__).resolve().parents[1]/'data'/'scenario.json').read_text())
r=simulate(spec)
assert len(r['reserve_max']['rejected'])>0
assert len(r['grow_contiguous']['growth_failures'])>0
assert r['paged_4']['rejected']==[] and r['paged_4']['growth_failures']==[]
assert r['paged_8']['rejected'] or r['paged_8']['growth_failures']
assert r['paged_16']['rejected'] or r['paged_16']['growth_failures']
assert r['paged_4']['peak_waste']<=r['paged_16']['peak_waste']
print('PASS: allocator policy, fragmentation counterexample, and block-size assertions')
