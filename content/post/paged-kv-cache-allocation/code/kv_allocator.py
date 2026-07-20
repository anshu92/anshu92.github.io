#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from dataclasses import dataclass
from pathlib import Path
@dataclass
class Request:
    rid: str; arrival: int; prompt: int; max_new: int; actual_new: int
class Contiguous:
    def __init__(self, capacity, reserve_max): self.capacity,self.reserve_max,self.intervals=capacity,reserve_max,{}
    def _holes(self):
        used=sorted((s,s+n) for s,n,_ in self.intervals.values()); holes=[]; cur=0
        for s,e in used:
            if cur<s: holes.append((cur,s-cur))
            cur=max(cur,e)
        if cur<self.capacity: holes.append((cur,self.capacity-cur))
        return holes
    def admit(self,r):
        need=r.prompt+r.max_new if self.reserve_max else r.prompt
        for s,n in self._holes():
            if n>=need: self.intervals[r.rid]=(s,need,r.prompt); return True
        return False
    def grow(self,rid):
        s,n,live=self.intervals[rid]
        if self.reserve_max: self.intervals[rid]=(s,n,live+1); return True
        end=s+n; next_start=min([x for k,(x,_,_) in self.intervals.items() if k!=rid and x>=end] or [self.capacity])
        if next_start-end>=1: self.intervals[rid]=(s,n+1,live+1); return True
        return False
    def free(self,rid): self.intervals.pop(rid,None)
    def stats(self):
        live=sum(v[2] for v in self.intervals.values()); reserved=sum(v[1] for v in self.intervals.values()); holes=self._holes()
        return {'live':live,'reserved':reserved,'waste':reserved-live,'free':self.capacity-reserved,'largest_hole':max([n for _,n in holes] or [0])}
class Paged:
    def __init__(self,capacity,block): self.capacity,self.block,self.free_blocks,self.alloc,self.live=capacity,block,list(range(capacity//block)),{},{}
    def _ensure(self,rid,tokens):
        need=(tokens+self.block-1)//self.block; have=len(self.alloc.get(rid,[])); extra=need-have
        if extra>len(self.free_blocks): return False
        self.alloc.setdefault(rid,[]).extend(self.free_blocks[:extra]); del self.free_blocks[:extra]; self.live[rid]=tokens; return True
    def admit(self,r): return self._ensure(r.rid,r.prompt)
    def grow(self,rid): return self._ensure(rid,self.live[rid]+1)
    def free(self,rid): self.free_blocks.extend(self.alloc.pop(rid,[])); self.free_blocks.sort(); self.live.pop(rid,None)
    def stats(self):
        live=sum(self.live.values()); reserved=sum(len(x) for x in self.alloc.values())*self.block
        return {'live':live,'reserved':reserved,'waste':reserved-live,'free':self.capacity-reserved,'largest_hole':None}
def simulate(spec):
    reqs=[Request(**x) for x in spec['requests']]; policies={'reserve_max':Contiguous(spec['capacity'],True),'grow_contiguous':Contiguous(spec['capacity'],False)}
    for b in spec['block_sizes']: policies[f'paged_{b}']=Paged(spec['capacity'],b)
    out={}
    for name,p in policies.items():
        active={}; rejected=[]; growth_failures=[]; trace=[]
        for t in range(spec['steps']):
            for r in reqs:
                if r.arrival==t:
                    if p.admit(r): active[r.rid]=[r,0]
                    else: rejected.append(r.rid)
            done=[]
            for rid,(r,g) in list(active.items()):
                if g<r.actual_new:
                    if p.grow(rid): active[rid][1]+=1
                    else: growth_failures.append({'time':t,'request':rid,'stats':p.stats()}); done.append(rid)
                else: done.append(rid)
            for rid in done: p.free(rid); active.pop(rid,None)
            trace.append({'time':t,**p.stats(),'active':sorted(active)})
        out[name]={'rejected':rejected,'growth_failures':growth_failures,'trace':trace,'peak_reserved':max(x['reserved'] for x in trace),'peak_waste':max(x['waste'] for x in trace)}
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--scenario',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    result=simulate(json.loads(Path(a.scenario).read_text())); Path(a.output).write_text(json.dumps(result,indent=2)+chr(10))
    print(json.dumps({k:{'rejected':v['rejected'],'growth_failures':len(v['growth_failures']),'peak_reserved':v['peak_reserved'],'peak_waste':v['peak_waste']} for k,v in result.items()},indent=2))
if __name__=='__main__': main()
