from __future__ import annotations
import argparse, csv, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch import nn

RELATIONS = 5
COORDINATES = 4
POINTS = 2

def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def make_data(n:int, canvas:int, seed:int):
    rng=np.random.default_rng(seed)
    pts=rng.uniform(0, canvas, size=(n,4)).astype(np.float32)
    x1,y1,x2,y2=[pts[:,i] for i in range(4)]
    # Four-way relation determined by dominant normalized displacement.
    dx=(x2-x1)/canvas; dy=(y2-y1)/canvas
    direction=np.where(np.abs(dx)>=np.abs(dy), np.where(dx>=0,0,1), np.where(dy>=0,2,3)); labels=np.where(np.sqrt(dx*dx+dy*dy)<0.25,4,direction).astype(np.int64)
    return pts,labels

def encode(x:np.ndarray, canvas:int, kind:str):
    z=x.astype(np.float32)
    if kind=='raw_pixels': return z
    if kind=='normalized': return z/canvas
    if kind=='quantized_32': return np.floor(np.clip(z/canvas,0,0.999999)*32)/31.0
    if kind=='fourier':
        u=z/canvas
        feats=[u]
        for k in (1,2,4,8):
            feats += [np.sin(2*np.pi*k*u), np.cos(2*np.pi*k*u)]
        return np.concatenate(feats,axis=1).astype(np.float32)
    raise ValueError(kind)

def encode_tokens(x:np.ndarray, canvas:int, kind:str):
    z=encode(x,canvas,kind)
    per_coordinate=z.shape[1]//COORDINATES
    return z.reshape(len(z),POINTS,2*per_coordinate)

class MLP(nn.Module):
    def __init__(self,d:int):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,RELATIONS))
    def forward(self,x): return self.net(x)

class TinyCoordinateTransformer(nn.Module):
    def __init__(self, token_dim:int):
        super().__init__()
        self.project=nn.Linear(token_dim,32)
        self.cls=nn.Parameter(torch.zeros(1,1,32))
        self.coord_type=nn.Parameter(torch.zeros(1,POINTS,32))
        layer=nn.TransformerEncoderLayer(32,4,64,dropout=0.0,batch_first=True,norm_first=True,activation='gelu')
        self.transformer=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.head=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,RELATIONS))
    def forward(self,x):
        h=self.project(x)+self.coord_type
        cls=self.cls.expand(h.shape[0],-1,-1)
        h=self.transformer(torch.cat([cls,h],dim=1))
        return self.head(h[:,0])

def make_model(model_name:str, kind:str, x:np.ndarray):
    if model_name=='mlp':
        return MLP(encode(x[:1],256,kind).shape[1])
    if model_name=='transformer':
        return TinyCoordinateTransformer(encode_tokens(x[:1],256,kind).shape[2])
    raise ValueError(model_name)

def encode_for_model(x:np.ndarray, canvas:int, kind:str, model_name:str):
    if model_name=='mlp':
        return encode(x,canvas,kind)
    if model_name=='transformer':
        return encode_tokens(x,canvas,kind)
    raise ValueError(model_name)

def train_eval(seed:int, kind:str, model_name:str):
    seed_all(seed)
    xtr,ytr=make_data(2500,256,seed)
    model=make_model(model_name,kind,xtr)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-4)
    tx=torch.tensor(encode_for_model(xtr,256,kind,model_name)); ty=torch.tensor(ytr)
    steps=45 if model_name=='mlp' else 180
    for _ in range(steps):
        idx=torch.randint(0,len(tx),(256,))
        loss=nn.functional.cross_entropy(model(tx[idx]),ty[idx]); opt.zero_grad(); loss.backward(); opt.step()
    rows=[]
    scenarios=[('in_domain',256,seed+100),('scale_2x',512,seed+200),('scale_4x',1024,seed+300),('translated_crop',1024,seed+400)]
    for name,canvas,s in scenarios:
        x,y=make_data(1200,canvas,s)
        if name=='translated_crop':
            # concentrate points in upper-right half; relations are unchanged but absolute range shifts.
            x=0.5*canvas + 0.5*x
        with torch.no_grad(): pred=model(torch.tensor(encode_for_model(x,canvas,kind,model_name))).argmax(1).numpy()
        acc=float((pred==y).mean())
        rows.append({'seed':seed,'model':model_name,'encoding':kind,'scenario':name,'canvas':canvas,'accuracy':acc})
    return rows

def main(out:Path):
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in [3,7,11,19,23]:
        for model_name in ['mlp','transformer']:
            for kind in ['raw_pixels','normalized','quantized_32','fourier']:
                rows += train_eval(seed,kind,model_name)
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n'); w.writeheader(); w.writerows(rows)
    summary={}
    for model_name in sorted({r['model'] for r in rows}):
        summary[model_name]={}
        for kind in sorted({r['encoding'] for r in rows}):
            summary[model_name][kind]={}
            for sc in sorted({r['scenario'] for r in rows}):
                vals=[r['accuracy'] for r in rows if r['model']==model_name and r['encoding']==kind and r['scenario']==sc]
                summary[model_name][kind][sc]={'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1)),'values':vals}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=[]
    for model_name,encodings in summary.items():
        for kind,v in encodings.items():
            lines.append(f"{model_name}/{kind}: "+', '.join(f"{s}={m['mean']:.3f}±{m['std']:.3f}" for s,m in v.items()))
    (out/'raw-output.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,default=Path('data')); a=p.parse_args(); main(a.output_dir)
