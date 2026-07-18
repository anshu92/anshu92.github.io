from __future__ import annotations
import argparse, json, math, platform
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class Config:
    seq: int = 5
    batch: int = 2
    hidden: int = 16
    query_heads: int = 4
    kv_heads: int = 2
    head_dim: int = 4
    world_size: int = 2
    eps: float = 1e-6


def rms_norm(x, weight, eps):
    """
    x ── square/mean ── rsqrt ── scale by x ── scale by weight ── normalized x
    """
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def apply_rope(x):
    """
    x[..., :half] ─┐
                   ├─ rotate by position/frequency phase ── RoPE(x)
    x[..., half:] ─┘
    """
    # x: [S,B,H,D], D even
    s = x.shape[0]
    half = x.shape[-1] // 2
    pos = torch.arange(s, dtype=x.dtype, device=x.device).view(s,1,1,1)
    freq = torch.arange(half, dtype=x.dtype, device=x.device).view(1,1,1,half)
    theta = pos / (10000.0 ** (freq / max(1, half)))
    c, si = theta.cos(), theta.sin()
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a*c - b*si, a*si + b*c], dim=-1)


def repeat_kv(x, groups):
    """
    KV heads [S,B,Hkv,D] ── repeat each head `groups` times ── [S,B,Hq,D]
    """
    return x.repeat_interleave(groups, dim=2)


def attention(q,k,v):
    """
    q,k ── scaled dot product ── causal mask ── softmax ─┐
                                                        ├─ weighted sum ─ context
    v ──────────────────────────────────────────────────┘
    """
    # [S,B,H,D]
    scores = torch.einsum('sbhd,tbhd->bhst', q, k) / math.sqrt(q.shape[-1])
    mask = torch.triu(torch.ones(q.shape[0], q.shape[0], dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask.view(1,1,*mask.shape), float('-inf'))
    probs = scores.softmax(-1)
    return torch.einsum('bhst,tbhd->sbhd', probs, v)


def init_params(cfg: Config, seed: int):
    """
    Config + seed ── deterministic tensors ── norm, wq, wk, wv, wo
    """
    g = torch.Generator().manual_seed(seed)
    def rand(*shape): return torch.randn(*shape, generator=g) * 0.15
    return {
        'norm': torch.ones(cfg.hidden),
        'wq': rand(cfg.query_heads*cfg.head_dim, cfg.hidden),
        'wk': rand(cfg.kv_heads*cfg.head_dim, cfg.hidden),
        'wv': rand(cfg.kv_heads*cfg.head_dim, cfg.hidden),
        'wo': rand(cfg.hidden, cfg.query_heads*cfg.head_dim),
    }


def clone_trainable(params):
    """
    frozen parameter dict ── clone/detach/requires_grad ── trainable parameter dict
    """
    return {k: v.clone().detach().requires_grad_(True) for k,v in params.items()}


def dense_forward(x, p, cfg):
    """
    x ── RMSNorm ── Q,K,V ── RoPE ── repeat KV ── attention ── output proj ─┐
    └──────────────────────────────────────── residual ─────────────────── (+)
    """
    residual = x
    n = rms_norm(x, p['norm'], cfg.eps)
    q = F.linear(n, p['wq']).view(cfg.seq,cfg.batch,cfg.query_heads,cfg.head_dim)
    k = F.linear(n, p['wk']).view(cfg.seq,cfg.batch,cfg.kv_heads,cfg.head_dim)
    v = F.linear(n, p['wv']).view(cfg.seq,cfg.batch,cfg.kv_heads,cfg.head_dim)
    q, k = apply_rope(q), apply_rope(k)
    groups = cfg.query_heads // cfg.kv_heads
    ctx = attention(q, repeat_kv(k, groups), repeat_kv(v, groups))
    return residual + F.linear(ctx.reshape(cfg.seq,cfg.batch,-1), p['wo'])


def shard_params(p, cfg):
    """
    dense params ── split complete Q/KV head groups ── rank-local parameter shards
    """
    q_per = cfg.query_heads // cfg.world_size
    kv_per = cfg.kv_heads // cfg.world_size
    shards=[]
    for r in range(cfg.world_size):
        q0,q1=r*q_per,(r+1)*q_per
        k0,k1=r*kv_per,(r+1)*kv_per
        shards.append({
            'norm': p['norm'],
            'wq': p['wq'][q0*cfg.head_dim:q1*cfg.head_dim],
            'wk': p['wk'][k0*cfg.head_dim:k1*cfg.head_dim],
            'wv': p['wv'][k0*cfg.head_dim:k1*cfg.head_dim],
            'wo': p['wo'][:, q0*cfg.head_dim:q1*cfg.head_dim],
        })
    return shards


def tp_forward(x, p, cfg, reduce_output=True, reduce_input_grad=True):
    """
                     ┌─ rank 0: local Q/KV group ─ attention ─ output partial ─┐
    x ── RMSNorm ────┤                                                           ├─ sum? ─ (+ residual)
                     └─ rank 1: local Q/KV group ─ attention ─ output partial ─┘
    """
    residual=x
    n=rms_norm(x,p['norm'],cfg.eps)
    partials=[]
    q_per=cfg.query_heads//cfg.world_size
    kv_per=cfg.kv_heads//cfg.world_size
    local_groups=q_per//kv_per
    for rank, s in enumerate(shard_params(p,cfg)):
        # Each TP rank sees the same normalized input in the forward pass.
        # Detaching rank 1 simulates forgetting to sum its contribution to dX.
        n_local = n if (reduce_input_grad or rank == 0) else n.detach()
        q=F.linear(n_local,s['wq']).view(cfg.seq,cfg.batch,q_per,cfg.head_dim)
        k=F.linear(n_local,s['wk']).view(cfg.seq,cfg.batch,kv_per,cfg.head_dim)
        v=F.linear(n_local,s['wv']).view(cfg.seq,cfg.batch,kv_per,cfg.head_dim)
        q,k=apply_rope(q),apply_rope(k)
        ctx=attention(q,repeat_kv(k,local_groups),repeat_kv(v,local_groups))
        partials.append(F.linear(ctx.reshape(cfg.seq,cfg.batch,-1),s['wo']))
    out = sum(partials) if reduce_output else partials[0]
    return residual + out


def run(seed=11, mode='equivalence'):
    """
    seed + mode ── dense step and TP step ── compare forward, grads, and SGD update
    """
    cfg=Config(); torch.manual_seed(seed)
    x0=torch.randn(cfg.seq,cfg.batch,cfg.hidden)
    target=torch.randn_like(x0)
    base=init_params(cfg,seed+100)
    pd=clone_trainable(base); pt=clone_trainable(base)
    xd=x0.clone().requires_grad_(True); xt=x0.clone().requires_grad_(True)
    yd=dense_forward(xd,pd,cfg)
    yt=tp_forward(
        xt, pt, cfg,
        reduce_output=(mode!='missing_output_reduce'),
        reduce_input_grad=(mode!='missing_input_grad_reduce'),
    )
    ld=F.mse_loss(yd,target); lt=F.mse_loss(yt,target)
    ld.backward(); lt.backward()
    metrics={
      'seed':seed,'mode':mode,
      'forward_max_abs_diff':(yd-yt).abs().max().detach().item(),
      'loss_abs_diff':(ld-lt).abs().detach().item(),
      'input_grad_max_abs_diff':(xd.grad-xt.grad).abs().max().detach().item(),
      'parameter_grad_max_abs_diff':max((pd[k].grad-pt[k].grad).abs().max().detach().item() for k in pd),
    }
    lr=0.05
    with torch.no_grad():
      for k in pd:
        pd[k]-=lr*pd[k].grad; pt[k]-=lr*pt[k].grad
    metrics['post_step_weight_max_abs_diff']=max((pd[k]-pt[k]).abs().max().detach().item() for k in pd)
    metrics['passed']= all(metrics[k] < 1e-6 for k in ['forward_max_abs_diff','loss_abs_diff','input_grad_max_abs_diff','parameter_grad_max_abs_diff','post_step_weight_max_abs_diff'])
    return metrics


def inspect():
    """
    Config/runtime ── printable fixture metadata and tensor-shape summary
    """
    cfg=Config()
    return {
      'python':platform.python_version(),'pytorch':torch.__version__,
      'config':cfg.__dict__,
      'dense_shapes':{'x':[cfg.seq,cfg.batch,cfg.hidden], 'q':[cfg.seq,cfg.batch,cfg.query_heads,cfg.head_dim], 'k_v':[cfg.seq,cfg.batch,cfg.kv_heads,cfg.head_dim]},
      'rank_local':{'query_heads':cfg.query_heads//cfg.world_size,'kv_heads':cfg.kv_heads//cfg.world_size},
      'rule':'keep complete query/KV groups on each rank; sum output-projection partials'
    }

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['inspect','equivalence','missing_output_reduce','missing_input_grad_reduce','matrix'],default='equivalence'); ap.add_argument('--seed',type=int,default=11); a=ap.parse_args()
    if a.mode=='inspect': out=inspect()
    elif a.mode=='matrix': out=(
        [run(s,'equivalence') for s in [11,17,23,29,31]]
        + [run(s,'missing_output_reduce') for s in [11,17,23,29,31]]
        + [run(s,'missing_input_grad_reduce') for s in [11,17,23,29,31]]
    )
    else: out=run(a.seed,a.mode)
    print(json.dumps(out,indent=2))
