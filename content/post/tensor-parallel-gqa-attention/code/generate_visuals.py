from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)

def svg(title, subtitle, blocks, arrows, path, width=1200, height=650):
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#0b1020"/>',
           f'<text x="60" y="70" fill="#f8fafc" font-family="Arial" font-size="34" font-weight="700">{title}</text>',
           f'<text x="60" y="108" fill="#94a3b8" font-family="Arial" font-size="19">{subtitle}</text>']
    for b in blocks:
        x,y,w,h,label,detail=b
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="#172033" stroke="#64748b" stroke-width="2"/>')
        parts.append(f'<text x="{x+20}" y="{y+38}" fill="#f8fafc" font-family="Arial" font-size="22" font-weight="700">{label}</text>')
        for i,line in enumerate(detail.split('\n')):
            parts.append(f'<text x="{x+20}" y="{y+70+i*25}" fill="#cbd5e1" font-family="Arial" font-size="17">{line}</text>')
    for x1,y1,x2,y2,label in arrows:
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#38bdf8" stroke-width="4"/>')
        parts.append(f'<polygon points="{x2},{y2} {x2-14},{y2-8} {x2-14},{y2+8}" fill="#38bdf8"/>')
        if label:
            parts.append(f'<text x="{(x1+x2)//2}" y="{(y1+y2)//2-12}" text-anchor="middle" fill="#7dd3fc" font-family="Arial" font-size="16">{label}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts))

svg('Tensor-parallel GQA', 'Partition complete query/KV groups, then reduce output partials.',
    [(70,180,290,220,'Dense GQA','4 query heads\n2 KV heads\n2 queries per KV'),
     (455,155,300,180,'Rank 0','Q heads 0–1\nKV head 0\nlocal attention'),
     (455,385,300,180,'Rank 1','Q heads 2–3\nKV head 1\nlocal attention'),
     (850,260,280,220,'Dense output','sum rank-local\noutput projections\nthen residual add')],
    [(360,290,455,245,'group 0'),(360,290,455,475,'group 1'),(755,245,850,335,'partial 0'),(755,475,850,385,'partial 1')], ROOT/'cover.svg')

svg('First transformation: preserve semantic head groups', 'Each rank owns complete query heads and the KV head they reuse.',
    [(70,180,300,260,'Dense head topology','Q0 Q1 → KV0\nQ2 Q3 → KV1\nhead_dim = 4'),
     (480,160,280,180,'Rank 0','Q0, Q1\nK0, V0\nrepeat KV locally'),
     (480,390,280,180,'Rank 1','Q2, Q3\nK1, V1\nrepeat KV locally')],
    [(370,270,480,240,'slice by group'),(370,330,480,470,'slice by group')], FIG/'figure-01-head-groups.svg')

svg('Second transformation: row-parallel output projection', 'Local contexts produce full-width partial outputs that must be summed.',
    [(70,180,280,180,'Rank 0 context','[S,B,2,D]\nWo columns 0–7'),
     (70,400,280,180,'Rank 1 context','[S,B,2,D]\nWo columns 8–15'),
     (460,180,280,180,'Partial output 0','[S,B,H]\nnot independently valid'),
     (460,400,280,180,'Partial output 1','[S,B,H]\nnot independently valid'),
     (850,285,280,180,'All-reduce sum','dense attention output\nthen residual add')],
    [(350,270,460,270,'local GEMM'),(350,490,460,490,'local GEMM'),(740,270,850,350,'sum'),(740,490,850,400,'sum')], FIG/'figure-02-equivalence.svg')

svg('Two bugs, two first-failure signals', 'Forward and backward communication must be tested independently.',
    [(70,170,320,180,'Missing output reduce','first failure: output\nforward error ≈ 0.701\nloss changes immediately'),
     (70,400,320,180,'Missing input-grad reduce','forward still matches\nfirst failure: dX\ninput-grad error ≈ 0.011'),
     (520,285,580,180,'Required test hierarchy','forward comparison catches row-parallel bug\nbackward comparison catches replicated-input bug\noptimizer-step comparison catches state divergence')],
    [(390,260,520,335,'forward test'),(390,490,520,415,'backward test')], FIG/'figure-03-failure-signatures.svg')
