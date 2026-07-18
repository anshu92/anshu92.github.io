from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; F=ROOT/'figures'; F.mkdir(exist_ok=True)

def svg(name, body, w=1200,h=620):
    text=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<style>text{{font-family:Inter,Arial,sans-serif;fill:#172033}} .t{{font-size:34px;font-weight:700}} .s{{font-size:22px}} .m{{font-size:18px}} .box{{fill:#f4f6fa;stroke:#67748a;stroke-width:2}} .rank0{{fill:#e8f0ff;stroke:#4169a1;stroke-width:2}} .rank1{{fill:#fff0e6;stroke:#a65f2c;stroke-width:2}} .arrow{{stroke:#26354a;stroke-width:3;marker-end:url(#a)}} </style>
<defs><marker id="a" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#26354a"/></marker></defs>{body}</svg>'''
    (F/name).write_text(text)

svg('cover.svg','''<text x="60" y="70" class="t">Megatron tensor parallelism, rebuilt inside a Transformer block</text>
<text x="60" y="110" class="s">pre-RMSNorm → SwiGLU → row-parallel down projection → residual</text>
<rect x="55" y="205" width="180" height="90" rx="16" class="box"/><text x="95" y="258" class="s">Tokens X</text>
<line x1="235" y1="250" x2="315" y2="250" class="arrow"/>
<rect x="325" y="205" width="190" height="90" rx="16" class="box"/><text x="365" y="258" class="s">RMSNorm</text>
<line x1="515" y1="250" x2="590" y2="190" class="arrow"/><line x1="515" y1="250" x2="590" y2="380" class="arrow"/>
<rect x="600" y="135" width="260" height="125" rx="16" class="rank0"/><text x="655" y="185" class="s">Rank 0</text><text x="625" y="220" class="m">local gate/up → SiLU × up → partial down</text>
<rect x="600" y="330" width="260" height="125" rx="16" class="rank1"/><text x="655" y="380" class="s">Rank 1</text><text x="625" y="415" class="m">local gate/up → SiLU × up → partial down</text>
<line x1="860" y1="200" x2="940" y2="285" class="arrow"/><line x1="860" y1="390" x2="940" y2="315" class="arrow"/>
<rect x="950" y="245" width="200" height="110" rx="16" class="box"/><text x="990" y="290" class="s">all-reduce</text><text x="980" y="325" class="m">then add residual</text>''')

svg('figure-01-sharding.svg','''<text x="60" y="65" class="t">The gated expansion is column parallel; the down projection is row parallel</text>
<text x="60" y="105" class="s">Each rank owns matched gate, value, and down-projection slices of the intermediate width.</text>
<rect x="60" y="175" width="230" height="250" rx="16" class="box"/><text x="100" y="225" class="s">Dense fused gate/up</text><text x="125" y="265" class="m">Wgu [24, 8]</text><text x="110" y="315" class="m">gate [12, 8]</text><text x="118" y="350" class="m">up [12, 8]</text>
<line x1="290" y1="280" x2="355" y2="220" class="arrow"/><line x1="290" y1="320" x2="355" y2="390" class="arrow"/>
<rect x="365" y="150" width="290" height="145" rx="16" class="rank0"/><text x="420" y="200" class="s">Rank 0 local SwiGLU</text><text x="405" y="240" class="m">gate/up weight [12, 8]</text><text x="430" y="270" class="m">activation [3, 2, 6]</text>
<rect x="365" y="330" width="290" height="145" rx="16" class="rank1"/><text x="420" y="380" class="s">Rank 1 local SwiGLU</text><text x="405" y="420" class="m">gate/up weight [12, 8]</text><text x="430" y="450" class="m">activation [3, 2, 6]</text>
<line x1="655" y1="220" x2="735" y2="220" class="arrow"/><line x1="655" y1="400" x2="735" y2="400" class="arrow"/>
<rect x="745" y="150" width="250" height="145" rx="16" class="rank0"/><text x="800" y="200" class="s">Rank 0 down slice</text><text x="820" y="240" class="m">Wd [8, 6]</text><text x="790" y="270" class="m">partial [3, 2, 8]</text>
<rect x="745" y="330" width="250" height="145" rx="16" class="rank1"/><text x="800" y="380" class="s">Rank 1 down slice</text><text x="820" y="420" class="m">Wd [8, 6]</text><text x="790" y="450" class="m">partial [3, 2, 8]</text>
<line x1="995" y1="220" x2="1090" y2="300" class="arrow"/><line x1="995" y1="400" x2="1090" y2="325" class="arrow"/><text x="1050" y="285" class="m">SUM</text>''')

svg('figure-02-collectives.svg','''<text x="60" y="65" class="t">One missing collective breaks the forward pass; the other breaks only backward</text>
<text x="60" y="105" class="s">The first changed signal tells you which boundary is wrong.</text>
<rect x="75" y="170" width="450" height="135" rx="16" class="rank0"/><text x="115" y="220" class="s">Missing forward all-reduce</text><text x="115" y="258" class="m">first mismatch: output = 1.102</text><text x="115" y="288" class="m">loss and every gradient diverge afterward</text>
<rect x="75" y="355" width="450" height="135" rx="16" class="rank1"/><text x="115" y="405" class="s">Missing backward all-reduce</text><text x="115" y="443" class="m">forward still matches: 1.27e-07</text><text x="115" y="473" class="m">first mismatch: input/RMSNorm gradients</text>
<rect x="675" y="190" width="430" height="260" rx="16" class="box"/><text x="725" y="245" class="s">Debugging rule</text><text x="715" y="295" class="m">1. compare output</text><text x="715" y="335" class="m">2. compare local parameter gradients</text><text x="715" y="375" class="m">3. compare gradient entering replicated path</text><text x="715" y="415" class="m">4. compare post-step weights</text>''')
