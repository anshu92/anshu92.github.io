import csv, random, statistics
from pathlib import Path

def trial(seed, ranks=8):
    rng=random.Random(seed)
    counts=[rng.randint(1,64) for _ in range(ranks)]
    means=[rng.uniform(0.5,4.0) for _ in range(ranks)]
    correct=sum(c*m for c,m in zip(counts,means))/sum(counts)
    wrong=sum(means)/len(means)
    return counts,means,correct,wrong,abs(correct-wrong)
rows=[]
for seed in range(100):
    c,m,correct,wrong,error=trial(seed)
    rows.append({'seed':seed,'min_tokens':min(c),'max_tokens':max(c),'correct_loss':correct,'mean_of_means':wrong,'absolute_error':error})
out=Path(__file__).parents[1]/'data'/'simulation.csv'
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
summary={'runs':len(rows),'mean_absolute_error':statistics.mean(r['absolute_error'] for r in rows),'max_absolute_error':max(r['absolute_error'] for r in rows)}
(Path(__file__).parents[1]/'data'/'simulation-summary.json').write_text(__import__('json').dumps(summary,indent=2))
print(summary)
