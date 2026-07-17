import torch
from loss_reduction import exact_global_loss, wrong_mean_of_rank_means, local_stats, IGNORE_INDEX

def make_logits(targets, confidence):
    logits=torch.zeros(len(targets),3,dtype=torch.float64,requires_grad=True)
    with torch.no_grad():
        for i,t in enumerate(targets):
            if t != IGNORE_INDEX:
                logits[i,t]=confidence[i]
    return logits

def test_mean_of_means_is_wrong_with_unequal_tokens():
    a_t=torch.tensor([0])
    b_t=torch.tensor([1,1,1])
    a=make_logits(a_t,[0.1]); b=make_logits(b_t,[3.0,3.0,3.0])
    shards=[(a,a_t),(b,b_t)]
    correct=exact_global_loss(shards)
    wrong=wrong_mean_of_rank_means(shards)
    assert not torch.allclose(correct,wrong,atol=1e-12)

def test_exact_loss_matches_concatenated_reference():
    t1=torch.tensor([0,1,IGNORE_INDEX,2]); t2=torch.tensor([2,2])
    l1=make_logits(t1,[1.2,0.3,0.0,2.0]); l2=make_logits(t2,[0.2,1.5])
    exact=exact_global_loss([(l1,t1),(l2,t2)])
    ref=torch.nn.functional.cross_entropy(torch.cat([l1,l2]),torch.cat([t1,t2]),ignore_index=IGNORE_INDEX,reduction='mean')
    assert torch.allclose(exact,ref,atol=1e-12)

def test_exact_gradient_matches_concatenated_reference():
    t1=torch.tensor([0,1,IGNORE_INDEX,2]); t2=torch.tensor([2,2])
    l1=make_logits(t1,[1.2,0.3,0.0,2.0]); l2=make_logits(t2,[0.2,1.5])
    exact=exact_global_loss([(l1,t1),(l2,t2)])
    exact.backward()
    g1=l1.grad.clone(); g2=l2.grad.clone()
    c=torch.cat([l1.detach(),l2.detach()]).requires_grad_(True)
    ref=torch.nn.functional.cross_entropy(c,torch.cat([t1,t2]),ignore_index=IGNORE_INDEX,reduction='mean')
    ref.backward()
    assert torch.allclose(torch.cat([g1,g2]),c.grad,atol=1e-12)

def test_zero_token_rank_is_safe_and_does_not_change_result():
    t1=torch.tensor([IGNORE_INDEX,IGNORE_INDEX]); t2=torch.tensor([1,2])
    l1=make_logits(t1,[0,0]); l2=make_logits(t2,[1.0,2.0])
    exact=exact_global_loss([(l1,t1),(l2,t2)])
    ref=torch.nn.functional.cross_entropy(l2,t2,reduction='mean')
    assert torch.isfinite(exact) and torch.allclose(exact,ref,atol=1e-12)

def test_all_tokens_ignored_returns_differentiable_zero():
    t=torch.tensor([IGNORE_INDEX,IGNORE_INDEX]); l=make_logits(t,[0,0])
    loss=exact_global_loss([(l,t)])
    assert loss.item()==0.0
    loss.backward()
    assert torch.all(l.grad==0)
