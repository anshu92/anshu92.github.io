from tp_gqa_attention import Config, init_params, dense_forward, tp_forward, run
import torch

def test_forward_equivalence():
    r=run(11,'equivalence'); assert r['passed']

def test_multiple_seeds():
    for s in [11,17,23,29,31]: assert run(s,'equivalence')['passed']

def test_missing_output_reduce_fails():
    r=run(11,'missing_output_reduce')
    assert r['forward_max_abs_diff'] > 1e-2 and not r['passed']

def test_missing_input_gradient_reduce_fails_after_correct_forward():
    r=run(11,'missing_input_grad_reduce')
    assert r['forward_max_abs_diff'] < 1e-6
    assert r['input_grad_max_abs_diff'] > 1e-4
    assert not r['passed']

def test_shapes():
    c=Config(); p=init_params(c,1); x=torch.randn(c.seq,c.batch,c.hidden)
    assert dense_forward(x,p,c).shape==x.shape
    assert tp_forward(x,p,c).shape==x.shape
