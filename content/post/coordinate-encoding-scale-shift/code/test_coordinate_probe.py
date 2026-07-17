from pathlib import Path
import csv, json, subprocess, sys
import numpy as np
import torch
from coordinate_probe import TinyCoordinateTransformer, encode, encode_tokens, make_data

def test_normalized_scale_equivalence():
    x=np.array([[32,64,128,192]],dtype=np.float32)
    assert np.allclose(encode(x,256,'normalized'),encode(x*4,1024,'normalized'))

def test_raw_pixels_break_scale_equivalence():
    x=np.array([[32,64,128,192]],dtype=np.float32)
    assert not np.allclose(encode(x,256,'raw_pixels'),encode(x*4,1024,'raw_pixels'))

def test_quantization_bounded():
    x=np.array([[0,255,128,64]],dtype=np.float32)
    z=encode(x,256,'quantized_32'); assert z.min()>=0 and z.max()<=1

def test_transformer_token_encoder_preserves_point_slots():
    x=np.array([[0,64,128,255]],dtype=np.float32)
    z=encode_tokens(x,256,'fourier')
    assert z.shape==(1,2,18)
    model=TinyCoordinateTransformer(z.shape[-1])
    assert tuple(model(torch.tensor(z)).shape)==(1,5)

def test_labels_invariant_to_uniform_scaling():
    x,y=make_data(1000,256,9)
    x2=x*4
    dx=(x2[:,2]-x2[:,0])/1024; dy=(x2[:,3]-x2[:,1])/1024
    direction=np.where(np.abs(dx)>=np.abs(dy),np.where(dx>=0,0,1),np.where(dy>=0,2,3)); y2=np.where(np.sqrt(dx*dx+dy*dy)<0.25,4,direction)
    assert np.array_equal(y,y2)

def test_retained_results_have_all_runs():
    p=Path(__file__).parents[1]/'data/results.csv'
    rows=list(csv.DictReader(p.open()))
    assert len(rows)==5*2*4*4
    assert {r['model'] for r in rows}=={'mlp','transformer'}
    assert {r['encoding'] for r in rows}=={'raw_pixels','normalized','quantized_32','fourier'}
