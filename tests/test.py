#! /usr/bin/env python

from os import path
dir_script = path.dirname(path.realpath(__file__))
import sys
sys.path.append(dir_script+'/../')
import pydssp
import numpy as np
import torch
import tqdm

testset_dir = dir_script+'/testset/TS50/'

# function for reading DSSP file
c3_convert = {' ':0, 'S':0, 'T':0, 'H':1, 'G':1, 'I':1, 'E':2, 'B':2}
def read_dssp_reference(dsspfile):
    with open(dsspfile, 'r') as f:
        lines = [l.rstrip() for l in f.readlines()]
        sw, orig = False, []
        for l in lines:
            if '!' in l: continue # skip chain-break
            if sw == True: orig.append(l[16])
            if l.startswith('  # '): sw = True
    c3_index = np.array([c3_convert[c8] for c8 in orig])
    return c3_index

# TS50 targets
listfile = testset_dir + 'list'
with open(listfile, 'r') as f:
    targets = [l.rstrip() for l in f.readlines()]

# pydssp calcuration with numpy
print(f"correlation check with numpy")
correlation_stack = []
for target in tqdm.tqdm(targets):
    dsspfile = testset_dir + '/dssp/' + target + '.dssp'
    pdbfile = testset_dir + '/pdb/' + target + '.pdb'    
    reference_idx = read_dssp_reference(dsspfile)
    coord = pydssp.read_pdbtext(open(pdbfile, 'r').read())
    pydssp_idx = pydssp.assign(coord, out_type='index')
    correlation = (reference_idx == pydssp_idx).mean()
    correlation_stack.append(correlation)

# check correlation
correlation_mean = np.array(correlation_stack).mean()
assert correlation_mean >0.97, 'Low correlation in TS50 testset'
print(f"correlation_mean = {correlation_mean:.5f} > 0.97 @ NumPy")

# pydssp calcuration with torch
print(f"correlation check with torch")
correlation_stack = []
for target in tqdm.tqdm(targets):
    dsspfile = testset_dir + '/dssp/' + target + '.dssp'
    pdbfile = testset_dir + '/pdb/' + target + '.pdb'    
    reference_idx = torch.Tensor(read_dssp_reference(dsspfile))
    coord = torch.Tensor(pydssp.read_pdbtext(open(pdbfile, 'r').read()))
    pydssp_idx = pydssp.assign(coord, out_type='index')
    correlation = (reference_idx == pydssp_idx).to(torch.float).mean()
    correlation_stack.append(correlation)

# check correlation
correlation_mean = np.array(correlation_stack).mean()
assert correlation_mean >0.97, 'Low correlation in TS50 testset'
print(f"correlation_mean = {correlation_mean:.5f} > 0.97 @ PyTorch")
