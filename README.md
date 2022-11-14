# PyDSSP
A simplified implementation of DSSP algorithm for PyTorch and NumPy

# What's this?
DSSP (Dictionary of Secondary Structure of Protein) is a popular algorithm for assigning secondary structure of protein backbone structure. [<a href="https://onlinelibrary.wiley.com/doi/abs/10.1002/bip.360221211">
Wolfgang Kabsch, and Christian Sander (1983)</a>] This repository is a python implementation of DSSP algorithm that simplifies some parts of the algorithm.

# General Info
- It's NOT a complete implementation of the original DSSP, as some parts have been simplified (some more details [here](#differences-from-the-original-dssp)). However, an average of over 97% of secondary structure determinations agree with the original.
- The algorithm used to identify hydrogen bonded residue pairs is exactly the same as the original DSSP algorithm, but is extended to output the hydrogen-bond-pair-matrix as continuous values in the range [0,1].
- With the continuous variable extension above, the hydrogen-bond-pair-matrix is differentiable with torch.Tensor as input.

# Install
## install through PyPi
``` bash
pip install pydssp
```
## install by git clone
``` bash
git clone https://github.com/ShintaroMinami/PyDSSP.git
cd PyDSSSP
python setup.py install
```

# How to use
## To use pydssp script
If you have already installed pydssp, you should be able to use pydssp command.
``` bash
pydssp  input_01.pdb input_02.pdb ... input_N.pdb -o output.result
```
The output.result will be a text format, looking like follows,
``` bash
-EEEEE-E--EEEEEE---EEEE-HHHH--EEEE--------- input_01.pdb
-HHHHHHHHHHHHHH----HHHHHHHHHHHHHHHHHHH--- input_02.pdb
-EEEE-----EEEE----EEEE--E---EEE-----EEE-EEE-- input_03.pdb
...
```

## To use as python module
### Import & test coordinates
``` python
# Import
import torch
import pydssp

# Sample coordinates
batch, length, atoms, xyz = 10, 100, 4, 3
## atoms should be 4 (N, CA, C, O) or 5 (N, CA, C, O, H)
coord = torch.randn([batch, length, atom, xyz]) # batch-dim is optional
```

### To get hydrogen-bond matrix: ```pydssp.get_hbond_map()```
``` python
hbond_matrix = pydssp.get_hbond_map(coord)

print(hbond_matrix.shape) # should be (batch, length, length)
```
- For hbond_matrix[b, i, j], index 'i' is for donner (N-H) and 'j' is for acceptor (C=O), respectively
- The output matrix consists of constant values in the range [0,1], which is defined as follows.

$HbondMat(i,j) = (1+\sin((-0.5-E(i,j)-margin)/margin*\pi/2))/2$

Here $E$ is the electrostatic energy defined by (Kabsch and Sander 1983) and $margin(=1.0)$ is introduced to control smoothness.

### To get secondary structure assignment: ```pydssp.assign()```
``` python
dssp = pydssp.assign(coord, out_type='c3')
## output is batched np.ndarray of C3 annotation, like ['-', 'H', 'H', ..., 'E', '-']

# To get secondary str. as index
dssp = pydssp.assign(coord, out_type='index')
## 0: loop,  1: alpha-helix,  2: beta-strand

# To get secondary str. as onehot representation
dssp = pydssp.assign(coord, out_type='onehot')
## dim-0: loop,  dim-1: alpha-helix,  dim-2: beta-strand
```

# Differences from the original DSSP
This implementation was simplified from the original DSSP algorithm. The differences from the original DSSP are as follows
- The implementation omitted β-bulge annotation, so β-bulge is determined as a loop instead of β-strand.
- Parameters for adding hydrogen atoms are slightly different from the original DSSP, which may cause small differences in hydrogen bond annotation.
- Only support C3 ('-', 'H', and 'E') type assignment instead of C8 type (B, E, G, H, I, S, T, and ' ').

Although the above simplifications, the C3 type annotation still matches with the original DSSP for more than 97% of residues on average.

## Reference
@article{kabsch1983dictionary,
  title={Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features},
  author={Kabsch, Wolfgang and Sander, Christian},
  journal={Biopolymers: Original Research on Biomolecules},
  volume={22},
  number={12},
  pages={2577--2637},
  year={1983},
  publisher={Wiley Online Library}
}
