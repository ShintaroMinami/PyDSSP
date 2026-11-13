"""Tests for missing residues given by residue_index (issue #8)."""
from __future__ import annotations

import sys
from os import path

import numpy as np
import pytest
import torch

sys.path.insert(0, path.join(path.dirname(path.realpath(__file__)), '..'))
import pydssp  # noqa: E402

TESTSET_DIR = path.join(path.dirname(path.realpath(__file__)), 'testset')
PDB_DIR = path.join(TESTSET_DIR, 'TS50', 'pdb')
TARGETS = ['1ahsA', '1bvyF', '4dkcA']  # beta-rich, mixed, helix-rich


def load(target: str) -> tuple[str, np.ndarray, np.ndarray]:
    with open(path.join(PDB_DIR, f'{target}.pdb')) as f:
        text = f.read()
    coord, sequence = pydssp.read_pdbtext(text, return_sequence=True)
    return text, coord, sequence != 'PRO'


def assign_padded(coord: np.ndarray, donor: np.ndarray, pos: np.ndarray, length: int) -> np.ndarray:
    """Reference: NaN at missing positions so that array index == sequence position."""
    padded = np.full((length, 4, 3), np.nan)
    padded[pos] = coord
    padded_donor = np.zeros(length, dtype=bool)
    padded_donor[pos] = donor
    return pydssp.assign(padded, donor_mask=padded_donor)[pos]


def gapped_positions(length: int, n: int = 30, seed: int = 0) -> list[np.ndarray]:
    """Observed positions after deleting a stretch of residues (including gaps > MAX_SEQ_GAP)."""
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        gap = int(rng.choice([1, 2, 3, 6, 20]))
        start = int(rng.integers(3, length - gap - 3))
        cases.append(np.setdiff1d(np.arange(length), np.arange(start, start + gap)))
    return cases


@pytest.mark.parametrize('to_input', [np.asarray, torch.tensor])
def test_contiguous_index_is_unchanged(to_input):
    _, coord, donor = load(TARGETS[0])
    ri = np.arange(len(coord)) + 10
    x = to_input(coord)
    assert (pydssp.assign(x, donor, residue_index=ri) == pydssp.assign(x, donor)).all()
    hb, hb_ri = pydssp.get_hbond_map(x, donor), pydssp.get_hbond_map(x, donor, residue_index=ri)
    assert np.array_equal(np.asarray(hb), np.asarray(hb_ri))


@pytest.mark.parametrize('target', TARGETS)
def test_gap_matches_padded_reference(target):
    _, full, full_donor = load(target)
    n_affected = 0
    for pos in gapped_positions(len(full)):
        coord, donor = full[pos], full_donor[pos]
        ref = assign_padded(coord, donor, pos, len(full))
        assert (pydssp.assign(coord, donor, residue_index=pos) == ref).all()
        assert (pydssp.assign(torch.tensor(coord), donor, residue_index=torch.tensor(pos)) == ref).all()
        n_affected += (pydssp.assign(coord, donor) != ref).any()
    assert n_affected > 0, 'No case is affected by the gap; the test does not exercise issue #8'


def test_chain_break_by_index_restart():
    _, c1, d1 = load(TARGETS[0])
    _, c2, d2 = load(TARGETS[1])
    c2 = c2 + 1000.  # no inter-chain hydrogen bond
    ri = np.concatenate([np.arange(len(c1)), np.arange(len(c2))])
    joint = pydssp.assign(np.concatenate([c1, c2]), np.concatenate([d1, d2]), residue_index=ri)
    assert (joint == np.concatenate([pydssp.assign(c1, d1), pydssp.assign(c2, d2)])).all()


def test_batch_with_different_gaps():
    _, full, _ = load(TARGETS[0])
    cases = gapped_positions(len(full), n=4, seed=1)
    pos = np.stack([p[:min(map(len, cases))] for p in cases])  # same length L for every sample
    expected = np.stack([pydssp.assign(full[p], residue_index=p) for p in pos])
    assert (pydssp.assign(full[pos], residue_index=pos) == expected).all()
    assert (pydssp.assign(torch.tensor(full[pos]), residue_index=torch.tensor(pos)) == expected).all()


def test_gradient_is_finite_with_gap():
    _, full, _ = load(TARGETS[0])
    pos = gapped_positions(len(full))[0]
    coord = torch.tensor(full[pos], requires_grad=True)
    pydssp.get_hbond_map(coord, residue_index=torch.tensor(pos)).sum().backward()
    assert torch.isfinite(coord.grad).all()


def test_residue_index_from_pdb():
    text, full, full_donor = load(TARGETS[0])
    atoms = [l for l in text.splitlines() if l.startswith('ATOM')]
    resnums = sorted({int(l[22:26]) for l in atoms})
    # missing residues (broken peptide bond) -> gap in residue_index
    missing = set(resnums[40:43])
    gapped = '\n'.join(l for l in atoms if int(l[22:26]) not in missing)
    coord, ri = pydssp.read_pdbtext(gapped, return_residue_index=True)
    pos = np.array([i for i, n in enumerate(resnums) if n not in missing])
    assert np.array_equal(np.diff(ri), np.diff(pos))
    assert (pydssp.assign(coord, full_donor[pos], residue_index=ri)
            == assign_padded(coord, full_donor[pos], pos, len(full))).all()
    # skipped numbers with intact peptide bonds (e.g. IMGT numbering) -> contiguous
    shift = resnums[60]
    renumbered = '\n'.join(l[:22] + f'{int(l[22:26]) + 10 * (int(l[22:26]) > shift):4d}' + l[26:] for l in atoms)
    _, ri = pydssp.read_pdbtext(renumbered, return_residue_index=True)
    assert (np.diff(ri) == 1).all()


@pytest.mark.parametrize('to_input', [np.asarray, torch.tensor])
def test_issue8_example_3s9d(to_input):
    """3S9D chain B: residues 88-89 (label_seq_id 82-83) are missing, so 87 and 91 must not be i, i+2."""
    with open(path.join(TESTSET_DIR, '3S9D', '3S9D_B.pdb')) as f:
        text = f.read()
    coord, sequence, ri = pydssp.read_pdbtext(text, return_sequence=True, return_residue_index=True)
    resnum = np.array([int(l[22:26]) for l in text.splitlines() if l.startswith('ATOM') and l[12:16] == ' CA '])
    donor = sequence != 'PRO'
    assert len(coord) == 180
    assert np.array_equal(ri, resnum)  # all gaps found from PDB numbering and geometry
    ref = assign_padded(coord, donor, resnum - resnum[0], resnum[-1] - resnum[0] + 1)
    ss = pydssp.assign(to_input(coord), donor, residue_index=ri)
    assert (ss == ref).all()
    i87, i91 = np.flatnonzero(resnum == 87)[0], np.flatnonzero(resnum == 91)[0]
    assert ss[i87] == 'E' and ss[i91] == 'E'  # '-' without residue_index (issue #8)
