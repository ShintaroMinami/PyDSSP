#! /usr/bin/env python

import numpy as np
from einops import repeat, rearrange

CONST_Q1Q2 = 0.084
CONST_F = 332
DEFAULT_CUTOFF = -0.5
DEFAULT_MARGIN = 1.0
# Gaps longer than this are shortened to it internally. DSSP patterns look at most 5 residues ahead
# (turn of 5), so any gap of 6 or more gives the same result.
MAX_SEQ_GAP = 6


def _unfold(a: np.ndarray, window: int, axis: int):
    idx = np.arange(window)[:, None] + np.arange(a.shape[axis] - window + 1)[None, :]
    unfolded = np.take(a, idx, axis=axis)
    return  np.moveaxis(unfolded, axis-1, -1)


def _check_input(coord):
    org_shape = coord.shape
    assert (len(org_shape)==3) or (len(org_shape)==4), "Shape of input tensor should be [batch, L, atom, xyz] or [L, atom, xyz]"
    coord = repeat(coord, '... -> b ...', b=1) if len(org_shape)==3 else coord
    return coord, org_shape


def _get_seq_position(residue_index, l: int) -> np.ndarray:
    """Convert residue_index ([L] or [B, L]) into increasing sequence positions ([1, L] or [B, L]).

    e.g. residue_index [10, 11, 14, 1, 2] -> [0, 1, 4, 10, 11]
    (a difference > 1 keeps the missing residues, a difference <= 0 is a chain break)
    """
    if residue_index is None:
        return np.arange(l)[None]
    residue_index = np.asarray(residue_index).astype(int)
    assert residue_index.shape[-1] == l, "Length of residue_index must match the number of residues"
    step = np.diff(residue_index.reshape(-1, l), axis=-1)
    step = np.where(step > 0, np.minimum(step, MAX_SEQ_GAP), MAX_SEQ_GAP)
    return np.pad(np.cumsum(step, axis=-1), [[0, 0], [1, 0]])


def _get_hydrogen_atom_position(coord: np.ndarray) -> np.ndarray:
    # A little bit lazy (but should be OK) definition of H position here.
    vec_cn = coord[:,1:,0] - coord[:,:-1,2]
    vec_cn = vec_cn / np.linalg.norm(vec_cn, axis=-1, keepdims=True)
    vec_can = coord[:,1:,0] - coord[:,1:,1]
    vec_can = vec_can / np.linalg.norm(vec_can, axis=-1, keepdims=True)
    vec_nh = vec_cn + vec_can
    vec_nh = vec_nh / np.linalg.norm(vec_nh, axis=-1, keepdims=True)
    return coord[:,1:,0] + 1.01 * vec_nh


def get_hbond_map(
    coord: np.ndarray,
    donor_mask: np.ndarray=None,
    cutoff: float=DEFAULT_CUTOFF,
    margin: float=DEFAULT_MARGIN,
    return_e: bool=False,
    residue_index: np.ndarray=None,
    ) -> np.ndarray:
    # check input
    coord, org_shape = _check_input(coord)
    b, l, a, _ = coord.shape
    # add pseudo-H atom if not available
    assert (a==4) or (a==5), "Number of atoms should be 4 (N,CA,C,O) or 5 (N,CA,C,O,H)"
    h = coord[:,1:,4] if a == 5 else _get_hydrogen_atom_position(coord)
    # distance matrix
    nmap = repeat(coord[:,1:,0], '... m c -> ... m n c', n=l-1)
    hmap = repeat(h, '... m c -> ... m n c', n=l-1)
    cmap = repeat(coord[:,0:-1,2], '... n c -> ... m n c', m=l-1)
    omap = repeat(coord[:,0:-1,3], '... n c -> ... m n c', m=l-1)
    d_on = np.linalg.norm(omap - nmap, axis=-1)
    d_ch = np.linalg.norm(cmap - hmap, axis=-1)
    d_oh = np.linalg.norm(omap - hmap, axis=-1)
    d_cn = np.linalg.norm(cmap - nmap, axis=-1)
    # electrostatic interaction energy
    e = np.pad(CONST_Q1Q2 * (1./d_on + 1./d_ch - 1./d_oh - 1./d_cn)*CONST_F, [[0,0],[1,0],[0,1]])
    if return_e: return e
    # mask for local pairs (i,i), (i,i+1), (i,i+2) in sequence space
    pos = _get_seq_position(residue_index, l)
    sep = pos[:, :, None] - pos[:, None, :] # donor - acceptor
    local_mask = (sep < 0) | (sep > 2)
    # mask for donor H absence (Proline)
    donor_mask = np.array(donor_mask).astype(float) if donor_mask is not None else np.ones(l, dtype=float)
    donor_mask = donor_mask.reshape(-1, l)
    # mask for donor H that cannot be estimated (preceding residue is missing)
    if a == 4:
        donor_mask = donor_mask * np.pad(np.diff(pos, axis=-1) == 1, [[0, 0], [1, 0]], constant_values=True)
    # hydrogen bond map (continuous value extension of original definition)
    hbond_map = np.clip(cutoff - margin - e, a_min=-margin, a_max=margin)
    hbond_map = (np.sin(hbond_map/margin*np.pi/2)+1.)/2
    hbond_map = hbond_map * local_mask
    hbond_map = hbond_map * donor_mask[:, :, None]
    # return h-bond map
    hbond_map = np.squeeze(hbond_map, axis=0) if len(org_shape)==3 else hbond_map
    return hbond_map


def assign(coord: np.ndarray, donor_mask: np.ndarray=None, residue_index: np.ndarray=None) -> np.ndarray:
    # check input
    coord, org_shape = _check_input(coord)
    b, l = coord.shape[:2]
    # get hydrogen bond map
    hbmap = get_hbond_map(coord, donor_mask=donor_mask, residue_index=residue_index)
    # place residues at their sequence positions so that no pattern is formed across gaps
    if residue_index is not None:
        pos = _get_seq_position(residue_index, l)
        hbmap_seq = np.zeros((b, pos.max()+1, pos.max()+1), dtype=hbmap.dtype)
        hbmap_seq[np.arange(b)[:, None, None], pos[:, :, None], pos[:, None, :]] = hbmap
        hbmap = hbmap_seq
    hbmap = rearrange(hbmap, '... l1 l2 -> ... l2 l1') # convert into "i:C=O, j:N-H" form
    # identify turn 3, 4, 5
    turn3 = np.diagonal(hbmap, axis1=-2, axis2=-1, offset=3) > 0.
    turn4 = np.diagonal(hbmap, axis1=-2, axis2=-1, offset=4) > 0.
    turn5 = np.diagonal(hbmap, axis1=-2, axis2=-1, offset=5) > 0.
    # assignment of helical sses
    h3 = np.pad(turn3[:,:-1] * turn3[:,1:], [[0,0],[1,3]])
    h4 = np.pad(turn4[:,:-1] * turn4[:,1:], [[0,0],[1,4]])
    h5 = np.pad(turn5[:,:-1] * turn5[:,1:], [[0,0],[1,5]])
    # helix4 first
    helix4 = h4 + np.roll(h4, 1, 1) + np.roll(h4, 2, 1) + np.roll(h4, 3, 1)
    h3 = h3 * ~np.roll(helix4, -1, 1) * ~helix4 # helix4 is higher prioritized
    h5 = h5 * ~np.roll(helix4, -1, 1) * ~helix4 # helix4 is higher prioritized
    helix3 = h3 + np.roll(h3, 1, 1) + np.roll(h3, 2, 1)
    helix5 = h5 + np.roll(h5, 1, 1) + np.roll(h5, 2, 1) + np.roll(h5, 3, 1) + np.roll(h5, 4, 1)
    # identify bridge
    unfoldmap = _unfold(_unfold(hbmap, 3, -2), 3, -2) > 0.
    unfoldmap_rev = rearrange(unfoldmap, 'b l1 l2 ... -> b l2 l1 ...')
    p_bridge = (unfoldmap[:,:,:,0,1] * unfoldmap_rev[:,:,:,1,2]) + (unfoldmap_rev[:,:,:,0,1] * unfoldmap[:,:,:,1,2])
    p_bridge = np.pad(p_bridge, [[0,0],[1,1],[1,1]])
    a_bridge = (unfoldmap[:,:,:,1,1] * unfoldmap_rev[:,:,:,1,1]) + (unfoldmap[:,:,:,0,2] * unfoldmap_rev[:,:,:,0,2])
    a_bridge = np.pad(a_bridge, [[0,0],[1,1],[1,1]])
    # ladder
    ladder = (p_bridge + a_bridge).sum(-1) > 0
    # H, E, L of C3
    helix = (helix3 + helix4 + helix5) > 0
    strand = ladder
    loop = (~helix * ~strand)
    onehot = np.stack([loop, helix, strand], axis=-1)
    if residue_index is not None:
        onehot = onehot[np.arange(b)[:, None], pos] # back to input residues
    onehot = rearrange(onehot, '1 ... -> ...') if len(org_shape)==3 else onehot
    return onehot
