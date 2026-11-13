from typing import Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal # for Python3.6/3.7 users

import torch
import numpy as np
from .pydssp_numpy import (
    get_hbond_map as get_hbond_map_numpy,
    assign as assign_numpy
)
from .pydssp_torch import (
    get_hbond_map as get_hbond_map_torch,
    assign as assign_torch
)

C3_ALPHABET = np.array(['-', 'H', 'E'])


def get_hbond_map(
    coord: Union[torch.Tensor, np.ndarray],
    donor_mask: Union[torch.Tensor, np.ndarray]=None,
    return_e: bool=False,
    residue_index: Union[torch.Tensor, np.ndarray, list]=None,
    ) -> Union[torch.Tensor, np.ndarray]:
    """Calculate the hydrogen bond map.

    Args:
        coord: Backbone coordinates [(batch,) L, 4 or 5 (N, CA, C, O, (H)), 3].
        donor_mask: Residues able to donate N-H (e.g. False for proline) [(batch,) L].
        return_e: Return the electrostatic energy map instead of the hydrogen bond map.
        residue_index: Sequence position of each residue (e.g. label_seq_id) [(batch,) L].
            Residues whose indices differ by 1 are treated as connected; a larger difference
            means missing residues, and a difference <= 0 means a chain break.
            If None, consecutive entries in coord are treated as consecutive residues.

    Returns:
        Hydrogen bond map [(batch,) L, L] in [0, 1], where [i, j] is for donor i (N-H)
        and acceptor j (C=O).
    """
    assert type(coord) in [torch.Tensor, np.ndarray], 'Input type must be torch.Tensor or np.ndarray'
    if type(coord) == torch.Tensor:
        return get_hbond_map_torch(coord, donor_mask=donor_mask, return_e=return_e, residue_index=residue_index)
    elif type(coord) == np.ndarray:
        return get_hbond_map_numpy(coord, donor_mask=donor_mask, return_e=return_e, residue_index=residue_index)


def assign(
    coord: Union[torch.Tensor, np.ndarray],
    donor_mask: Union[torch.Tensor, np.ndarray, list]=None,
    out_type: Literal['onehot', 'index', 'c3'] = 'c3',
    residue_index: Union[torch.Tensor, np.ndarray, list]=None,
    ) -> np.ndarray:
    """Assign secondary structure (C3: loop, helix, strand).

    Args:
        coord: Backbone coordinates [(batch,) L, 4 or 5 (N, CA, C, O, (H)), 3].
        donor_mask: Residues able to donate N-H (e.g. False for proline) [(batch,) L].
        out_type: 'c3' ('-', 'H', 'E'), 'index' (0, 1, 2), or 'onehot'.
        residue_index: Sequence position of each residue (e.g. label_seq_id) [(batch,) L].
            Residues whose indices differ by 1 are treated as connected; a larger difference
            means missing residues, and a difference <= 0 means a chain break.
            If None, consecutive entries in coord are treated as consecutive residues.

    Returns:
        Secondary structure of each residue [(batch,) L] (plus [3] for 'onehot').
    """
    assert type(coord) in [torch.Tensor, np.ndarray], "Input type must be torch.Tensor or np.ndarray"
    assert out_type in ['onehot', 'index', 'c3'], "Output type must be 'onehot', 'index', or 'c3'"
    # main calcuration
    if type(coord) == torch.Tensor:
        onehot = assign_torch(coord, donor_mask=donor_mask, residue_index=residue_index)
    elif type(coord) == np.ndarray:
        onehot = assign_numpy(coord, donor_mask=donor_mask, residue_index=residue_index)
    # output one-hot
    if out_type == 'onehot':
        return onehot
    # output index
    index = torch.argmax(onehot.to(torch.float), dim=-1) if type(onehot) == torch.Tensor else np.argmax(onehot, axis=-1)
    if out_type == 'index':
        return index
    # output c3
    c3 = C3_ALPHABET[index.cpu().numpy()] if type(index) == torch.Tensor else C3_ALPHABET[index]
    return c3
