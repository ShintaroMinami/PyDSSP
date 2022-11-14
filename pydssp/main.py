from typing import Union, Literal
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

CONST_Q1Q2 = 0.084
CONST_F = 332
DEFAULT_CUTOFF = -0.5
DEFAULT_MARGIN = 1.0
C3_ALPHABET = ['-', 'H', 'E']


def get_hbond_map(
    coord: Union[torch.Tensor, np.ndarray],
    return_e: bool=False
    ) -> Union[torch.Tensor, np.ndarray]:
    assert type(coord) in [torch.Tensor, np.ndarray], 'Input type must be torch.Tensor or np.ndarray'
    if type(coord) == torch.Tensor:
        return get_hbond_map_torch(coord, return_e=return_e)
    elif type(coord) == np.ndarray:
        return get_hbond_map_numpy(coord, return_e=return_e)


def assign(
    coord: Union[torch.Tensor, np.ndarray],
    out_type: Literal['onehot', 'index', 'c3'] = 'c3'
    ) -> np.ndarray:
    assert type(coord) in [torch.Tensor, np.ndarray], "Input type must be torch.Tensor or np.ndarray"
    assert out_type in ['onehot', 'index', 'c3'], "Output type must be 'onehot', 'index', or 'c3'"
    # main calcuration
    if type(coord) == torch.Tensor:
        onehot = assign_torch(coord).cpu().numpy()
    elif type(coord) == np.ndarray:
        onehot = assign_numpy(coord)
    # output one-hot
    if out_type == 'onehot':
        return onehot
    # output index
    index = np.argmax(onehot, axis=-1)
    if out_type == 'index':
        return index
    # output c3
    c3 = np.array([C3_ALPHABET[i] for i in index])
    return c3
