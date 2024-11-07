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
    return_e: bool=False
    ) -> Union[torch.Tensor, np.ndarray]:
    assert type(coord) in [torch.Tensor, np.ndarray], 'Input type must be torch.Tensor or np.ndarray'
    if type(coord) == torch.Tensor:
        return get_hbond_map_torch(coord, donor_mask=donor_mask, return_e=return_e)
    elif type(coord) == np.ndarray:
        return get_hbond_map_numpy(coord, donor_mask=donor_mask, return_e=return_e)


def assign(
    coord: Union[torch.Tensor, np.ndarray],
    donor_mask: Union[torch.Tensor, np.ndarray, list]=None,
    out_type: Literal['onehot', 'index', 'c3'] = 'c3'
    ) -> np.ndarray:
    assert type(coord) in [torch.Tensor, np.ndarray], "Input type must be torch.Tensor or np.ndarray"
    assert out_type in ['onehot', 'index', 'c3'], "Output type must be 'onehot', 'index', or 'c3'"
    # main calcuration
    if type(coord) == torch.Tensor:
        onehot = assign_torch(coord, donor_mask=donor_mask)
    elif type(coord) == np.ndarray:
        onehot = assign_numpy(coord, donor_mask=donor_mask)
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
