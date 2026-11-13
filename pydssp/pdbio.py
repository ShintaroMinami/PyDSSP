import numpy as np
from .pydssp_numpy import MAX_SEQ_GAP

atomnum = {' N  ':0, ' CA ': 1, ' C  ': 2, ' O  ': 3}
MAX_PEPTIDE_BOND_LENGTH = 2.5 # [Angstrom] C-N distance, same threshold as the original DSSP


def get_residue_index(coord: np.ndarray, resnum: np.ndarray) -> np.ndarray:
    """Make residue_index for assign() from PDB residue numbers and backbone geometry.

    A peptide bond (C-N distance) decides whether neighbors are connected, so numbering with
    insertion codes or skipped numbers (e.g. IMGT) is kept contiguous. Across a broken bond,
    the residue numbers give the number of missing residues, or a chain break if they don't increase.

    Args:
        coord: Backbone coordinates [L, 4 (N, CA, C, O), 3].
        resnum: PDB residue numbers [L].

    Returns:
        Residue index [L].
    """
    d_cn = np.linalg.norm(coord[1:, 0] - coord[:-1, 2], axis=-1)
    d_num = np.diff(resnum)
    step = np.where(d_cn <= MAX_PEPTIDE_BOND_LENGTH, 1, np.where(d_num > 1, d_num, MAX_SEQ_GAP))
    return resnum[0] + np.concatenate([[0], np.cumsum(step)]).astype(int)


def read_pdbtext_no_checking(pdbstring: str, return_sequence: bool=False):
    lines = pdbstring.split("\n")
    coords, atoms, resid_old, resname_old, sequence = [], None, None, None, []
    for l in lines:
        if l.startswith('ATOM'):
            iatom = atomnum.get(l[12:16], None)
            resid = l[21:26]
            resname = l[17:20]
            if resid != resid_old:
                if atoms is not None:
                    coords.append(atoms)
                    sequence.append(resname_old)
                atoms, resid_old, resname_old = [], resid, resname
            if iatom is not None:
                xyz = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                atoms.append(xyz)
    if atoms is not None:
        coords.append(atoms)
        sequence.append(resname_old)
    coords = np.array(coords)
    sequence = np.array(sequence)
    if return_sequence:
        return coords, sequence
    else:
        return coords


def read_pdbtext_with_checking(pdbstring: str, return_sequence: bool=False, return_residue_index: bool=False):
    """Read backbone coordinates from PDB text.

    Args:
        pdbstring: Content of a PDB file.
        return_sequence: Also return the residue names.
        return_residue_index: Also return residue_index for assign() (see get_residue_index).

    Returns:
        coords [L, 4, 3], followed by sequence [L] and residue_index [L] if requested.
    """
    lines = pdbstring.split("\n")
    coords, atoms, resid_old, resname_old, sequence, check, resnum = [], None, None, None, [], [], []
    for l in lines:
        if l.startswith('ATOM'):
            iatom = atomnum.get(l[12:16], None)
            resid = l[21:26]
            resname = l[17:20]
            if resid != resid_old:
                if atoms is not None:
                    coords.append(atoms)
                    sequence.append(resname_old)
                    check.append(atom_check)
                atoms, resid_old, resname_old, atom_check = [], resid, resname, []
                resnum.append(int(l[22:26]))
            if iatom is not None:
                xyz = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                atoms.append(xyz)
                atom_check.append(iatom)
    if atoms is not None:
        coords.append(atoms)
        sequence.append(resname_old)
        check.append(atom_check)
    coords = np.array(coords)
    sequence = np.array(sequence)
    # check
    assert len(coords.shape) == 3, "Some required atoms [N,CA,C,O] are missing in the input PDB file"
    check = np.array(check)
    assert np.all(check[:,0]==0), "Order of PDB line may be broken. It's required to be N->CA->C->O w/o any duplicate or lack"
    assert np.all(check[:,1]==1), "Order of PDB line may be broken. It's required to be N->CA->C->O w/o any duplicate or lack"
    assert np.all(check[:,2]==2), "Order of PDB line may be broken. It's required to be N->CA->C->O w/o any duplicate or lack"
    assert np.all(check[:,3]==3), "Order of PDB line may be broken. It's required to be N->CA->C->O w/o any duplicate or lack"
    # output
    outputs = [coords]
    if return_sequence:
        outputs.append(sequence)
    if return_residue_index:
        outputs.append(get_residue_index(coords, np.array(resnum)))
    return tuple(outputs) if len(outputs) > 1 else coords
