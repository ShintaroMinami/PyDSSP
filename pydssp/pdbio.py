import numpy as np

atomnum = {' N  ':0, ' CA ': 1, ' C  ': 2, ' O  ': 3}

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


def read_pdbtext_with_checking(pdbstring: str, return_sequence: bool=False):
    lines = pdbstring.split("\n")
    coords, atoms, resid_old, resname_old, sequence, check = [], None, None, None, [], []
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
    if return_sequence:
        return coords, sequence
    else:
        return coords
