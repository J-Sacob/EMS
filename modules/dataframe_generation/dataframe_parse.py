import pandas as pd
from tqdm import tqdm
from EMS.utils.periodic_table import Get_periodic_table

def make_atoms_df(ems_list, write=False, format="pickle"):
    p_table = Get_periodic_table()

    # construct dataframes
    # atoms has: molecule_name, atom, labeled atom,
    molecule_name = []  # molecule name
    atom_index = []  # atom index
    typestr = []  # atom type (string)
    typeint = []  # atom type (integer)
    x = []  # x coordinate
    y = []  # y coordinate
    z = []  # z coordinate
    conns = []
    atom_props = []
    smiles = []
    for propname in ems_list[0].atom_properties.keys():
        atom_props.append([])

    pbar = tqdm(ems_list, desc="Constructing atom dictionary", leave=False)

    m = -1
    for ems in pbar:
        m += 1
        # Add atom values to lists
        for t, type in enumerate(ems.type):
            molecule_name.append(ems.id)
            atom_index.append(t)
            typestr.append(p_table[type])
            typeint.append(type)
            x.append(ems.xyz[t][0])
            y.append(ems.xyz[t][1])
            z.append(ems.xyz[t][2])
            conns.append(ems.conn[t])
            smiles.append(ems.mol_properties["SMILES"])
            for p, prop in enumerate(ems.atom_properties.keys()):
                atom_props[p].append(ems.atom_properties[prop][t])

            # for p, prop in enumerate(ems.atom_properties.keys()):
            #     if prop == 'shift' and atom_list == 'all':
            #         atom_props[p].append(ems.atom_properties[prop][t])
            #     elif prop == 'shift' and atom_list != 'all':
            #         if p_table[type] in atom_list:
            #             atom_props[p].append(ems.atom_properties[prop][t])
            #         else:
            #             atom_props[p].append(0.0)
            #     else:
            #         atom_props[p].append(ems.atom_properties[prop][t])

    # Construct dataframe
    atoms = {
        "molecule_name": molecule_name,
        "atom_index": atom_index,
        "typestr": typestr,
        "typeint": typeint,
        "x": x,
        "y": y,
        "z": z,
        "conn": conns,
        "SMILES": smiles,
    }
    for p, propname in enumerate(ems.atom_properties.keys()):
        atoms[propname] = atom_props[p]

    atoms = pd.DataFrame(atoms)

    pbar.close()

    atoms.astype(
        {
            "molecule_name": "category",
            "atom_index": "Int16",
            "typestr": "category",
            "typeint": "Int8",
            "x": "Float32",
            "y": "Float32",
            "z": "Float32",
            "SMILES": "category",
        }
    )

    if write:
        if format == "csv":
            atoms.to_csv(f"{write}/atoms.csv")
        elif format == "pickle":
            atoms.to_pickle(f"{write}/atoms.pkl")
        elif format == "parquet":
            atoms.to_parquet(f"{write}/atoms.parquet")

    else:
        return atoms


def make_pairs_df(ems_list, write=False, format="pickle", max_pathlen=6):
    # construct dataframe for pairs in molecule
    # only atom pairs with bonds < max_pathlen are included

    molecule_name = []  # molecule name
    atom_index_0 = []  # atom index for atom 1
    atom_index_1 = []  # atom index for atom 2
    dist = []  # distance between atoms
    path_len = []  # number of pairs between atoms (shortest path)
    pair_props = []
    bond_existence = []
    aromatic_bond_order = []

    for propname in ems_list[0].pair_properties.keys():
        pair_props.append([])

    pbar = tqdm(ems_list, desc="Constructing pairs dictionary", leave=False)

    m = -1
    for ems in pbar:
        m += 1

        for t, type in enumerate(ems.type):
            for t2, type2 in enumerate(ems.type):
                # Add pair values to lists
                if ems.path_topology[t][t2] > max_pathlen:
                    continue
                molecule_name.append(ems.id)
                atom_index_0.append(t)
                atom_index_1.append(t2)
                dist.append(ems.path_distance[t][t2])
                path_len.append(int(ems.path_topology[t][t2]))
                bond_existence.append(ems.adj[t][t2])
                aromatic_bond_order.append(ems.aromatic_conn[t][t2])
                for p, prop in enumerate(ems.pair_properties.keys()):
                        pair_props[p].append(ems.pair_properties[prop][t][t2])

                # for p, prop in enumerate(ems.pair_properties.keys()):
                #     if prop == 'coupling' and coupling_list == 'all':
                #         pair_props[p].append(ems.pair_properties[prop][t][t2])
                #     elif prop == 'coupling' and coupling_list != 'all':
                #         if ems.pair_properties['nmr_types'][t][t2] in coupling_list:
                #             pair_props[p].append(ems.pair_properties[prop][t][t2])
                #         else:
                #             pair_props[p].append(0.0)
                #     else:
                #         pair_props[p].append(ems.pair_properties[prop][t][t2])

    # Construct dataframe
    pairs = {
        "molecule_name": molecule_name,
        "atom_index_0": atom_index_0,
        "atom_index_1": atom_index_1,
        "distance": dist,
        "path_len": path_len,
        "bond_existence": bond_existence,
        "bond_order": aromatic_bond_order,
    }
    for p, propname in enumerate(ems.pair_properties.keys()):
        pairs[propname] = pair_props[p]

    pairs = pd.DataFrame(pairs)
    pairs.loc[pairs['bond_order'] == 1.5, 'bond_order'] = 4.0
    pairs['bond_order'] = pairs['bond_order'].astype(int)

    pbar.close()

    if write:
        if format == "csv":
            pairs.to_csv(f"{write}/pairs.csv")
        elif format == "pickle":
            pairs.to_pickle(f"{write}/pairs.pkl")
        elif format == "parquet":
            pairs.to_parquet(f"{write}/pairs.parquet")
    else:
        return pairs