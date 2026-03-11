import logging
import sys
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from EMS.modules.properties.structure.rdkit_structure_read import sdf_to_rdmol
from EMS.modules.properties.structure.rdkit_structure_read import xyz_to_rdmol
from EMS.modules.properties.structure.rdkit_structure_read import mol2_to_rdmol
from EMS.modules.properties.structure.rdkit_structure_read import dataframe_to_rdmol
from EMS.modules.properties.structure.rdkit_structure_read import dataframe_to_rdmol_bond_order
from EMS.modules.properties.structure.rdkit_structure_read import cif_to_rdmol
from EMS.modules.properties.structure.rdkit_structure_read import structure_arrays_to_rdmol_NoConn
from EMS.modules.properties.nmr.nmr_read import nmr_read_sdf
from EMS.modules.properties.nmr.nmr_read import nmr_read_rdmol
from EMS.modules.properties.nmr.nmr_read import nmr_read_df
from EMS.modules.properties.nmr.nmr_read import nmr_read_gaussian
from EMS.modules.properties.nmr.nmr_read import nmr_read_cif
from EMS.modules.properties.nmr.nmr_ops import scale_chemical_shifts
from EMS.modules.comp_chem.gaussian.gaussian_read import gaussian_read_structure


########### Set up the logger system ###########
logger = logging.getLogger(__name__)
stdout = logging.StreamHandler(stream = sys.stdout)
formatter = logging.Formatter("%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s >>> %(message)s")
stdout.setFormatter(formatter)
logger.addHandler(stdout)
logger.setLevel(logging.WARNING)
########### Set up the logger system ###########


def assign_rdmol_name(rdmol, mol_id=None, extra_name=None):
    '''
    This function assigns a name to the RDKit molecule object (rdmol) based on the following order:
    (1) Extra names (if provided)
    (2) The '_Name' property of the RDKit molecule object
    (3) The 'FILENAME' property of the RDKit molecule object
        The 'FILENAME' property is not a standard property in the RDKit molecule object or the sdf file, but it is used in some sdf files in our lab.
    (4) The given 'mol_id' argument (if provided)
    If none of these properties are available, the function sets the '_Name' property to an empty string ''.

    Arguments:
    - rdmol: The RDKit molecule object to which the name will be assigned.
    - mol_id: The given molecule ID (optional). If provided, it will be used as the first choice for the name.
    - extra_name: Extra names to be added to the molecule name (optional). If provided, it will be used as the last choices for the name.
    '''

    # Format the extra_name argument
    if extra_name is None:
        extra_name = []

    else:
        if type(extra_name) == str:
            extra_name = [extra_name]
        elif type(extra_name) == list:
            extra_name = extra_name
        else:
            logger.error(f"Extra name should be a string or a list of strings, but got {type(extra_name)}")
            raise ValueError(f"Extra name should be a string or a list of strings, but got {type(extra_name)}")
        
            
    # Get the molecule name from the _Name property
    try:
        NameProp = rdmol.GetProp("_Name")
    except:
        NameProp = None
    
    if type(NameProp) == str:
        NameProp = NameProp.strip()

    # Get the molecule name from the FILENAME property
    try:
        filename = rdmol.GetProp("FILENAME")
    except:
        filename = None

    if type(filename) == str:
        filename = filename.strip()
    
    # Set the _Name property for the molecule according to the following order: NameProp, filename, mol_id
    name_order = extra_name + [NameProp, filename, mol_id]
    name_order = [i for i in name_order if i is not None and i != ""]
    
    if len(name_order) == 0:
        rdmol.SetProp("_Name", '')
    else:
        rdmol.SetProp("_Name", name_order[0])

    return rdmol

def file_to_rdmol(file, mol_id=None, streamlit=False):
    '''
    This function reads from various file formats and returns the corresponding RDKit molecule object (rdmol).
    It achieves the official name for the EMS molecule from the file and assigns the official name to the RDKit molecule object.

    Currently, it supports the following file formats:
    (1) .sdf file (str)
        - The name for the RDKit molecule is assigned in the order of _Name, FILENAME, mol_id.
        - If none of these properties are available, the _Name property will be set to an empty string ''.
        - The official name for the EMS molecule will be obtained from the '_Name' property of the name-assigned RDKit molecule object.
        - The 'streamlit' argument is used to read the sdf file in a website, but is not supported yet. Need to be implemented in the future.
    (2) .xyz file (str)
        - The .xyz files usually don't include a name for the molecule, so it is recommended to set a name for the molecule using the 'mol_id' argument.
        - All of the id and official name for the EMS molecule and the name in the RDKit molecule object will be usually the same.
    (3) .log file by Gaussian (str)
        - The Gaussian .log files usually don't include a name for the molecule, so it is recommended to set a name for the molecule using the 'mol_id' argument.
        - The name for the RDKit molecule is assigned in the order of _Name, FILENAME, mol_id.
    (4) line notation string (str), such as SMILES or SMARTS
        - Both the name for the RDKit molecule and the official name for the EMS molecule will be assigned using the line notation string.
        - The RDKit molecule object generated from the line notation string will be sanitized, hydrogen-added and kekulized.
    (5) RDKit molecule object (rdkit.Chem.rdchem.Mol)
        - The name for the RDKit molecule is assigned in the order of _Name, FILENAME, mol_id.
        - If none of these properties are available, the _Name property will be set to an empty string ''.
        - The official name for the EMS molecule will be obtained from the '_Name' property of the name-assigned RDKit molecule object.
    (6) atom and pair dataframes (tuple)
        - The tuple should contain two pandas dataframes: the atom dataframe and the pair dataframe.
        - Both the name for the RDKit molecule and the official name for the EMS molecule will be assigned using the molecule name in the atom dataframe.
    (7) .mol2 file (str)
        - The name for the RDKit molecule is assigned in the order of _Name, mol_id.
        - The .mol2 file usually includes a name in the first line of the @<TRIPOS>MOLECULE section.
        - The official name for the EMS molecule will be obtained from the '_Name' property of the name-assigned RDKit molecule object.
    (8) .cif file (str)
        - The .cif files usually don't include a name for the molecule, so it is recommended to set a name for the molecule using the 'mol_id' argument.
        - All of the id and official name for the EMS molecule and the name in the RDKit molecule object will be usually the same.
    '''

    file_type = None
    official_name = None
    rdmol = None
    
    # Check if the file is a string
    if isinstance(file, str):

        # Check if the file is a .sdf file
        if file.endswith('.sdf'):
            file_type = 'sdf'

            # Get the RDKit molecule object from the sdf file
            try:
                rdmol = sdf_to_rdmol(file, manual_read=False, streamlit=streamlit)
            except:
                logger.error(f"Fail to read RDKit molecule from the sdf file: {file}")
                raise ValueError(f"Fail to read RDKit molecule from the sdf file: {file}")
            
            # Assign a name to the RDKit molecule object in the order of _Name, FILENAME, mol_id
            # If none of these properties are available, set the _Name property to an empty string ''
            # The official name is obtained from the _Name property of the name-assigned RDKit molecule object
            rdmol = assign_rdmol_name(rdmol, mol_id=mol_id)
            official_name = rdmol.GetProp("_Name")
        

        # Check if the file is a .xyz file
        elif file.endswith('.xyz'):
            file_type = 'xyz'

            # Get the RDKit molecule object from the xyz file
            try:
                rdmol = xyz_to_rdmol(file)
            except:
                logger.error(f"Fail to read RDKit molecule from the xyz file: {file}")
                raise ValueError(f"Fail to read RDKit molecule from the xyz file: {file}")
            
            # Assign a name to the RDKit molecule object and get the official name from the _Name property
            # Because the .xyz files usually don't include a name for the molecule, the id and official name for the EMS molecule and the name in the RDKit molecule object will be the same.
            rdmol = assign_rdmol_name(rdmol, mol_id=mol_id)
            official_name = rdmol.GetProp("_Name")

        
        # Check if the file is a mol2 file
        elif file.endswith('.mol2'):
            file_type = 'mol2'

            # Get the RDKit molecule object from the mol2 file
            try:
                rdmol = mol2_to_rdmol(file)
            except:
                logger.error(f"Fail to read RDKit molecule from the mol2 file: {file}")
                raise ValueError(f"Fail to read RDKit molecule from the mol2 file: {file}")
            
            # Assign a name to the RDKit molecule object in the order of _Name, mol_id
            # The official name is obtained from the _Name property of the name-assigned RDKit molecule object
            rdmol = assign_rdmol_name(rdmol, mol_id=mol_id)
            official_name = rdmol.GetProp("_Name")

        
        # Check if the file is a .cif file
        elif file.endswith('.cif'):
            file_type = 'cif'

            # Get the RDKit molecule object from the cif file
            try:
                rdmol = cif_to_rdmol(file)
            except:
                logger.error(f"Fail to read RDKit molecule from the cif file: {file}")
                raise ValueError(f"Fail to read RDKit molecule from the cif file: {file}")
            
            # Assign a name to the RDKit molecule object and get the official name from the _Name property
            # Because the .cif files usually don't include a name for the molecule, the id and official name for the EMS molecule and the name in the RDKit molecule object will be the same.
            rdmol = assign_rdmol_name(rdmol, mol_id=mol_id)
            official_name = rdmol.GetProp("_Name")

        
        # Check if the file is a .log file
        elif file.endswith('.log'):
            with open(file, 'r') as f:
                first_line = f.readline()
            
            # Check if the .log file is a Gaussian log file
            if 'Gaussian' in first_line:
                file_type = 'gaussian-log'
                
                # Get the atom types and coordinates from the Gaussian log file
                try:
                    atom_types, atom_coords = gaussian_read_structure(file)
                    rdmol = structure_arrays_to_rdmol_NoConn(atom_types, atom_coords)
                except:
                    logger.error(f"Fail to read RDKit molecule from the Gaussian log file: {file}")
                    raise ValueError(f"Fail to read RDKit molecule from the Gaussian log file: {file}")
                
                # Assign a name to the RDKit molecule object in the order of _Name, FILENAME, mol_id
                rdmol = assign_rdmol_name(rdmol, mol_id=mol_id)
                official_name = rdmol.GetProp("_Name")

            # Raise an error if the .log file is not a supported type
            else:
                logger.error(f"Unable to determine the file type from the .log file: {file}")
                raise ValueError(f"Unable to determine the file type from the .log file: {file}")


        # If the file is not a path string, check if it is a line notation string
        else:
            as_smiles = None
            as_smarts = None

            # Try reading the file as a SMILES string
            try:
                as_smiles = Chem.MolFromSmiles(file)
            except:
                logger.warning(f"Fail to read RDKit molecule from the SMILES string: {file}")

            # Try reading the file as a SMARTS string
            try:
                as_smarts = Chem.MolFromSmarts(file)
            except:
                logger.warning(f"Fail to read RDKit molecule from the SMARTS string: {file}")
            
            # If reading the file as a line notation string fails, raise an error
            line_order = [('smiles', as_smiles), ('smarts', as_smarts)]
            line_order = [i for i in line_order if i[1] is not None]
            
            if len(line_order) == 0:
                logger.error(f"Fail to read RDKit molecule from the line notation string: {file}")
                raise ValueError(f"Fail to read RDKit molecule from the line notation string: {file}")
            
            # If reading the file as a line notation string succeeds, assign the line notation string as the name to the RDKit molecule object and the official name to the EMS molecule
            file_type = line_order[0][0]
            line_mol = line_order[0][1]
            line_mol.SetProp("_Name", file)
            official_name = file

            # Process the RDKit molecule object transformed from line notation string
            try:
                Chem.SanitizeMol(line_mol)
                line_mol = Chem.AddHs(line_mol)
                Chem.Kekulize(line_mol)
                AllChem.EmbedMolecule(line_mol)              # obtain the initial 3D structure for a molecule

                if AllChem.MMFFHasAllMoleculeParams(line_mol):
                    AllChem.MMFFOptimizeMolecule(line_mol)
                else:
                    AllChem.UFFOptimizeMolecule(line_mol)

            except Exception as e:
                logger.error(f"Fail to process the rdkit molecule object transformed from line notation string by RDKit: {file}")
                raise e
            
            # Assign the line_mol to the rdmol variable
            rdmol = line_mol

    
    # Check if the file is an RDKit molecule object
    elif isinstance(file, Chem.rdchem.Mol):
        file_type = 'rdmol'

        # Assign the RDKit molecule object in the file to the rdmol variable
        # The name for the RDKit molecule is assigned in the order of _Name, FILENAME, mol_id
        rdmol = file
        rdmol = assign_rdmol_name(rdmol, mol_id=mol_id)

        # Get the official name from the name-assigned RDKit molecule object
        official_name = rdmol.GetProp("_Name")

    
    # Check if the file is atom and pair dataframes
    elif isinstance(file, tuple) and isinstance(file[0], pd.DataFrame):
        file_type = 'dataframe'

        # Get the atom dataframe and the molecule name from the atom dataframe
        atom_df = file[0]
        mol_name = list(atom_df['molecule_name'])[0]
        pair_df = file[1]

        # Get the RDKit molecule object from the atom dataframe
        try:
            rdmol = dataframe_to_rdmol(atom_df,mol_name=mol_name)
            #valency_check
            for atom in rdmol.GetAtoms():
                try:
                    if atom.GetImplicitValence() != 0:
                        raise
                except:
                    raise
        except Exception as e:
            try: 
                rdmol = dataframe_to_rdmol_bond_order(atom_df,pair_df,mol_name=mol_name)
            except Exception as f:
                logger.error(f"Fail to read RDKit molecule from the atom and pair dataframes \n{e}\n{f}")
                raise ValueError(f"Fail to read RDKit molecule from the atom and pair dataframes \n{e}\n{f}")
            
        # Assign the molecule name in the atom dataframe to both the RDKit molecule object and the official name for the EMS molecule
        rdmol.SetProp("_Name", mol_name)
        official_name = mol_name
    

    # If the file is not a valid type, raise an error
    else:
        logger.error(f"Invalid file type! Fail to read RDKit molecule from the file: {file}")
        raise ValueError(f"Invalid file type! Fail to read RDKit molecule from the file: {file}")
    
    # Check if the RDKit molecule object is successfully generated
    if not isinstance(rdmol, Chem.rdchem.Mol):
        logger.error(f"Fail to read RDKit molecule from the file: {file}")
        raise ValueError(f"Fail to read RDKit molecule from the file: {file}")
    
    # Return the file type, official name and RDKit molecule object
    return file_type, official_name, rdmol


def nmr_to_rdmol(rdmol):
    '''
    This function reads NMR data from various file formats and assigns the NMR data to the atom and pair properties of the EMS molecule (rdmol).

    It supports reading NMR data from the following file formats:
    (1) atom and pair dataframes (tuple)
    (2) RDKit molecule object (rdkit.Chem.rdchem.Mol)
    (3) .sdf file (str)
    (4) Gaussian .log file (str)
        For Gaussian .log files, this function saves the shielding tensors in the "raw_shift" attribute of atom properties, and then scales the shielding tensors to chemical shifts and saves them in the "shift" attribute.
    (5) .cif file (str)
    '''
            
    # Read NMR data if rdmol.file is atom and pair dataframes
    # The difference between pair_properties["nmr_types"] and pair_properties["nmr_types_df"] is:
    # (1) pair_properties["nmr_types"] is the matrix of coupling types between every two atoms, so distant atoms are also included, like '11JCH'
    # (2) pair_properties["nmr_types_df"] is the matrix of coupling types only based on the atom pairs in the pair dataframe. 
    #     If the dataframe is a 6-path one, atom pairs with 7 or more bonds are not included, like '7JCH'. The not-included atom pairs are set to a '0' string.
    if rdmol.filetype == "dataframe":
        try:
            atom_df = rdmol.file[0]
            pair_df = rdmol.file[1]
            shift, shift_var, coupling_array, coupling_vars, coupling_types = nmr_read_df(atom_df, pair_df, rdmol.filename)

            rdmol.pair_properties["nmr_types_df"] = coupling_types

            # Check if the non-zero elements of pair_properties["nmr_types_df"] also exists in pair_properties["nmr_types"]
            nmr_type_mask = rdmol.pair_properties["nmr_types_df"] != '0'
            nmr_types_match = rdmol.pair_properties["nmr_types_df"] == rdmol.pair_properties["nmr_types"]

            if not (nmr_types_match == nmr_type_mask).all():
                logger.warning(f"Some coupling types in pair_properties['nmr_types_df'] do not match with pair_properties['nmr_types'] for molecule {rdmol.id}")

        except Exception as e:
            logger.error(f'Fail to read NMR data for molecule {rdmol.id} from dataframe')
            raise e
    
    # Read NMR data if rdmol.file is an RDKit molecule object
    elif rdmol.filetype == "rdmol":
        try:
            shift, shift_var, coupling_array, coupling_vars = nmr_read_rdmol(rdmol.rdmol, rdmol.id)
        except Exception as e:
            logger.error(f'Fail to read NMR data for molecule {rdmol.id} from rdkit molecule object')
            raise e
    
    # Read NMR data if rdmol.file is an SDF file
    elif rdmol.filetype == 'sdf':
        try:
            shift, shift_var, coupling_array, coupling_vars = nmr_read_sdf(rdmol.file, rdmol.streamlit)
        except Exception as e:
            logger.error(f'Fail to read NMR data for molecule {rdmol.id} from SDF file {rdmol.file}')
            raise e
    
    # Read NMR data if rdmol.file is a Gaussian .log file
    elif rdmol.filetype == 'gaussian-log':
        try:
            shift, coupling_array = nmr_read_gaussian(rdmol.file)
            shift_var = np.zeros_like(shift)
            coupling_vars = np.zeros_like(coupling_array)
        except Exception as e:
            logger.error(f'Fail to read NMR data for molecule {rdmol.id} from Gaussian .log file {rdmol.file}')
            raise e
        
    # Read NMR data if rdmol.file is a .cif file
    elif rdmol.filetype == 'cif':
        try:
            shift, shift_var, coupling_array, coupling_vars = nmr_read_cif(rdmol.file)
        except Exception as e:
            logger.error(f'Fail to read NMR data for molecule {rdmol.id} from .cif file {rdmol.file}')
            raise e

    # Raise error if the file type is not among the above
    else:
        logger.error(f'File {rdmol.id} with file type {rdmol.filetype} is not supported for reading NMR data')
        raise ValueError(f'File {rdmol.id} with file type {rdmol.filetype} is not supported for reading NMR data')


    # If file type is 'gaussian-log', save the unscaled shift values in the "raw_shift" attribute of atom properties
    # Then save the scaled shift values in the "shift" attribute
    # The coupling values generally don't need to be scaled, so save them in the "coupling" attribute
    if rdmol.filetype == 'gaussian-log':
        rdmol.atom_properties["raw_shift"] = shift
        rdmol.atom_properties["shift_var"] = shift_var
        rdmol.pair_properties["coupling"] = coupling_array
        rdmol.pair_properties["coupling_var"] = coupling_vars

        rdmol.atom_properties["shift"] = scale_chemical_shifts(shift, rdmol.type)

    # Assign the NMR data to the atom and pair properties for other file types
    else:
        rdmol.atom_properties["shift"] = shift
        rdmol.atom_properties["shift_var"] = shift_var
        rdmol.pair_properties["coupling"] = coupling_array
        rdmol.pair_properties["coupling_var"] = coupling_vars

