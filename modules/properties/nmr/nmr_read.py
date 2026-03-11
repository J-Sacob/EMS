import numpy as np
import sys
import logging

from EMS.modules.comp_chem.gaussian.gaussian_read import gaussian_read_nmr


########### Set up the logger system ###########
logger = logging.getLogger(__name__)
stdout = logging.StreamHandler(stream = sys.stdout)
formatter = logging.Formatter("%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s >>> %(message)s")
stdout.setFormatter(formatter)
logger.addHandler(stdout)
logger.setLevel(logging.INFO)
########### Set up the logger system ###########


def nmr_read_sdf(stringfile, streamlit=False):
    '''
    This function is used to read NMR data from an SDF file. When reading NMR data for an EMS object, the SDF file is the self.file attribute.
    It reads the SDF file line by line and extracts the NMR data from the <NMREDATA_ASSIGNMENT> and <NMREDATA_J> properties in the SDF file.
    Details of this function:
    (1) This function is designed to read NMR data from both V2000 and V3000 SDF files.
    (2) This function loops over the lines of the SDF file only once, so as to minimize the time taken to read the NMR data.
    (3) This function first reads the info line of the SDF file to get the number of atoms in the molecule ('V2000' line in V2000 SDF, or 'M  V30 COUNTS' line in V3000 SDF),
        and then reads the structure block to get the number of atom lines (before 'M  END' line). If the two numbers are not equal, an error will be raised.

    Args:
    - stringfile (str): The path to the SDF file, which is the self.file attribute of the EMS object.
    - streamlit (bool): If True, the SDF file is as read as streamlit.
    '''

    # Get the file as a list of lines. The streamlit part will be developed later.
    if streamlit:
        pass
    else:
        with open(stringfile, "r") as f:
            lines = f.readlines()

    # Initialize variables when reading the SDF file
    structure_end_check = False          # Check if the 'M  END' line or the end of structure information block is reached
    atom_block_start_check = False       # Check if the atom block start line is reached
    atom_block_end_check = False         # Check if the atom block end line is reached
    shift_switch = False                 # Switch to read the chemical shift lines
    cpl_switch = False                   # Switch to read the coupling constant lines

    chkatoms = 0                         # Number of atoms read in the 'V2000' line in V2000 SDF file or 'M  V30 COUNTS' line in V3000 SDF file
    sdf_version = None                   # SDF version (V2000 or V3000) read in the info line
    atom_block_start = 0                 # The index of the line where the atom block starts
    atom_block_end = 0                   # The index of the line where the atom block ends

    # Initialize arrays for saving NMR data
    shift_array = None
    shift_var = None
    coupling_array = None
    coupling_len = None
    coupling_var = None

    # Get the index of the first atom. Some files may have 1 as the first atom index, but this is not standard. The standard is 0.
    index_check_shift_switch = False
    first_atom_index = 0

    for idx, line in enumerate(lines):
        if "<NMREDATA_ASSIGNMENT>" in line:
            index_check_shift_switch = True
            continue

        if index_check_shift_switch:
            first_atom_index = int(line.split()[0])
            break

    # Loop over the lines of the SDF file only once
    for idx, line in enumerate(lines):

        # Break the loop if the end of the molecule ('$$$$') is reached
        if '$$$$' in line:
            break

        # Check if the block of structure information has ended
        if 'M  END' in line:
            structure_end_check = True

        # Enter the mode of getting the atom number, which is before the 'M  END' line
        if not structure_end_check:
            # Get the SDF version
            if 'V3000' in line:
                sdf_version = 'V3000'
            elif 'V2000' in line:
                sdf_version = 'V2000'

            # For V3000 SDF, get the number of atoms from the 'M  V30 COUNTS' line 
            # and indices of the start and end lines of the atom block from the 'M  V30 BEGIN ATOM' and 'M  V30 END ATOM' lines
            if sdf_version == 'V3000':
                if 'M  V30 COUNTS' in line:
                    chkatoms = int(line.split()[3])
                
                if 'M  V30 BEGIN ATOM' in line:
                    atom_block_start = idx + 1
                    atom_block_start_check = True
                
                if 'M  V30 END ATOM' in line:
                    atom_block_end = idx
                    atom_block_end_check = True
            
            # For V2000 SDF, get the number of atoms from the 'V2000' line 
            # and the indices of the start and end lines of the atom block from change of word numbers in the lines
            elif sdf_version == 'V2000':
                if 'V2000' in line:
                    chkatoms = int(line[0:3].strip())
                
                if atom_block_start_check == False and len(line.split()) >= 14:
                    atom_block_start = idx
                    atom_block_start_check = True
                
                if atom_block_end_check == False and atom_block_start_check == True and len(line.split()) <= 8:
                    atom_block_end = idx
                    atom_block_end_check = True
        

        # Enter the mode of checking if the atom number is correct and reading NMR data, which is after the 'M  END' line
        else:      # if structure_end_check == True
            
            # When in the 'M  END' line, initialize the arrays for saving NMR data and check if the atom number is correctly read
            if 'M  END' in line:
                # Check if the atom number is correctly read
                atoms = atom_block_end - atom_block_start
                if chkatoms != atoms or chkatoms == 0:
                    logger.error(f'Number of atoms in the SDF file is not correctly read: {stringfile}')
                    raise ValueError(f'Number of atoms in the SDF file is not correctly read: {stringfile}')
                
                # Define empty arrays for saving NMR data
                # Variance is used for machine learning
                shift_array = np.zeros(atoms, dtype=np.float64)
                shift_var = np.zeros(atoms, dtype=np.float64)
                coupling_array = np.zeros((atoms, atoms), dtype=np.float64)
                coupling_len = np.zeros((atoms, atoms), dtype=np.int64)
                coupling_var = np.zeros((atoms, atoms), dtype=np.float64)
            
            # After the 'M  END' line, read the NMR data from the <NMREDATA_ASSIGNMENT> and <NMREDATA_J> properties
            else:
                if "<NMREDATA_ASSIGNMENT>" in line:
                    shift_switch = True
                if "<NMREDATA_J>" in line:
                    shift_switch = False
                    cpl_switch = True

                # If shift assignment label found, process shift rows
                if shift_switch:
                    # Shift assignment row looks like this
                    #  0    , -33.56610000   , 8    , 0.00000000     \
                    items = line.split()
                    try:
                        int(items[0])
                    except:
                        continue
                    shift_array[int(items[0]) - first_atom_index] = float(items[2])
                    shift_var[int(items[0]) - first_atom_index] = float(items[6])

                # If coupling assignment label found, process coupling rows
                if cpl_switch:
                    # Coupling row looks like this
                    #  0         , 4         , -0.08615310    , 3JON      , 0.00000000
                    items = line.split()
                    try:
                        int(items[0])
                    except:
                        continue
                    length = int(items[6].strip()[0])
                    coupling_array[int(items[0]) - first_atom_index][int(items[2]) - first_atom_index] = float(items[4])
                    coupling_array[int(items[2]) - first_atom_index][int(items[0]) - first_atom_index] = float(items[4])
                    coupling_var[int(items[0]) - first_atom_index][int(items[2]) - first_atom_index] = float(items[8])
                    coupling_var[int(items[2]) - first_atom_index][int(items[0]) - first_atom_index] = float(items[8])
                    coupling_len[int(items[0]) - first_atom_index][int(items[2]) - first_atom_index] = length
                    coupling_len[int(items[2]) - first_atom_index][int(items[0]) - first_atom_index] = length

    # Raise errors if the following conditions are not met. These conditions make sure that the NMR data is correctly read.
    if structure_end_check == False:
        logger.error(f"No 'M  END' line found in the file: {stringfile}")
        raise ValueError(f"No 'M  END' line found in the file: {stringfile}")
    
    if not (atom_block_start_check and atom_block_end_check):
        logger.error(f"Structure block not found in the file: {stringfile}")
        raise ValueError(f"Structure block not found in the file: {stringfile}")
    
    if sdf_version is None:
        logger.error(f'SDF version not found in the file: {stringfile}')
        raise ValueError(f'SDF version not found in the file: {stringfile}')
    
    if chkatoms == 0 or atom_block_end - atom_block_start == 0:
        logger.error(f'Number of atoms in the SDF file is not correctly read: {stringfile}')
        raise ValueError(f'Number of atoms in the SDF file is not correctly read: {stringfile}')
    
    if shift_array is None:
        logger.error(f'NMR data in the SDF file is not correctly read: {stringfile}')
        raise ValueError(f'NMR data in the SDF file is not correctly read: {stringfile}')

    return shift_array, shift_var, coupling_array, coupling_var

    
def nmr_read_rdmol(rdmol, mol_id):
    '''
    This function is used to read NMR data from an RDKit molecule object.

    Args:
    - rdmol (rdkit.Chem.rdchem.Mol): RDKit molecule object.
    - mol_id (str): Molecule ID.
    '''

    # Get all the properties of the RDKit molecule object
    prop_dict = rdmol.GetPropsAsDict()
    
    # Get the NMR data (NMREDATA_ASSIGNMENT for chemical shifts and NMREDATA_J for coupling constants) from the properties
    try:
        shift = prop_dict['NMREDATA_ASSIGNMENT']
        coupling = prop_dict['NMREDATA_J']
    except Exception as e:
        logger.error(f'No NMR data found for molecule {mol_id}')
        raise ValueError(f'No NMR data found for molecule {mol_id}')
    
    # Get the index of the first atom. Some files may have 1 as the first atom index, but this is not standard. The standard is 0.
    first_atom_index = int(shift.split('\n')[0].split()[0])

    # Split the NMR data block into lines and then into items
    shift_items = []
    for line in shift.split('\n'):
        if line:
            shift_items.append(line.split())
    
    coupling_items = []
    for line in coupling.split('\n'):
        if line:
            coupling_items.append(line.split())
    
    # Initialize arrays for saving NMR data
    num_atom = len(shift_items)
    shift_array = np.zeros(num_atom, dtype=np.float64)
    shift_var = np.zeros(num_atom, dtype=np.float64)
    coupling_array = np.zeros((num_atom, num_atom), dtype=np.float64)
    coupling_var = np.zeros((num_atom, num_atom), dtype=np.float64)

    # Read the NMR data from the lines
    # Shift assignment row looks like this
    #  0    , -33.56610000   , 8    , 0.00000000     \
    for item in shift_items:
        shift_array[int(item[0]) - first_atom_index] = float(item[2])
        shift_var[int(item[0]) - first_atom_index] = float(item[6])
    
    # Coupling row looks like this
    #  0         , 4         , -0.08615310    , 3JON      , 0.00000000
    for item in coupling_items:
        coupling_array[int(item[0]) - first_atom_index][int(item[2]) - first_atom_index] = float(item[4])
        coupling_array[int(item[2]) - first_atom_index][int(item[0]) - first_atom_index] = float(item[4])
        coupling_var[int(item[0]) - first_atom_index][int(item[2]) - first_atom_index] = float(item[8])
        coupling_var[int(item[2]) - first_atom_index][int(item[0]) - first_atom_index] = float(item[8])
    
    return shift_array, shift_var, coupling_array, coupling_var


def nmr_read_df(atom_df, pair_df, mol_name):
    '''
    This function is used to read NMR data from atom and pair dataframes with the assigned molecule name.

    Args:
    - atom_df (pd.DataFrame): DataFrame containing atom-level NMR data.
    - pair_df (pd.DataFrame): DataFrame containing pair-level NMR data.
    - mol_name (str): The molecule to read NMR data for.
    '''

    # Get the atom and pair dataframes for the given molecule name
    mol_atom_df = atom_df[atom_df['molecule_name'] == mol_name]
    mol_pair_df = pair_df[pair_df['molecule_name'] == mol_name]

    # Check whether the indexes in the molecule are continuous
    # If not, that means two molecules in the dataframe share the same molecule name
    atom_index = list(mol_atom_df.index)
    pair_index = list(mol_pair_df.index)

    atom_check = True
    pair_check = True

    for i in range(len(atom_index)-1):
        if atom_index[i+1] - atom_index[i] != 1:
            atom_check = False
            break
    
    for i in range(len(pair_index)-1):
        if pair_index[i+1] - pair_index[i] != 1:
            pair_check = False
            break
    
    if not (atom_check and pair_check):
        logger.error(f"The indexes in the molecule {mol_name} are not continuous. Two molecules may share the same molecule name.")
        raise ValueError(f"The indexes in the molecule {mol_name} are not continuous. Two molecules may share the same molecule name.")
    

    # Get the atom-level NMR parameters
    num_atoms = len(atom_df)
    shift = np.array(atom_df['shift'], dtype=np.float64)
    try:
        shift_var = np.array(atom_df['shift_var'], dtype=np.float64)
    except:
        shift_var = np.zeros(len(atom_df['shift']), dtype=np.float64)

    if not (shift.ndim == 1 and shift_var.ndim == 1):
        logger.error(f"Shift and shift variance arrays should be one-dimensional for molecule {mol_name}!")
        raise ValueError(f"Shift and shift variance arrays should be one-dimensional for molecule {mol_name}!")
    
    if not (len(shift) == num_atoms and len(shift_var) == num_atoms):
        logger.error(f"Shift and shift variance arrays should be the same length as the number of atoms for molecule {mol_name}!")
        raise ValueError(f"Shift and shift variance arrays should be the same length as the number of atoms for molecule {mol_name}!")
    
    # Get the pair-level NMR parameters
    coupling_mat = np.zeros((num_atoms, num_atoms), dtype=np.float64)
    coupling_var_mat = np.zeros((num_atoms, num_atoms), dtype=np.float64)
    coupling_types_mat = np.zeros((num_atoms, num_atoms), dtype=np.int32).astype(str)

    i = np.array(pair_df['atom_index_0'], dtype=np.int32)
    j = np.array(pair_df['atom_index_1'], dtype=np.int32)
    coupling = np.array(pair_df['coupling'], dtype=np.float64)
    try:
        coupling_var = np.array(pair_df['coupling_var'], dtype=np.float64)
    except:
        coupling_var = np.zeros(len(pair_df['coupling']), dtype=np.float64)

    coupling_types = np.array(pair_df['nmr_types'], dtype=str)
    
    coupling_mat[i, j] = coupling
    coupling_var_mat[i, j] = coupling_var
    coupling_types_mat[i, j] = coupling_types

    # Return the NMR data
    return shift, shift_var, coupling_mat, coupling_var_mat, coupling_types_mat


def nmr_read_gaussian(file):
    '''
    This function reads the Gaussian log file and extracts the chemical shielding tensors and coupling constants.

    Args:
    - file (str): The path to the Gaussian log file.
    '''

    shift_array, couplings = gaussian_read_nmr(file)

    return shift_array, couplings


def nmr_read_cif(file):
    '''
    This function reads NMR data from a .cif file.
    '''

    with open(file, 'r') as f:
        block = f.read()
        lines = block.strip().split('\n')
    
    # Initialize lists to store chemical shifts and coupling constants
    shift_list = []
    coupling_list = []

    # Initialize flags
    shift_flag = False
    coupling_flag = False

    # Initialize the methods to read NMR data from .cif file
    shift_method = None
    coupling_method = None

    for line in lines:
        # Check for the start of chemical shift data block
        if 'shiftml' in line.lower() and 'cs' in line.lower():
            shift_flag = True
            coupling_flag = False
            shift_method = 'shiftml'
            continue

        ############ This section is reserved for reading coupling constants ############

        ############ This section is reserved for reading coupling constants ############
        
        # Read chemical shift data
        if shift_flag:
            # ShiftML format
            if shift_method == 'shiftml':
                line_split = line.strip().split()
                line_split = [item.replace('[','').replace(']','') for item in line_split if item != '[' and item != ']']
                shift_list.extend(line_split)
                # Stop reading when reaching the end of the block
                if ']' in line:
                    shift_flag = False
                    continue
            
            # Raise error if the format is not recognized
            else:
                logger.error(f'Unrecognized chemical shift format in .cif file: {file}')
                raise ValueError(f'Unrecognized chemical shift format in .cif file: {file}')
            
        ############ This section is reserved for reading coupling constants ############

        ############ This section is reserved for reading coupling constants ############

    # Convert lists to numpy arrays
    num_atom = len(shift_list)
    shift_array = np.array(shift_list, dtype=np.float64)
    shift_var = np.zeros(num_atom, dtype=np.float64)

    if coupling_method is None:
        coupling_array = np.zeros((num_atom, num_atom), dtype=np.float64)
        coupling_var = np.zeros((num_atom, num_atom), dtype=np.float64)
    else:
        coupling_array = np.array(coupling_list, dtype=np.float64)
        coupling_var = np.zeros_like(coupling_array, dtype=np.float64)
    
    return shift_array, shift_var, coupling_array, coupling_var
        