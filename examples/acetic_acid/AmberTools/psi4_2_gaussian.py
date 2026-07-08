#!/usr/bin/env python

def convert_psi4_to_gaussian_esp_from_xyz(grid_input_file, esp_input_file, xyz_file, gaussian_output_file):
    """
    Converts Psi4 formatted grid and ESP files back to a Gaussian electrostatic potential (.dat) file.
    Matches the specific Gaussian layout: X Y Z Atomic_Number Label
    
    Parameters:
    grid_input_file (str): Path to the input Psi4 grid file (Angstroms)
    esp_input_file (str): Path to the input Psi4 ESP file
    xyz_file (str): Path to the input geometry file (.xyz format, Angstroms)
    gaussian_output_file (str): Path to the reconstructed output Gaussian ESP file
    """
    # Conversion factor from Angstrom to Bohr (CODATA 2018)
    ANGSTROM_TO_BOHR = 1 / 0.5291772109
    
    # Mapping table for atomic numbers (Add more elements here if needed)
    ELEMENT_TO_ATOMIC_NUM = {
        'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'NE': 10
    }
    
    # 1. Parse the XYZ file
    with open(xyz_file, 'r') as f_xyz:
        xyz_lines = [line.strip() for line in f_xyz if line.strip()]
        
    if not xyz_lines:
        raise ValueError("The XYZ file is empty.")
        
    num_atoms = int(xyz_lines[0])
    atom_lines_raw = xyz_lines[2:2 + num_atoms]
    
    if len(atom_lines_raw) != num_atoms:
        raise ValueError(f"XYZ header specified {num_atoms} atoms, but found {len(atom_lines_raw)} lines.")

    # Format the atom block to match Gaussian's exact structure: X Y Z Atomic_Number Label
    gaussian_atom_block = []
    for line in atom_lines_raw:
        tokens = line.split()
        symbol = tokens[0]
        
        # Determine atomic number (default to 1 if not found, or use upper case)
        atom_upper = symbol.upper()
        atomic_num = ELEMENT_TO_ATOMIC_NUM.get(atom_upper, 1)
        
        # Create a lowercase label variant (e.g., 'ow', 'hw') based on your example
        # If your XYZ uses 'O' -> 'ow', if it uses 'H' -> 'hw'
        label = "ow" if atom_upper == "O" else ("hw" if atom_upper == "H" else symbol.lower())
        
        # Convert XYZ coordinates (Angstrom) to Bohr
        x_atom_bohr = float(tokens[1]) * ANGSTROM_TO_BOHR
        y_atom_bohr = float(tokens[2]) * ANGSTROM_TO_BOHR
        z_atom_bohr = float(tokens[3]) * ANGSTROM_TO_BOHR
        
        # Formatting mimics the exact wide margin style seen in your example template
        formatted_atom = (f"                  {x_atom_bohr:15.7E}  {y_atom_bohr:14.7E}  "
                          f"{z_atom_bohr:14.7E}  {atomic_num}  {label}\n")
        gaussian_atom_block.append(formatted_atom)

    # 2. Read the Psi4 grid coordinates (Angstrom) and convert to Bohr
    grid_coords_bohr = []
    with open(grid_input_file, 'r') as f_grid:
        for line in f_grid:
            tokens = line.split()
            if not tokens:
                continue
            x_bohr = float(tokens[0]) * ANGSTROM_TO_BOHR
            y_bohr = float(tokens[1]) * ANGSTROM_TO_BOHR
            z_bohr = float(tokens[2]) * ANGSTROM_TO_BOHR
            grid_coords_bohr.append((x_bohr, y_bohr, z_bohr))
            
    # 3. Read the Psi4 ESP values
    esp_values = []
    with open(esp_input_file, 'r') as f_esp:
        for line in f_esp:
            token = line.split()
            if not token:
                continue
            esp_values.append(float(token[0]))
            
    if len(grid_coords_bohr) != len(esp_values):
        raise ValueError(f"Mismatch: Found {len(grid_coords_bohr)} grid points but {len(esp_values)} ESP values.")
        
    # 4. Write out the Gaussian formatted file
    num_points = len(esp_values)
    
    with open(gaussian_output_file, 'w') as f_out:
        # Enforce fixed-width columns (atoms: I5, points: I6, 0flag: I5 format to match Gaussian)
        f_out.write(f"{num_atoms:4d}{num_points:6d}{0:5d}\n")

        # Write the formatted atom lines
        for line in gaussian_atom_block:
            f_out.write(line)
            
        # Write the grid block: ESP, X, Y, Z (matching exact spacing layout)
        for esp, (x, y, z) in zip(esp_values, grid_coords_bohr):
            f_out.write(f"  {esp:14.7E}  {x:14.7E}  {y:14.7E}  {z:14.7E}\n")

    print(f"Successfully reconstructed Gaussian ESP file '{gaussian_output_file}' with {num_points} points.")


convert_psi4_to_gaussian_esp_from_xyz('../acetic_acid_grid.dat', '../acetic_acid_esp.dat', '../acetic_acid.xyz', 'acetic_acid_gaussian.dat')
convert_psi4_to_gaussian_esp_from_xyz('../acetic_acid_grid.dat', '../acetic_acid_esp.dat', '../acetic_acid_x.xyz', 'acetic_acid_x_gaussian.dat')