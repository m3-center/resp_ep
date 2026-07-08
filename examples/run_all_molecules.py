import glob
import os
import sys

from resp_ep import driver


if not os.path.basename(os.getcwd()) == 'examples':
    sys.exit("ERROR: This script must be run from within the 'examples' subdirectory.")
else:
    molecule_dirs = ['1_bromobutane',
                     '1_bromobutane/multiple_orientations',
                     '1_bromobutane/multiple_conformers',
                     '2_methylpropanal',
                     'acetic_acid',
                     'dmso',
                     'methanol',
                     'methanol/multiple_orientations',
                     'phosphate',
                    ]

    for mol_dir in molecule_dirs:
        ini_search_pattern = os.path.join(mol_dir, '*.ini')
        ini_files = glob.glob(ini_search_pattern)

        for ini in ini_files:
            mol_name = os.path.splitext(os.path.basename(ini))[0].capitalize()

            print('-' * 40)
            print(f'Molecule: {mol_name}')

            charges = driver.resp(ini)

            print(f"\n{'-' * 10} Results {'-' * 10}\n")
            print('Unrestrained Electrostatic Potential Charges')
            print(f'  {charges[0]}\n')

            print('Restrained Electrostatic Potential (RESP) Charges')
            print(f'  {charges[1]}\n')
