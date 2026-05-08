import os
import pytest
import sys


if not os.path.basename(os.getcwd()) == 'examples':
    pytest.exit("ERROR: This test suite must be run from within the /tests directory "
                "due to relative path dependencies for .ini and .xyz files.")

else:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    import driver as resp

    molecule_inputs = ['methanol.ini', 'phosphate.ini', 'dmso_opt_c1_a.ini', 'dmso_opt_c1_b.ini',
                       '2_methylpropanal_stage_1.ini', '2_methylpropanal_stage_2.ini'
                      ]

    for ini in molecule_inputs:
        print('-' * 40)
        print(f'Molecule: {ini.split(".")[0].capitalize()}')
        charges = resp.resp(ini)
    
        print('Unrestrained Electrostatic Potential Charges')
        print(f'{charges[0]}\n')

        print('Restrained Electrostatic Potential (RESP) Charges')
        print(f'{charges[1]}\n')
