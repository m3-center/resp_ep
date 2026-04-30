import os
import sys
import pytest
import numpy as np
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
DATA_DIR = TEST_DIR / "data"
ROOT_DIR = TEST_DIR.parent

sys.path.insert(0, str(ROOT_DIR))

from resp_ep import driver as resp

def get_data(filename: str) -> str:
    """ Helper to get absolute path to test data files. """
    path = DATA_DIR / filename
    if not path.exists():
        pytest.fail(f"Test data file not found: {path}")
    return str(path)


def print_results(charges: list, reference_charges: list):
    print('Unrestrained Electrostatic Potential Charges')
    print(f'{charges[0]}\n')

    print('Restrained Electrostatic Potential (RESP) Charges')
    print(f'{charges[1]}\n')

    print('Reference RESP Charges (via RED-III 5)')
    print(f'{reference_charges}\n')

    print('Difference')
    print(f'{charges[1]-reference_charges}\n')


## --- TESTS ---

def test_resp_unconstrained_a():
    ''' One-stage fitting of charges. '''
    reference_charges = np.array([-0.294974,  0.107114,  0.107114,  0.084795,
                                   0.803999, -0.661279,  0.453270, -0.600039])

    charges = resp.resp(get_data('resp_unconstrained_a.ini'))
    assert np.allclose(charges[1], reference_charges, atol=5.0e-4)


def test_resp_unconstrained_b():
    ''' One-stage fitting of charges (ESP read in). '''
    reference_charges = np.array([-0.294974,  0.107114,  0.107114,  0.084795,
                                   0.803999, -0.661279,  0.453270, -0.600039])

    charges = resp.resp(get_data('resp_unconstrained_b.ini'))
    assert np.allclose(charges[1], reference_charges, atol=5.0e-4)


def test_resp_constrained_a():
    ''' Two-stage fitting of charges with equivalence constraints. '''
    reference_charges = np.array([-0.290893,  0.098314,  0.098314,  0.098314,
                                   0.803999, -0.661279,  0.453270, -0.600039])

    charges = resp.resp(get_data('resp_constrained_a.ini'))
    assert np.allclose(charges[1], reference_charges, atol=5.0e-4)


def test_resp_two_conformers_a():
    ''' One-stage fitting using two conformations. '''
    reference_charges = np.array([-0.149134, 0.274292, -0.630868,  0.377965, -0.011016,
                                  -0.009444, 0.058576,  0.044797,  0.044831])

    charges = resp.resp(get_data('resp_two_confs_a.ini'))
    assert np.allclose(charges[1], reference_charges, atol=1.0e-5)


def test_resp_two_conformers_b():
    ''' Two-stage fitting using two conformations. '''
    reference_charges = np.array([-0.079853, 0.253918, -0.630868, 0.377965, -0.007711,
                                  -0.007711, 0.031420,  0.031420, 0.031420])

    charges = resp.resp(get_data('resp_two_confs_b.ini'))
    assert np.allclose(charges[1], reference_charges, atol=1.5e-5)


def test_bromoethene():
    reference_charges = np.array([-0.22793428, 0.14970713, 0.17055694, -0.23219228,
                                  -0.08681686, 0.22667934])

    charges = resp.resp(get_data('bromoethene.ini'))
    assert np.allclose(charges[1], reference_charges, atol=1.0e-5)


def test_bromoethene_x():
    reference_charges = np.array([-0.37318215, 0.16577433, 0.21990568, -0.00848194,
                                  -0.26039446, 0.17776180, 0.07861675])

    charges = resp.resp(get_data('bromoethene_x.ini'))
    assert np.allclose(charges[1], reference_charges, atol=1.0e-5)


def test_methanol():
    reference_charges = np.array([0.10493159, 0.03719950, 0.03719950, 0.03719950,
                                  -0.60185099, 0.38532091])

    charges = resp.resp(get_data('methanol.ini'))
    assert np.allclose(charges[1], reference_charges, atol=1.0e-5)


def test_methanol_x():
    reference_charges = np.array([-0.03646927, 0.05658907, 0.05658907, 0.05658907,
                                  -0.32640632, 0.30812274, -0.05750718, -0.05750718])

    charges = resp.resp(get_data('methanol_x.ini'))
    assert np.allclose(charges[1], reference_charges, atol=1.0e-5)