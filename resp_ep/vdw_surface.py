from __future__ import division, absolute_import, print_function

import numpy as np


"""
A srcipt to generate van der Waals surface of molecules.
"""


def vdw_radii(element: str, radii_set: str = 'legacy') -> float:
    """ Assign the van der Waals radius to an element.
        The values were taken from GAMESS.
        
        Args:
            element: one or two letter element identifier
            radii_set: the definition of the vdw radii to use. Options are:

                - 'legacy': the original vdw radii defined in
                    A. Alenaizan, L. A. Burns, C. D. Sherrill - https://doi.org/10.1002/qua.26035 
                - 'Tkatchenko2024': The Charry amd Tkatchenko 2024 vdw radii - https://doi.org/10.1021/acs.jctc.4c00784

        Returns:
            van der Waals radius (Angstrom)
    """
    BOHR_TO_ANGSTROM = 0.529177210

    if not isinstance(element, str):
        raise TypeError(f'The element was not given as a string (i.e., {element} variable).')

    if radii_set == 'legacy':
        lookup_key = element.capitalize()
        radii = {'H':  1.20, 'He': 1.20,
                 'Li': 1.37, 'Be': 1.45, 'B':  1.45, 'C':  1.50,
                 'N':  1.50, 'O':  1.40, 'F':  1.35, 'Ne': 1.30,
                 'Na': 1.57, 'Mg': 1.36, 'Al': 1.24, 'Si': 1.17,
                 'P':  1.80, 'S':  1.75, 'Cl': 1.70}
    elif radii_set == 'Tkatchenko2024':
        lookup_key = element.capitalize()
        radii_bohr = {'H' : 3.164697, 'He': 2.672999, 'Li': 5.289595, 'Be': 4.2875, 'B' : 3.9302,
                      'C' : 3.6096,   'N' : 3.398,    'O' : 3.240,    'F' : 3.0822, 'Ne': 2.935712,
                      'Na': 5.2850,   'Mg': 4.6952,   'Al': 4.5574,   'Si': 4.281,  'P' : 4.043,
                      'S' : 3.8993,   'Cl': 3.7441,   'Ar': 3.600377, 'K' : 5.7384, 'Ca': 5.276,
                      'Sc': 4.907,    'Ti': 4.929,    'V' : 4.832,    'Cr': 4.799,  'Mn': 4.664,
                      'Fe': 4.603,    'Co': 4.525,    'Ni': 4.451,    'Cu': 4.425,  'Zn': 4.3036,
                      'Ga': 4.464,    'Ge': 4.324,    'As': 4.150,    'Se': 4.130,  'Br': 3.944,
                      'Kr': 3.819973, 'Rb': 5.8196,   'Sr': 5.4300,   'Y' : 5.280,  'Zr': 5.009,
                      'Nb': 4.914,    'Mo': 4.832,    'Tc': 4.765,    'Ru': 4.703,  'Rh': 4.64,
                      'Pd': 4.0681,   'Ag': 4.525,    'Cd': 4.411,    'In': 4.635,  'Sn': 4.501,
                      'Sb': 4.369,    'Te': 4.292,    'I' : 4.2049,   'Xe': 4.0943, 'Cs': 6.0103,
                      'Ba': 5.686,    'La': 5.498,    'Ce': 5.461,    'Pr': 5.502,  'Nd': 5.472,
                      'Pm': 5.442,    'Sm': 5.410,    'Eu': 5.377,    'Gf': 5.262,  'Tb': 5.317,
                      'Dy': 5.294,    'ho': 5.252,    'Er': 5.223,    'Tm': 5.192,  'Yb': 5.166,
                      'Lu': 5.155,    'Hf': 4.950,    'Ta': 4.72,     'W' : 4.66,   'Re': 4.603,
                      'Os': 4.548,    'Ir': 4.513,    'Pt': 4.438,    'Au': 4.259,  'Hg': 4.2230,
                      'Tl': 4.464,    'Pb': 4.425,    'Bi': 4.438,    'Po': 4.383,  'At': 4.354,
                      'Rn': 4.242,    'Fr': 5.8144,   'Ra': 5.605,    'Ac': 5.453,  'Th': 5.51,
                      'Pa': 5.242,    'U' : 5.111,    'Np': 5.228,    'Pu': 5.13,   'Am': 5.12,
                      'Cm': 5.19,     'Bk': 5.09,     'Cf': 5.07,     'Es': 5.05,   'Fm': 5.02,
                      'Md': 4.99,     'No': 4.996,    'Lr': 5.820,    'Rf': 5.009,  'Db': 4.354,
                      'Sg': 4.324,    'Bh': 4.292,    'Hs': 4.259,    'Mt': 4.225,  'Ds': 4.188,
                      'Rg': 4.19,     'Cn': 4.109,    'Nh': 4.130,    'Fl': 4.169,  'Mc': 4.69,
                      'Ts': 4.74,     'Og': 4.560}
        # Convert (rounded to 6 decimal places - max. decimals above)
        radii = {element: round(value * BOHR_TO_ANGSTROM, 6) for element, value in radii_bohr.items()}
    else:
        raise ValueError(f'Invalid radii_set: {radii_set}. Valid options are "legacy" and "Tkatchenko2024".')

    if lookup_key in radii.keys():
        return radii[lookup_key]
    else:
        raise KeyError(f'{element} is not internally supported; use the "vdw_radii" option to add its vdw radius.')


def surface(n_points: int) -> np.ndarray:
    """ Computes approximately `n_points` points on a unit sphere.

        Code was adapted from GAMESS. 

        Args:
            n_points : the desired number of surface points.

        Returns:
            xyz coordinates of surface points.

        Dependencies:
            numpy
    """
    if not isinstance(n_points, int):
        raise TypeError(f'The number of points was not given as an integer (i.e., {n_points} variable).')
    else:
        surface_points = []
        eps = 1e-10

        nequat = int(np.sqrt(np.pi * n_points))
        nvert = int(nequat / 2)
        nu = 0

        for i in range(nvert + 1):
            fi = np.pi * i / nvert
            z = np.cos(fi)
            xy = np.sin(fi)
            nhor = int(nequat * xy + eps)

            if nhor < 1:
                nhor = 1

            for j in range(nhor):
                fj = 2 * np.pi * j / nhor
                x = np.cos(fj) * xy
                y = np.sin(fj) * xy

                if nu >= n_points:
                    return np.array(surface_points)

                nu += 1
                surface_points.append([x, y, z])

        return np.array(surface_points)


def vdw_surface(coordinates: np.ndarray, element_list: list, scale_factor: float,
                density: float, radii: dict) -> np.ndarray:
    """ Computes a molecular surface at points extended from the atoms' van der
        Waals radii.

        This is done using the Connolly [1] approach. As stated by Besler et al. [2],
        "Such a method is the surface generation algorithm of Connolly. It
        computes a spherical surface of points around each atom at a specified
        multiple of the atoms' van der Waals radius and density. The molecular
        surface is then constructed by taking the union of all of the atom
        surfaces and eliminating those points that are within the specified
        multiple of the van derWaals radius of any of the atoms."

        Args:
            coordinates  : cartesian coordinates of the nuclei (Angstroms)
            element_list : element symbols (e.g., C, H)
            scale_factor : scaling factor - the points on the molecular surface are
                           set at a distance of scale_factor * vdw_radius away from
                           each of the atoms. Recommended scaling factors are
                           1.4, 1.6, 1.8, 2.0 [3]
            density      : The (approximate) number of points to generate per Angstrom^2
                           of surface area. 1.0 is recommended [2].
            radii        : VDW radii

        Returns:
            surface_points (ndarray) : coordinates of the points on the extended surface

        Dependencies:
            numpy

        References:
        1. Connolly, M. L. Analytical Molecular-surface Calculation Journal of Applied
            Crystallography, 1983, 16, 548-558
        2. Besler, B. H.; Merz Jr., K. M. & Kollman, P. A. Atomic charges derived from
            semiempirical methods J. Comput. Chem., 1990, 11, 431-439
        3. Singh, U. C. & Kollman, P. A. An approach to computing electrostatic charges
            or molecules J. Comput. Chem., John Wiley & Sons, Ltd, 1984, 5, 129-145

        Also related:
        4. Bayly, C. I.; Cieplak, P.; Cornell, W. & Kollman, P. A. A well-behaved
            electrostatic potential based method using charge restraints for deriving
            atomic charges: the RESP model J. Phys. Chem., 1993, 97, 10269-10280
    """
    if not isinstance(coordinates, np.ndarray):
        raise TypeError(f'The coordinate were not given as an ndarray (i.e., {coordinates} variable).')
    elif not isinstance(element_list, list):
        raise TypeError(f'The elements were not given as a list (i.e., {element_list} variable).')
    elif not isinstance(scale_factor, float):
        raise TypeError(f'The scale_factor was not given as a float (i.e., {scale_factor} variable).')
    elif not isinstance(density, float):
        raise TypeError(f'The density was not given as a float (i.e., {density} variable).')
    elif not isinstance(radii, dict):
        raise TypeError(f'The radii were not given as a dictionary (i.e., {radii} variable).')
    else:
        radii_scaled = {}
        surface_points = []

        # scale radii
        for element in element_list:
            radii_scaled[element] = radii[element] * scale_factor

        # loop over atomic coordinates
        for i in range(len(coordinates)):

            # calculate approximate number of ESP grid points
            n_points = int(density * 4.0 * np.pi * np.power(radii_scaled[element_list[i]], 2))  # why 4.0?

            # generate an array of n_points in a unit sphere around the atom
            dots = surface(n_points=n_points)

            # scale the unit sphere by the VDW radius and translate
            dots = coordinates[i] + radii_scaled[element_list[i]] * dots

            # determine which points should be included or removed due to overlaps
            for j in range(len(dots)):
                save = True
                for k in range(len(coordinates)):
                    if i == k:
                        continue

                    # exclude points within the scaled VDW radius of other atoms
                    d = np.linalg.norm(dots[j] - coordinates[k])

                    if d < radii_scaled[element_list[k]]:
                        save = False
                        break
                if save:
                    surface_points.append(dots[j])

        # could also return radii_scaled if desired
        return np.array(surface_points)
