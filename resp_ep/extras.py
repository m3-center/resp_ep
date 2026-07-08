import copy
import os

import numpy as np


def parse_xyz(filepath: str):
    """ Parses an XYZ file and returns atom names and coordinates.

        Args:
            filepath: The path to the XYZ file.

        Returns:
            tuple: A tuple containing:
                - list: A list of atom names (strings).
                - numpy.ndarray: A 2D NumPy array of coordinates (N x 3), where N is the number of atoms.
    """
    if not os.path.exists(filepath):
        raise ValueError(f'File not found: {filepath}')

    atom_names = []
    coords = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

        if len(lines) < 2:
            raise ValueError(f"XYZ file '{filepath}' is too short. Expected at least 2 lines (num_atoms, comment).")

        try:
            num_atoms = int(lines[0].strip())
        except ValueError:
            raise ValueError(f"Could not parse number of atoms from first line of '{filepath}'.")

        if len(lines) < num_atoms + 2:
            raise ValueError(f"Number of atom lines ({len(lines) - 2}) in '{filepath}' does not match declared number of atoms ({num_atoms}).")

        for i in range(2, num_atoms + 2): # Start from the third line (index 2)
            parts = lines[i].strip().split()
            if len(parts) < 4:
                print(f"Warning: Skipping malformed line in '{filepath}': '{lines[i].strip()}'")
                continue # Skip malformed lines

            try:
                atom_names.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                print(f"Warning: Skipping line with non-numeric coordinates in '{filepath}': '{lines[i].strip()}'")
                continue # Skip lines with non-numeric coordinates

    if len(coords) != num_atoms:
         raise ValueError(f"Number of successfully parsed atom lines ({len(coords)}) does not match declared number of atoms ({num_atoms}) in '{filepath}'.")

    return atom_names, np.array(coords)


def write_xyz(elements: list, xyz: list, output_filepath: str):
    """ Function that creates an XYZ-formatted file.

        Args:
            elements (list of strings): elements in the molecule
            xyz (nested list of floats): xyz coordinates for each element
            output_filepath: name of the output file to be created

        Returns:
            output_filepath (file): XYZ-formatted structure file

        Library requirement:
            copy
    """
    if not isinstance(elements, list):
        raise TypeError(f'The value for elements ({elements}) is not a list.')
    elif not isinstance(xyz, list):
        raise TypeError(f'The value for xyz ({xyz}) is not a list.')
    elif not isinstance(output_filepath, str):
        raise TypeError(f'The value for output_filepath ({output_filepath}) is not a string.')
    else:
        natoms = len(elements)
        elem_xyz = copy.deepcopy(xyz)
        for index in range(0, len(elem_xyz)):
            elem_xyz[index].insert(0, elements[index])

        with open(output_filepath, 'wt') as out:
            out.write(f'{natoms}\n\n')
            for atom_line in elem_xyz:
                out.write(f'{" ".join(map(str, atom_line))}\n')


def translate_point_along_vector(vector: np.ndarray,
                                 initial_point: np.ndarray,
                                 distance: float) -> np.ndarray:
    """ Compute a point translated from `origin` by `distance` along `vector`.

        Args:
            vector: 1D array of shape (3,) defining the directional vector.
            initial_point: 1D array of shape (3,) defining the starting position.
            distance: Translation distance.

        Returns:
            np.ndarray: 1D array of shape (3,) representing the new point coordinates.
    """
    if not isinstance(vector, np.ndarray):
        raise TypeError("Input vector is not a numpy array.")
    elif not isinstance(initial_point, np.ndarray):
        raise TypeError("initial_point vector is not a numpy array.")
    elif not isinstance(distance, (int, float)):
        raise TypeError("Distance must be a numeric type.")
    else:
        # Flatten inputs just in case they are wrapped in extra dimensions
        v = vector.ravel()
        p0 = initial_point.ravel()

        # Calculate vector norm safely
        norm = np.linalg.norm(v)
        if norm == 0.0:
            raise ValueError("Cannot move along a zero-length vector.")

        new_point = p0 - (float(distance) / norm) * v

        return new_point


def unit_vector(input_vector: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    """Return the unit-normalized version of a 3D vector.

    Parameters:
        input_vector : Input vector with shape (3,)
        atol : Absolute tolerance for the vector norm.
               If ||input_vector|| <= atol, raise ValueError.

    Returns:
        A unit vector with shape (3,) 
    """
    if not isinstance(input_vector, np.ndarray):
        raise TypeError("Input vector is not a numpy array.")
    else:
        vector = np.asarray(input_vector, dtype=float).ravel()
        n = np.linalg.norm(vector)
        if n <= atol:
            raise ValueError("Cannot normalize a near-zero vector.")
        return vector / n


def add_sigma_hole(molecule: str,
                   atom_pairs: list,
                   distance: int|float,
                   out_suffix: str = "_x",
                   ep_label: str = "x",
                   work_dir: str = "./"
                  ):
    """ Append extra point centers to an XYZ file by extending selected bonds.
        
        For each pair (i, j) in `atom_pairs` (1-indexed), a extra point center is placed
        at a fixed distance from atom j along the bond axis defined by atoms i and j.
        By convention, the point is placed starting at atom j and extending *away from atom i*
        (i.e., along the direction from i -> j).

        Args:
            molecule:
                Base name of the input XYZ file (without '.xyz').
            atom_pairs: list[list[int]]
                1-indexed atom pairs defining bond axis (e.g., [[1, 2], [3, 4]]).
            distance:
                Distance from atom j to the extra point centers (Å).
            out_suffix:
                Suffix appended to the output filename.
            ep_label:
                Label written for extra point centers in the XYZ output.
            work_dir:
                Input/output directory.

        Writes:
            {work_dir}/{molecule}{out_suffix}.xyz
                XYZ file with dummy atoms appended.
    """
    if not isinstance(molecule, str):
        raise TypeError('The parameter passed molecule is not a string.')
    elif not isinstance(atom_pairs, list):
        raise TypeError('The parameter passed atom_pairs is not a list.')
    elif not isinstance(distance, (int, float)):
        raise TypeError('The parameter passed distance must be a number.')
    else:
        atoms_xyz_rows = []
        extra_points = []

        with open(f'{work_dir}/{molecule}.xyz') as input_file:
            next(input_file)  # skip atom count
            next(input_file)  # skip comment line
            for line in input_file:
                parts = line.strip().split()
                if parts:
                    atoms_xyz_rows.append(parts)

        for atom_pair in atom_pairs:
            atom1_index = atom_pair[0]
            atom2_index = atom_pair[1]

            atom_1 = np.array([float(a) for a in atoms_xyz_rows[atom1_index - 1][1:]])
            atom_2 = np.array([float(a) for a in atoms_xyz_rows[atom2_index - 1][1:]])

            bond_vector = atom_1 - atom_2

            ep = translate_point_along_vector(vector=bond_vector,
                                              initial_point=atom_2,
                                              distance=distance)
            extra_points.append(ep)

        with open(f'{work_dir}/{molecule}{out_suffix}.xyz', 'w') as outfile:
            outfile.write(f"{len(atoms_xyz_rows) + len(extra_points)}\n")
            outfile.write(f"{molecule} with extra points added\n")

            for atom_line in atoms_xyz_rows:
                outfile.write(f"{' '.join(atom_line)}\n")

            for entry in extra_points:
                outfile.write(f'{ep_label} {entry[0]:.8f} {entry[1]:.8f} {entry[2]:.8f}\n')

        print(f'Created: {molecule}{out_suffix}.xyz')


def add_lone_pairs(molecule: str,
                   lp_specs: list[dict],
                   lp_distance: int|float,
                   lp_angle_deg: int|float,
                   mode: str,
                   out_suffix: str = "_x",
                   ep_label: str = "x",
                   work_dir: str = "./"
                  ):
    """ Append lone-pair extra point sites (two points per center) to an XYZ file.

        Each entry in `lp_specs` defines a central atom (1-indexed) and either:
        - Two-Neighbor Mode (e.g., sp3 alcohol/ether oxygen)
            {"center": c, "neighbors": [n1, n2]}
            Places two lone-pair extra points at distance `lp_distance` from the center,
            separated by `lp_angle_deg`, oriented relative to the neighbor-center-neighbor plane.

        - One-Neighbor Mode (e.g., sp2 carbonyl oxygen):
            {"center": c, "neighbors": [n1], "plane_atom": p}
            Defines a plane using (center, neighbor, plane_atom) and places both points in that plane.
        
        Args:
            molecule: 
                Base name of the input XYZ file (without '.xyz').
            lp_specs:
                Lone-pair placement specifications (all atom indices are 1-indexed).
            lp_distance: 
                Distance from the center atom to each lone-pair point (Å).
            lp_angle_deg: 
                Angle between the two lone-pair directions (degrees).
            mode:
                Controls extra point pair placement. 
                Options are "out_of_plane" (good for sp3 atoms) or "in_plane" (good for sp2 atoms).
             out_suffix: 
                Suffix appended to the output filename
            ep_label: 
                Label written for lone-pair dummy atoms in the XYZ output.
           work_dir: 
                The directory path containing the input file and where the output will be saved.

        Writes
            {work_dir}/{molecule}{out_suffix}.xyz
                XYZ file with lone-pair extra points appended.
    """            
    if not isinstance(molecule, str):
        raise TypeError('The parameter passed molecule is not a string.')
    elif not isinstance(lp_distance, (int, float)):
        raise TypeError("The lp_distance must be a number.")
    elif not isinstance(lp_angle_deg, (int, float)):
        raise TypeError("The lp_angle_deg must be a number.")
    else:
        lp_distance = float(lp_distance)
        half_angle = np.deg2rad(float(lp_angle_deg)) / 2.0

        all_atoms = []
        with open(f"{work_dir}/{molecule}.xyz") as input_file:
            next(input_file)
            next(input_file)
            for line in input_file:
                parts = line.split()
                if parts:
                    all_atoms.append(parts)

        coords = np.array([[float(a) for a in row[1:4]] for row in all_atoms], dtype=float)

        extra_points = []

        for spec in lp_specs:
            c = coords[spec["center"] - 1]
            neigh = spec.get("neighbors", [])
            if len(neigh) not in (1, 2):
                raise ValueError("The neighbors must have length 1 or 2.")

            n1 = coords[neigh[0] - 1]
            u1 = unit_vector(n1 - c)

            if len(neigh) == 2:
                assumed_mode = spec.get("mode", mode)  # per-spec override
                if assumed_mode not in {"out_of_plane", "in_plane"}:
                    raise ValueError("The mode must be 'out_of_plane' or 'in_plane'.")

                n2 = coords[neigh[1] - 1]
                u2 = unit_vector(n2 - c)

                # bisector points between substituents; EPs go opposite
                bis = u1 + u2
                if np.linalg.norm(bis) == 0.0:
                    raise ValueError("Opposite bonds: cannot define bisector.")
                b = -unit_vector(bis)

                # plane normal
                n = np.cross(u1, u2)
                if np.linalg.norm(n) == 0.0:
                    raise ValueError("Colinear neighbors: cannot define plane normal.")
                n_hat = unit_vector(n)

                if assumed_mode == "in_plane":
                    # spread within the bond plane
                    e_perp = unit_vector(np.cross(n_hat, b))
                    d1 = np.cos(half_angle) * b + np.sin(half_angle) * e_perp
                    d2 = np.cos(half_angle) * b - np.sin(half_angle) * e_perp
                elif assumed_mode == "out_of_plane":
                    # spread above/below the bond plane
                    d1 = np.cos(half_angle) * b + np.sin(half_angle) * n_hat
                    d2 = np.cos(half_angle) * b - np.sin(half_angle) * n_hat
                else:
                    raise ValueError("The mode must be 'out_of_plane' or 'in_plane'.")
            else:
                if "plane_atom" not in spec:
                    raise ValueError("The one-neighbor mode requires a plane_atom.")

                p = coords[spec["plane_atom"] - 1]
                u_plane = p - c

                n = np.cross(u1, u_plane)
                if np.linalg.norm(n) == 0.0:
                    raise ValueError("Cannot define plane: center, neighbor, plane_atom are colinear.")
                n_hat = unit_vector(n)

                b = -u1
                e_perp = unit_vector(np.cross(n_hat, b))

                d1 = np.cos(half_angle) * b + np.sin(half_angle) * e_perp
                d2 = np.cos(half_angle) * b - np.sin(half_angle) * e_perp

            lp1 = c + lp_distance * d1
            lp2 = c + lp_distance * d2
            extra_points.extend([lp1, lp2])

        out_path = f"{work_dir}/{molecule}{out_suffix}.xyz"
        with open(out_path, "w") as out:
            out.write(f"{len(all_atoms) + len(extra_points)}\n")
            out.write(f"{molecule} with lone-pair extra point centers added\n")
            for row in all_atoms:
                out.write(" ".join(row) + "\n")
            for p in extra_points:
                out.write(f"{ep_label} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

        print(f"Created: {molecule}{out_suffix}.xyz")


def test(extent='full', extras=None):
    """Runs a test suite through pytest.

    Parameters
    ----------
    extent : {'smoke', 'quick', 'full', 'long'}
        All choices are defined, but choices may be redundant in some projects.
        _smoke_ will be minimal "is-working?" test(s).
        _quick_ will be as much coverage as can be got quickly, approx. 1/3 tests.
        _full_ will be the whole test suite, less some exceedingly long outliers.
        _long_ will be the whole test suite.
    extras : list
        Additional arguments to pass to `pytest`.

    Returns
    -------
    int
        Return code from `pytest.main()`. 0 for pass, 1 for fail.

    """
    try:
        import pytest
    except ImportError:
        raise RuntimeError('Testing module `pytest` is not installed. Run `conda install pytest`.')
    abs_test_dir = os.path.sep.join([os.path.abspath(os.path.dirname(__file__)), "tests"])

    command = ['-rws', '-v']
    if extent.lower() in ['smoke', 'quick', 'full', 'long']:
        pass
    if extras is not None:
        command.extend(extras)
    command.extend(['--capture=sys', abs_test_dir])

    retcode = pytest.main(command)
    return retcode
