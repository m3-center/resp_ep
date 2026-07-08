# Restrained Electrostatic Potential (RESP) Charges with Extra Point Fitting
> Compute RESP partial atomic charges (PACs) from Psi4 electrostatic potentials. This tool supports fitting to off-center points, such as electron lone-pair regions and sigma holes, for improved electrostatic modeling.

**Author:** Karl N. Kirschner

**Based on the work of A Alenaizan, LA Burns, and CD Sherrill**: [https://github.com/cdsgroup/resp](https://github.com/cdsgroup/resp) (Alenaizan et al., 2020)

---

## Overview

This tool is a refactoring of the Python RESP code developed by Alenaizan et al., which is a plugin for the Psi4 quantum chemistry package. This revision extends the original code by enabling PAC fitting onto virtual sites (a.k.a., extra points, dummy particles), facilitating the modeling of molecules that exhibit strongly anisotropic electrostatics (e.g., $\sigma$-holes and electron lone pairs). Additionally, a standardized configuration workflow was introduced based on `.ini` inputs, enabling straightforward reproduction of workflows and their resulting PACs. The root mean square error (RMSE) and relative root mean square error (RRMSE) metrics are also included for fit quality assessment.

## Features

- **Psi4 Integration:** Grid and ESP generation are performed using the open-source **Psi4** quantum chemistry suite, or can be loaded directly from a previous calculation.
- **Automated Extra Point Placement:** Includes scripts to dynamically calculate and append $\sigma$-hole or lone-pair extra points (EP) directly onto standard XYZ files.
- **Easy Molecule Management:** Molecules are represented using simple human-readable standard XYZ format, with built-in XYZ parsing and writing utilities.
- **Flexible Fitting:** Multi-center formulation that includes both nuclei and extra points as charge sites.
- **Averaged Charges:** Support for using multiple conformers or orientations (Dupradeau et al.) to generate averaged charge sets.
- **Equivalency & Restraints:** Enforce charge equivalency and chemical constraints on specific atom groups.
- **Clean Configuration:** Molecular structures and fitting constraints are managed entirely through explicit `.ini` and XYZ files.
- **Quality Metrics:** Built-in calculation of RMSE and RRMSE metrics to evaluate fit quality.
- **vdW Radii Set Selection:** Supports both the legacy vdW radii parameters used in the original formulations and the radii set developed by Jorge Charry and Alexandre Tkatchenko.

## Dependencies

### Python standard library
The code was validated using Python 3.12.13.
- `copy`
- `os`
- `sys`

### Third-party libraries
Validated using the versions specified in parentheses.
- `Jupyter` (1.1.1)
- `NumPy` (2.5.1)
- `Psi4` (1.11; https://psicode.org)
- `Pytest` (8.3.5)
- `SciPy` (1.18.0)
- `Sphinx` (9.1.0)
- `sphinx-automodapi` (0.22.0)

## Installation

### 1. Automated Setup using Conda/Mamba (Recommended)
Clone the repository and create an isolated Conda/Mamba environment, installing NumPy, SciPy and Psi4, Jupyter, PyTest, and sphinx:

```bash
git clone https://github.com/m3-center/resp_ep.git
cd resp_ep
conda env create -f environment.yml
conda activate resp_ep
```
### 2. Manual Pip Installation

If you manage your own environment and have already pre-installed Psi4 on your system, you can link the repository directly:
```bash
git clone https://github.com/m3-center/resp_ep.git
cd resp_ep
pip install -e . --no-deps
```

---
## Repository Structure
The following files are included:
- `resp_ep/` - Core package source code
  - `driver.py`: Main execution driver and pipeline controller
  - `espfit.py`: Restrained electrostatic potential (RESP) fitting procedure and metrics
  - `extras.py`: Geometry generation scripts to automatically calculate and insert EPs that model $\sigma$-holes and electron lone pairs into XYZ data.
  - `vdw_surface.py`: van der Waals surface generation
  - `stage2_helper.py`: Helper utilities for two-stage fitting

- `tests/`
  - `test_resp_ep.py`: PyTest script containing different fitting scenarios:
  - `data/`: .xyz and .ini files
- `examples/`: Collection of Jupyter Notebooks and template scripts for specific molecules (e.g., acetic acid, 1-bromobutane, DMSO). **Note**: these are not meant to be recommended settings, but a collection of demonstrations that users can draw upon for devising their own modelling workflows.
- `docs/` - Configuration files for generating Sphinx API documentation.

## References
* [[BaylyCCK1993](https://pubs.acs.org/doi/abs/10.1021/j100142a004)] Bayly C. I., Cieplak, P., Cornell, W., Kollman, P.A. *A well-behaved electrostatic potential based method using charge restraints for deriving atomic charges: the RESP model.* *J. Phys. Chem.* **97**, 10269 (1993).
* [[DupradeauPZSLGLRC2010](https://doi.org/10.1039/c0cp00111b)] Dupradeau, F.Y., Pigache, A., Zaffran, T., Savineau, C., Lelong, R., Grivel, N., Lelong, D., Rosanski, W. and Cieplak, P. *The RED. Tools: Advances in RESP and ESP charge derivation and force field library building.* *Phys. Chem. Chem. Phys.* **12**, 7821 (2010).
* [[CharryT20024](https://doi.org/10.1021/acs.jctc.4c00784)] Charry, J. and Tkatchenko, A. *Van der Waals radii of free and bonded atoms from Hydrogen (Z=1) to Oganesson (Z=118).* *J. Chem. Theory Comput* **20**, 7469 (2024).

Please cite the following articles if you use this program:
* [[AlenaizanBS2020](https://doi.org/10.1002/qua.26035)] Alenaizan A., Burns L. A., Sherrill C. D. *Python implementation of the restrained electrostatic potential charge model.* *Int. J. Quantum Chem.* **120**, e26035 (2020).
* [[Kirschner2026](https://doi.org/10.26434/chemrxiv.15002675/v1)] Kirschner K. N. *Extending the Python RESP Framework for Extra Point Charge Fitting.* *ChemRxiv* 04 May 2026. DOI: https://doi.org/10.26434/chemrxiv.15002675/v1

## Disclaimer
This software is provided "as is," without warranty of any kind. While we have made every effort to ensure the accuracy and reliability of the code and associated files, the authors and contributors are not responsible for any errors, data loss, or inaccuracies that may occur through its use. Users are encouraged to validate results independently, especially for research intended for publication.
