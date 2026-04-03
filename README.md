# RobustMHD

A robust finite element method for magnetohydrodynamics (MHD) based on H(curl)-conforming discretizations, designed to ensure stability with respect to pressure, fluid Reynolds number, and magnetic Reynolds number.

**Author:** Enrico Zampa, University of Vienna  

---

## Requirements

- NGSolve 6.2.2504

---

## Usage

Run numerical experiments from the command line. Example:

```bash
python3 MHD_solver.py --ictype 1 --NMAX 20 --order 2 --nu 1e-3 --eta 1e-4 --dt 0.1




