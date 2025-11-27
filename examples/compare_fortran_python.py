#!/usr/bin/env python
"""Compare f2py Fortran backend against standalone Fortran executable.

This script verifies that the f2py wrapper produces identical results
to the original compiled Fortran code (within machine precision).

Usage:
    python examples/compare_fortran_python.py

Requirements:
    - tep package must be installed with Fortran backend
    - gfortran must be available for compiling standalone Fortran test
"""

import numpy as np
import subprocess
import sys
import os
from pathlib import Path

from tep import TEPSimulator
from tep.simulator import ControlMode


def compile_fortran_test(source_dir: Path, output_file: Path) -> bool:
    """Compile standalone Fortran test program.

    Args:
        source_dir: Directory containing Fortran source files
        output_file: Path for compiled executable

    Returns:
        True if compilation succeeded
    """
    fortran_test = '''
      PROGRAM TEP_TEST
      IMPLICIT NONE

      DOUBLE PRECISION YY(50), YP(50), TIME
      DOUBLE PRECISION XMEAS(41), XMV(12)
      INTEGER NN, I, STEP

      COMMON/PV/XMEAS, XMV

      NN = 50
      TIME = 0.0D0

      CALL TEINIT(NN, TIME, YY, YP)

      WRITE(*,'(A)') 'INITIAL_XMEAS:'
      DO I = 1, 41
        WRITE(*,'(I3,1X,E25.16)') I, XMEAS(I)
      END DO

      WRITE(*,'(A)') 'INITIAL_XMV:'
      DO I = 1, 12
        WRITE(*,'(I3,1X,E25.16)') I, XMV(I)
      END DO

      WRITE(*,'(A)') 'INITIAL_YY:'
      DO I = 1, 50
        WRITE(*,'(I3,1X,E25.16)') I, YY(I)
      END DO

      WRITE(*,'(A)') 'INITIAL_YP:'
      DO I = 1, 50
        WRITE(*,'(I3,1X,E25.16)') I, YP(I)
      END DO

      DO STEP = 1, 3600
        CALL TEFUNC(NN, TIME, YY, YP)
        DO I = 1, 50
          YY(I) = YY(I) + YP(I) * (1.0D0/3600.0D0)
        END DO
        TIME = TIME + 1.0D0/3600.0D0
      END DO

      WRITE(*,'(A)') 'FINAL_XMEAS:'
      DO I = 1, 41
        WRITE(*,'(I3,1X,E25.16)') I, XMEAS(I)
      END DO

      WRITE(*,'(A)') 'FINAL_XMV:'
      DO I = 1, 12
        WRITE(*,'(I3,1X,E25.16)') I, XMV(I)
      END DO

      WRITE(*,'(A)') 'FINAL_YY:'
      DO I = 1, 50
        WRITE(*,'(I3,1X,E25.16)') I, YY(I)
      END DO

      END PROGRAM
'''

    test_file = Path('/tmp/test_tep_compare.f90')
    test_file.write_text(fortran_test)

    try:
        result = subprocess.run(
            ['gfortran', '-o', str(output_file), str(test_file),
             str(source_dir / 'teprob.f')],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Compilation error: {result.stderr}")
            return False
        return True
    except FileNotFoundError:
        print("gfortran not found. Install with: brew install gcc (macOS)")
        return False


def run_fortran_test(executable: Path) -> dict:
    """Run standalone Fortran test and parse output.

    Returns:
        Dict with initial/final xmeas, xmv, yy values
    """
    result = subprocess.run([str(executable)], capture_output=True, text=True)

    data = {
        'initial_xmeas': np.zeros(41),
        'initial_xmv': np.zeros(12),
        'initial_yy': np.zeros(50),
        'initial_yp': np.zeros(50),
        'final_xmeas': np.zeros(41),
        'final_xmv': np.zeros(12),
        'final_yy': np.zeros(50),
    }

    current_section = None
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line.startswith('INITIAL_XMEAS'):
            current_section = 'initial_xmeas'
        elif line.startswith('INITIAL_XMV'):
            current_section = 'initial_xmv'
        elif line.startswith('INITIAL_YY'):
            current_section = 'initial_yy'
        elif line.startswith('INITIAL_YP'):
            current_section = 'initial_yp'
        elif line.startswith('FINAL_XMEAS'):
            current_section = 'final_xmeas'
        elif line.startswith('FINAL_XMV'):
            current_section = 'final_xmv'
        elif line.startswith('FINAL_YY'):
            current_section = 'final_yy'
        elif current_section and line:
            parts = line.split()
            if len(parts) >= 2:
                idx = int(parts[0]) - 1
                val = float(parts[1].replace('D', 'E'))
                data[current_section][idx] = val

    return data


def run_f2py_test() -> dict:
    """Run f2py wrapped simulation and return results.

    Returns:
        Dict with initial/final xmeas, xmv, yy values
    """
    sim = TEPSimulator(
        random_seed=1234,  # Default seed in Fortran
        control_mode=ControlMode.OPEN_LOOP
    )
    sim.initialize()

    data = {
        'initial_xmeas': sim.get_measurements().copy(),
        'initial_xmv': sim.get_manipulated_vars().copy(),
        'initial_yy': sim.get_states().copy(),
    }

    # Run 3600 steps (1 hour)
    result = sim.simulate(duration_hours=1.0, record_interval=3600)

    data['final_xmeas'] = result.measurements[-1].copy()
    data['final_xmv'] = result.manipulated_vars[-1].copy()
    data['final_yy'] = result.states[-1].copy()

    return data


def compare_results(fortran_data: dict, f2py_data: dict) -> bool:
    """Compare Fortran and f2py results.

    Returns:
        True if results match within machine precision
    """
    print(f"\n{'='*70}")
    print("COMPARISON: f2py Wrapper vs Standalone Fortran Executable")
    print(f"{'='*70}")

    all_ok = True

    comparisons = [
        ('Initial XMEAS', 'initial_xmeas', 41),
        ('Initial XMV', 'initial_xmv', 12),
        ('Initial States', 'initial_yy', 50),
        ('Final XMEAS', 'final_xmeas', 41),
        ('Final XMV', 'final_xmv', 12),
        ('Final States', 'final_yy', 50),
    ]

    for name, key, size in comparisons:
        f_data = fortran_data[key]
        p_data = f2py_data[key]

        max_diff = np.max(np.abs(f_data - p_data))

        # Calculate relative difference for non-zero values
        nonzero_mask = np.abs(f_data) > 1e-10
        if np.any(nonzero_mask):
            rel_diff = np.max(np.abs(f_data[nonzero_mask] - p_data[nonzero_mask]) /
                            np.abs(f_data[nonzero_mask]))
        else:
            rel_diff = max_diff

        # Machine precision is ~1e-15, but floating point accumulation
        # over 3600 steps can lead to ~1e-10 differences
        ok = max_diff < 1e-8
        status = "OK" if ok else "DIFF"
        all_ok = all_ok and ok

        print(f"{name:20} Max Abs Diff: {max_diff:12.4e}  Rel Diff: {rel_diff:12.4e}  [{status}]")

        # Show first few differences if any significant
        if not ok:
            print(f"  First differences:")
            for i in range(min(5, size)):
                if abs(f_data[i] - p_data[i]) > 1e-10:
                    print(f"    [{i+1}] Fortran: {f_data[i]:.10e}  f2py: {p_data[i]:.10e}")

    print(f"\n{'='*70}")
    if all_ok:
        print("RESULT: f2py wrapper produces IDENTICAL results to Fortran")
        print("        (within floating-point machine precision)")
    else:
        print("RESULT: DIFFERENCES FOUND between f2py wrapper and Fortran")
        print("        Check compilation flags or random seed initialization.")
    print(f"{'='*70}")

    return all_ok


def main():
    """Main comparison routine."""
    print("TEP f2py vs Fortran Verification")
    print("=" * 70)

    # Find source directory
    script_dir = Path(__file__).parent.parent
    source_dir = script_dir

    if not (source_dir / 'teprob.f').exists():
        print(f"Error: teprob.f not found in {source_dir}")
        print("Run this script from the package root directory.")
        return 1

    # Compile Fortran test
    print("\n1. Compiling standalone Fortran test...")
    executable = Path('/tmp/tep_compare_test')
    if not compile_fortran_test(source_dir, executable):
        return 1
    print("   Compilation successful.")

    # Run Fortran test
    print("\n2. Running standalone Fortran simulation (1 hour)...")
    fortran_data = run_fortran_test(executable)
    print("   Fortran simulation complete.")

    # Run f2py test
    print("\n3. Running f2py wrapped simulation (1 hour)...")
    f2py_data = run_f2py_test()
    print("   f2py simulation complete.")

    # Compare
    success = compare_results(fortran_data, f2py_data)

    # Cleanup
    executable.unlink(missing_ok=True)
    Path('/tmp/test_tep_compare.f90').unlink(missing_ok=True)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
