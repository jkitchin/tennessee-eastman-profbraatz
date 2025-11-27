project('tep', 'c',
  version: '1.1.0',
  license: 'MIT',
  meson_version: '>=1.0.0',
)

# Optional Fortran language
fc_available = add_languages('fortran', required: get_option('fortran') == 'enabled')
have_fortran = fc_available and get_option('fortran') != 'disabled'

# Python setup
py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# NumPy includes for f2py
incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

incdir_f2py = run_command(py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

if have_fortran
  subdir('fortran')
endif

subdir('tep')
