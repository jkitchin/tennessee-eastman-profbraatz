# Install Python source files
py.install_sources(
  [
    '__init__.py',
    'constants.py',
    'controllers.py',
    'dashboard.py',
    'dashboard_dash.py',
    'fortran_backend.py',
    'simulator.py',
  ],
  subdir: 'tep'
)

# Install _fortran subpackage
py.install_sources(
  ['_fortran/__init__.py'],
  subdir: 'tep/_fortran'
)

# Build Fortran extension if available - handled in fortran/meson.build
# The extension is built there and installed to tep/_fortran/
