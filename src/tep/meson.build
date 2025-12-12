# Install Python source files
py.install_sources(
  [
    '__init__.py',
    'cli.py',
    'constants.py',
    'controller_base.py',
    'controller_plugins.py',
    'controllers.py',
    'dashboard_dash.py',
    'detector_base.py',
    'detector_plugins.py',
    'fortran_backend.py',
    'python_backend.py',
    'simulator.py',
  ],
  subdir: 'tep'
)

# Install assets directory
install_subdir('assets', install_dir: py.get_install_dir() / 'tep')

# Process _fortran subpackage
subdir('_fortran')
