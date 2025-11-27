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

# Build Fortran extension if available
if have_fortran
  # Path to fortran sources relative to project root
  fortran_dir = meson.project_source_root() / 'fortran'

  # Generate f2py wrappers
  teprob_wrapper = custom_target('teprob_wrapper',
    input: [fortran_dir / 'teprob.f', fortran_dir / 'temain_mod.f'],
    output: ['teprobmodule.c', 'teprob-f2pywrappers.f'],
    command: [py, '-m', 'numpy.f2py',
              '@INPUT@',
              '-m', 'teprob',
              '--lower',
              '--build-dir', '@OUTDIR@']
  )

  # Get fortranobject.c path
  fortranobject_c = incdir_f2py / 'fortranobject.c'

  # Build the extension module
  py.extension_module('teprob',
    [fortran_dir / 'teprob.f', fortran_dir / 'temain_mod.f', teprob_wrapper, fortranobject_c],
    include_directories: inc_np,
    dependencies: py_dep,
    subdir: 'tep/_fortran',
    install: true,
  )
endif
