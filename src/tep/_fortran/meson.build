# Fortran extension subpackage

# Install __init__.py
py.install_sources(
  ['__init__.py'],
  subdir: 'tep/_fortran'
)

if have_fortran
  # Path to fortran sources relative to project root
  fortran_src = meson.project_source_root() / 'fortran'

  # Generate f2py wrappers using signature file for common block exposure
  # The pyf file defines the Python interface, Fortran sources are compiled separately
  teprob_wrapper = custom_target('teprob_wrapper',
    input: [fortran_src / 'teprob.pyf'],
    output: ['teprobmodule.c', 'teprob-f2pywrappers.f'],
    command: [py, '-m', 'numpy.f2py',
              '@INPUT@',
              '--lower',
              '--build-dir', '@OUTDIR@']
  )

  # Get fortranobject.c path
  fortranobject_c = incdir_f2py / 'fortranobject.c'

  # Build the extension module
  py.extension_module('teprob',
    [fortran_src / 'teprob.f', fortran_src / 'temain_mod.f', teprob_wrapper, fortranobject_c],
    include_directories: inc_np,
    dependencies: py_dep,
    subdir: 'tep/_fortran',
    install: true,
  )
endif
