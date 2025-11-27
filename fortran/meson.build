# f2py wrapper generation and Fortran extension build

# Generate f2py wrappers
teprob_wrapper = custom_target('teprob_wrapper',
  input: ['teprob.f', 'temain_mod.f'],
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
  ['teprob.f', 'temain_mod.f', teprob_wrapper, fortranobject_c],
  include_directories: inc_np,
  dependencies: py_dep,
  subdir: 'tep/_fortran',
  install: true,
)
