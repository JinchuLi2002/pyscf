#!/usr/bin/env python

'''
Hartree-Fock/DFT with k-points sampling for all-electron calculations

GDF (Gaussian density fitting), MDF (mixed density fitting), RSGDF
(range-separated Gaussian density fitting), or RS-JK builder
can be used in all electron calculations. They are more efficient than the
default SCF JK builder.
'''

import numpy
from pyscf.pbc import gto, scf, dft


def require_gpu(mf, label):
    to_gpu = getattr(mf, 'to_gpu', None)
    if not callable(to_gpu):
        raise RuntimeError(
            '%s: GPU4PySCF is required, but %s has no to_gpu() method'
            % (label, mf.__class__.__name__)
        )
    gpu_mf = to_gpu()
    print('%s: using GPU4PySCF' % label)
    return gpu_mf


cell = gto.M(
    a = numpy.eye(3)*3.5668,
    atom = '''C     0.      0.      0.
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    basis = '6-31g',
    verbose = 4,
)

nk = [4,4,4]  # 4 k-poins for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(nk)

#
# Mixed density fitting
#
kmf = scf.KRHF(cell, kpts).mix_density_fit()
# In the MDF scheme, modifying the default mesh for PWs to reduce the cost
# The default mesh for PWs is a very dense-grid scheme which is automatically
# generated based on the AO basis. It is often not necessary to use dense grid
# for MDF method.
kmf.with_df.mesh = [11,11,11]
kmf = require_gpu(kmf, 'mixed-density-fit KRHF 4x4x4')
kmf.kernel()

#
# Density fitting
#
kmf = dft.KRKS(cell, kpts).density_fit(auxbasis='weigend')
kmf.xc = 'bp86'
kmf = require_gpu(kmf, 'density-fit KRKS bp86 4x4x4')
kmf.kernel()

#
# Range-separated density fitting (RSDF)
# RSDF uses the same amount of memory & disk as GDF and achieves a similar
# accuracy as GDF but is often 5~10x faster than GDF in the DF initialization
# step. The following run should give an energy very close to the one above.
# see '35-range_separated_density_fitting.py' for more details of RSDF.
#
kmf = dft.KRKS(cell, kpts).rs_density_fit(auxbasis='weigend')
kmf.xc = 'bp86'
kmf = require_gpu(kmf, 'rs-density-fit KRKS bp86 4x4x4')
kmf.kernel()

#
# RS-JK builder is efficient for large number of k-points
#
kmf = scf.KRHF(cell, kpts).jk_method('RS')
kmf = require_gpu(kmf, 'RS-JK KRHF 4x4x4')
kmf.kernel()

#
# Second order SCF solver can be used in the PBC SCF code the same way in the
# molecular calculation.  Note second order SCF algorithm does not support
# smearing method.
#
mf = scf.KRHF(cell, kpts).density_fit()
mf = mf.newton()
mf = require_gpu(mf, 'density-fit Newton KRHF 4x4x4')
mf.kernel()
