#!/usr/bin/env python

'''
Mean field with k-points sampling

The 2-electron integrals are computed using Poisson solver with FFT by default.
In most scenario, it should be used with pseudo potential.
'''

from pyscf.pbc import gto, scf, dft
import os
import numpy


def maybe_to_gpu(mf, label):
    if os.environ.get('PYSCF_EXAMPLE_USE_GPU') != '1':
        return mf
    to_gpu = getattr(mf, 'to_gpu', None)
    if not callable(to_gpu):
        print('%s: GPU4PySCF unavailable for %s' % (label, mf.__class__.__name__))
        return mf
    try:
        gpu_mf = to_gpu()
    except Exception as err:
        print('%s: GPU4PySCF conversion failed: %s' % (label, err))
        return mf
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
    basis = 'gth-szv',
    pseudo = 'gth-pade',
    verbose = 4,
)

nk = [4,4,4]  # 4 k-points for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(nk)

# Note: the default JK builder is slow for KHF calculations. Changing to
# density fitting or rsjk builder can be more efficient (see example
# 21-k_points_all_electron_scf.py)
kmf = maybe_to_gpu(scf.KRHF(cell, kpts), 'KRHF 4x4x4')
kmf.kernel()

kmf = dft.KRKS(cell, kpts)
# Turn to the atomic grids if you like
kmf.grids = dft.gen_grid.BeckeGrids(cell)
kmf.xc = 'm06,m06'
kmf = maybe_to_gpu(kmf, 'KRKS m06 4x4x4')
kmf.kernel()


#
# Second order SCF solver can be used in the PBC SCF code the same way in the
# molecular calculation
#
mf = maybe_to_gpu(scf.KRHF(cell, kpts).newton(), 'Newton KRHF 4x4x4')
mf.kernel()

# When you are using the newton solver for the pbc mean-field, make sure
# you are setting the same exxdiv for mean-field object before and after decorating the mean-field
# with newton solver. Otherwise define it before decorating it with the newton solver.

# For more details see: https://github.com/pyscf/pyscf/issues/3108

# This can be done as follow: mf_opt1 and mf_opt2 are equivalent. However mf_opt3 can result in different energies.

mf_opt1 = maybe_to_gpu(scf.KRHF(cell, kpts, exxdiv = None).density_fit().newton(), 'density-fit Newton KRHF exxdiv=None')
# mf_opt1.kernel()

mf_opt2 = scf.KRHF(cell, kpts).density_fit().newton()
mf_opt2.exxdiv = None
mf_opt2._scf.exxdiv = None
# mf_opt2.kernel()

mf_opt3 = scf.KRHF(cell, kpts).density_fit().newton()
mf_opt3.exxdiv = None
# mf_opt3.kernel()
