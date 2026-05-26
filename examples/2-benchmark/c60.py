#!/usr/bin/env python

import pyscf
from pyscf.tools import c60struct
from benchmarking_utils import setup_logger, get_cpu_timings

log = setup_logger()


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


for bas in ('6-31g**', 'cc-pVTZ'):
    mol = pyscf.M(atom=[('C', r) for r in c60struct.make60(1.46,1.38)],
                  basis=bas,
                  max_memory=40000)

    cpu0 = get_cpu_timings()
    mf = pyscf.scf.fast_newton(require_gpu(mol.RHF(), 'SOSCF/%s' % bas))
    cpu0 = log.timer('SOSCF/%s' % bas, *cpu0)

    mf = require_gpu(mol.RHF().density_fit(), 'density-fitting-HF/%s' % bas).run()
    cpu0 = log.timer('density-fitting-HF/%s' % bas, *cpu0)
