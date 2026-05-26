#!/usr/bin/env python

import os

import pyscf
from pyscf.tools import c60struct
from benchmarking_utils import setup_logger, get_cpu_timings

log = setup_logger()


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


for bas in ('6-31g**', 'cc-pVTZ'):
    mol = pyscf.M(atom=[('C', r) for r in c60struct.make60(1.46,1.38)],
                  basis=bas,
                  max_memory=40000)

    cpu0 = get_cpu_timings()
    mf = pyscf.scf.fast_newton(maybe_to_gpu(mol.RHF(), 'SOSCF/%s' % bas))
    cpu0 = log.timer('SOSCF/%s' % bas, *cpu0)

    mf = maybe_to_gpu(mol.RHF().density_fit(), 'density-fitting-HF/%s' % bas).run()
    cpu0 = log.timer('density-fitting-HF/%s' % bas, *cpu0)
