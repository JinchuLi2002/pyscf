#!/usr/bin/env python
import pyscf
from pyscf.tools.mo_mapping import mo_comps
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


for bas in ('3-21g', '6-31g**', 'cc-pVTZ', 'ANO-Roos-TZ'):
    mol = pyscf.M(atom = '''
c   1.217739890298750 -0.703062453466927  0.000000000000000
h   2.172991468538160 -1.254577209307266  0.000000000000000
c   1.217739890298750  0.703062453466927  0.000000000000000
h   2.172991468538160  1.254577209307266  0.000000000000000
c   0.000000000000000  1.406124906933854  0.000000000000000
h   0.000000000000000  2.509154418614532  0.000000000000000
c  -1.217739890298750  0.703062453466927  0.000000000000000
h  -2.172991468538160  1.254577209307266  0.000000000000000
c  -1.217739890298750 -0.703062453466927  0.000000000000000
h  -2.172991468538160 -1.254577209307266  0.000000000000000
c   0.000000000000000 -1.406124906933854  0.000000000000000
h   0.000000000000000 -2.509154418614532  0.000000000000000
''',
                  basis = bas)
    cpu0 = get_cpu_timings()

    mf = require_gpu(mol.RHF(), 'C6H6 %s RHF' % bas).run()
    cpu0 = log.timer('C6H6 %s RHF'%bas, *cpu0)

    mymp2 = mf.MP2().run()
    cpu0 = log.timer('C6H6 %s MP2'%bas, *cpu0)

    mymc = mf.CASSCF(6, 6)
    idx_2pz = mo_comps('2pz', mol, mf.mo_coeff).argsort()[-6:]
    mo = mymc.sort_mo(idx_2pz, base=0)
    mymc.kernel(mo)
    cpu0 = log.timer('C6H6 %s CASSCF'%bas, *cpu0)

    mycc = mf.CCSD().run()
    cpu0 = log.timer('C6H6 %s CCSD'%bas, *cpu0)

    mf = require_gpu(mol.RKS(), 'C6H6 %s B3LYP' % bas).run(xc='b3lyp')
    cpu0 = log.timer('C6H6 %s B3LYP'%bas, *cpu0)

    mf = require_gpu(mf.density_fit(), 'C6H6 %s density-fit RKS' % bas).run()
    cpu0 = log.timer('C6H6 %s density-fit RHF'%bas, *cpu0)

