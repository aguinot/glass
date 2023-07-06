# author: Axel Guinot <axel.guinot.astro@gmail.com> ;
#         Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''GLASS module for CCL interoperability'''


import pyccl as ccl
import numpy as np


def matter_cls(cosmo, lmax, ws):
    '''Compute angular matter power spectra using CCL.

    NOTE:
    At the moment CCL do not support non-limber computation. This function will
    need to be updated when it does.
    '''

    ell = np.linspace(1, lmax+1, lmax+1)
    cls_ccl = []
    n_shells = len(ws)
    for i in range(0, n_shells):
        for j in range(i, -1, -1):
            zz_i = ws[i].za
            nz_i = ws[i].wa
            count_i = ccl.NumberCountsTracer(
                cosmo,
                dndz=(zz_i, nz_i),
                bias=(zz_i, np.ones_like(zz_i)),
                has_rsd=True
            )
            zz_j = ws[j].za
            nz_j = ws[j].wa
            count_j = ccl.NumberCountsTracer(
                cosmo,
                dndz=(zz_j, nz_j),
                bias=(zz_j, np.ones_like(zz_j)),
                has_rsd=True
            )
            cl = ccl.angular_cl(cosmo, count_i, count_j, ell)
            cls_ccl.append(cl)

    return cls_ccl
