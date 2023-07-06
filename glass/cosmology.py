# author: Axel Guinot <axel.guinot.astro@gmail.com> ;
#         Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
Cosmology (:mod:`glass.cosmology`)
===================================

.. currentmodule:: glass.cosmology

The :mod:`glass.cosmology` module that handle the cosmology needed by GLASS.
This is done through pyccl. Only the requiered functions are implemented.

'''

import numpy as np
import pyccl as ccl


class Cosmology(ccl.Cosmology):

    def __init__(
        self,
        Omega_c=None,
        Omega_b=None,
        h=None,
        n_s=None,
        sigma8=None,
        A_s=None,
        Omega_k=0.,
        Omega_g=None,
        Neff=3.046,
        m_nu=0.,
        m_nu_type=None,
        w0=-1.,
        wa=0.,
        T_CMB=None,
        bcm_log10Mc=np.log10(1.2e14),
        bcm_etab=0.5,
        bcm_ks=55.,
        mu_0=0.,
        sigma_0=0.,
        c1_mg=1.,
        c2_mg=1.,
        lambda_mg=0.,
        z_mg=None,
        df_mg=None,
        transfer_function='boltzmann_camb',
        matter_power_spectrum='halofit',
        baryons_power_spectrum='nobaryons',
        mass_function='tinker10',
        halo_concentration='duffy2008',
        emulator_neutrinos='strict',
        extra_parameters=None,
    ):
        super().__init__(
            Omega_c,
            Omega_b,
            h,
            n_s,
            sigma8,
            A_s,
            Omega_k,
            Omega_g,
            Neff,
            m_nu,
            m_nu_type,
            w0,
            wa,
            T_CMB,
            bcm_log10Mc,
            bcm_etab,
            bcm_ks,
            mu_0,
            sigma_0,
            c1_mg,
            c2_mg,
            lambda_mg,
            z_mg,
            df_mg,
            transfer_function,
            matter_power_spectrum,
            baryons_power_spectrum,
            mass_function,
            halo_concentration,
            emulator_neutrinos,
            extra_parameters,
        )

    @property
    def dh(self):
        """Hubble distance

        dh = 2997.92458 Mpc/h by definition

        """

        return 2997.92458/self.cosmo.params.h

    @property
    def omega_m(self):
        """
        """

        return self.cosmo.params.Omega_m

    @property
    def h(self):
        """
        """

        return self.cosmo.params.h

    @property
    def h0(self):
        """
        """

        return self.cosmo.params.H0

    def xm(self, z, zp=None):
        """Dimensionless transverse cmoving distance

        :math:`d_M(z) = d_M(z)/d_H`

        Parameters
        ----------
        z : float, numpy.ndarray
            redshift
        zp : _type_, optional
            _description_, by default None

        Returns
        -------
        float, numpy.ndarray
            Dimensionless transverse cmoving distance
        """

        return self.dm(z, zp)/self.dh

    def dm(self, z, zp=None):
        """Transverse comoving distance

        :math:`d_M(z)` in Mpc

        Parameters
        ----------
        z : float, numpy.ndarray
            redshift
        zp : _type_, optional
            _description_, by default None

        Returns
        -------
        float, numpy.ndarray
            Transverse comoving distance
        """

        O_k = self.cosmo.params.Omega_k
        if O_k == 0:
            return self.dc(z, zp)
        if O_k > 0:
            k = O_k**0.5
            return np.sinh(self.dc(z, zp)/self.dh*k)/k*self.dh
        else:
            k = (-O_k)**0.5
            return np.sinh(self.dc(z, zp)/self.dh*k)/k*self.dh

    def dc(self, z, zp=None):
        """Comoving distance

        :math:`d_c(z)` in Mpc

        Parameters
        ----------
        z : float, numpy.ndarray
            redshift
        zp : _type_, optional
            _description_, by default None

        Returns
        -------
        float, numpy.ndarray
            Comoving distance
        """

        a = 1./(1. + z)

        if zp is not None:
            ap = 1./(1. + zp)
            return self.comoving_radial_distance(ap) \
                - self.comoving_radial_distance(a)
        return self.comoving_radial_distance(a)

    def dc_inv(self, dc):
        """Inverse function for the comoving distance

        In Mpc

        Parameters
        ----------
        dc : float, numpy.ndarray
            comoving distance

        Returns
        -------
        float, numpy.ndarray
            redshift
        """

        a = self.scale_factor_of_chi(dc)

        return 1./a - 1

    def ef(self, z):
        """Standardised Hubble function

        :math:`E(z) = H(z)/H_0`

        Parameters
        ----------
        z : float, numpy.ndarray
            redshift

        Returns
        -------
        float, numpy.ndarray
            Standardised Hubble function
        """

        return self.h_over_h0(1./(1. + z))

    def rho_m_z(self, z):
        """Redshift-dependent matter density

        In Msol Mpc-3

        Parameters
        ----------
        z : float, numpy.ndarray
            redshift

        Returns
        -------
        float, numpy.ndarray
            Matter density
        """

        return self.rho_x(1./(1. + z), 'matter')
