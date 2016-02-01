# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
#  File Name : tardis_opacity.py
#
#  Purpose :
#
#  Creation Date : 28-01-2016
#
#  Last Modified : Mon 01 Feb 2016 12:22:21 CET
#
#  Created By :
#
# _._._._._._._._._._._._._._._._._._._._._.
import numpy as np
import astropy.units as units
import astropy.constants as csts
import pandas as pd
import logging
from astropy.analytic_functions import blackbody_nu

logger = logging.getLogger(__name__)

class opacity_calculator(object):
    def __init__(self, mdl, nbins=300, lam_min=100*units.AA,
                 lam_max=2e4*units.AA, bin_scaling="log"):

        self._mdl = None
        self._nbins = None
        self._lam_min = None
        self._lam_max = None
        self._bin_scaling = None

        self._r_inner = None
        self._r_outer = None
        self._t_exp = None
        self._nshells = None

        self._kappa_exp = None
        self._kappa_thom = None
        self._kappa_thom_grid = None
        self._kappa_tot = None
        self._planck_kappa = None
        self._planck_delta_tau = None
        self._planck_tau = None

        self.mdl = mdl
        self.nbins = nbins
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.bin_scaling = bin_scaling

    def _reset_opacities(self):
        self._kappa_exp = None
        self._kappa_thom = None
        self._kappa_thom_grid = None
        self._kappa_tot = None
        self._planck_kappa = None
        self._planck_delta_tau = None
        self._planck_tau = None

    def _reset_bins(self):
        self._nu_bins = None
        self._reset_opacities()

    def _reset_model(self):

        self._t_exp = None
        self._nshells = None
        self._r_inner = None
        self._r_outer = None

        self._reset_opacities()

    @property
    def bin_scaling(self):
        return self._bin_scaling

    @bin_scaling.setter
    def bin_scaling(self, val):
        allowed_values = ["log", "linear"]
        if not val in allowed_values:
            raise ValueError("wrong bin_scaling; must be among {:s}".format(", ".join(allowed_values)))
        self._reset_bins()
        self._bin_scaling = val

    @property
    def lam_min(self):
        return self._lam_min

    @lam_min.setter
    def lam_min(self, val):
        self._reset_bins()
        try:
            val.to("AA")
        except AttributeError:
            logger.warning("lam_min provided without units; assuming AA")
            val *= units.AA
        self._lam_min = val

    @property
    def lam_max(self):
        return self._lam_max

    @lam_max.setter
    def lam_max(self, val):
        self._reset_bins()
        try:
            val.to("AA")
        except AttributeError:
            logger.warning("lam_max provided without units; assuming AA")
            val *= units.AA
        self._lam_max = val

    @property
    def mdl(self):
        return self._mdl

    @mdl.setter
    def mdl(self, val):
        self._reset_model()
        self._mdl = val

    @property
    def nshells(self):
        if self._nshells is None:
            self._nshells = self.mdl.tardis_config["structure"]["no_of_shells"]
        return self._nshells

    @property
    def t_exp(self):
        if self._t_exp is None:
            self._t_exp = self.mdl.tardis_config['supernova']['time_explosion']
        return self._t_exp

    @property
    def r_inner(self):
        if self._r_inner is None:
            self._r_inner = self.mdl.tardis_config['structure']['r_inner']
        return self._r_inner

    @property
    def r_outer(self):
        if self._r_outer is None:
            self._r_outer = self.mdl.tardis_config['structure']['r_outer']
        return self._r_outer

    @property
    def nu_bins(self):
        if self._nu_bins is None:
            nu_max = self.lam_min.to("Hz", equivalencies=units.spectral())
            nu_min = self.lam_max.to("Hz", equivalencies=units.spectral())
            if self.bin_scaling == "log":
                nu_bins = np.logspace(np.log10(nu_min.value), np.log10(nu_max.value), self.nbins+1) * units.Hz
            elif self.bin_scaling == "linear":
                nu_bins = np.linspace(nu_min, nu_max, self.nbins+1)
            self._nu_bins = nu_bins
        return self._nu_bins

    @property
    def kappa_exp(self):
        if self._kappa_exp is None:
            self._kappa_exp = self._calc_expansion_opacity()
        return self._kappa_exp

    @property
    def kappa_thom(self):
        if self._kappa_thom is None:
            self._kappa_thom = self._calc_thomson_scattering_opacity()
        return self._kappa_thom

    @property
    def kappa_thom_grid(self):
        if self._kappa_thom_grid is None:
            kappa_thom_grid = np.zeros((self.nbins, self.nshells)) / units.cm
            for i in xrange(self.nbins):
                kappa_thom_grid[i, :] = self.kappa_thom
            self._kappa_thom_grid = kappa_thom_grid
        return self._kappa_thom_grid

    @property
    def kappa_tot(self):
        if self._kappa_tot is None:
            kappa_tot = self.kappa_exp + self.kappa_thom_grid
            self._kappa_tot = kappa_tot
        return self._kappa_tot

    @property
    def planck_kappa(self):
        if self._planck_kappa is None:
            planck_kappa = self._calc_planck_mean_opacity()
            self._planck_kappa = planck_kappa
        return self._planck_kappa

    @property
    def planck_delta_tau(self):
        if self._planck_delta_tau is None:
            planck_delta_tau = self._calc_planck_optical_depth()
            self._planck_delta_tau = planck_delta_tau
        return self._planck_delta_tau

    @property
    def planck_tau(self):
        if self._planck_tau is None:
            planck_tau = self._calc_integrated_planck_optical_depth()
            self._planck_tau = planck_tau
        return self._planck_tau

    def _calc_expansion_opacity(self):

        line_waves = self.mdl.atom_data.lines.ix[self.mdl.plasma_array.tau_sobolevs.index]
        line_waves = line_waves.wavelength.values * units.AA

        kappa_exp = np.zeros((self.nbins, self.nshells)) / units.cm

        for i in xrange(self.nbins):

            lam_low = self.nu_bins[i+1].to("AA", equivalencies=units.spectral())
            lam_up = self.nu_bins[i].to("AA", equivalencies=units.spectral())

            mask = np.argwhere((line_waves > lam_low) * (line_waves <
                                                         lam_up)).ravel()
            tmp = (1 - np.exp(-self.mdl.plasma_array.tau_sobolevs.iloc[mask])).sum().values
            kappa_exp[i,:] = tmp * self.nu_bins[i] / (self.nu_bins[i+1] - self.nu_bins[i]) / (csts.c * self.t_exp)

        return kappa_exp.to("1/cm")

    def _calc_thomson_scattering_opacity(self):

        try:
            sigma_T = csts.sigma_T
        except AttributeError:
            logger.warning("using astropy < 1.1.1: setting sigma_T manually")
            sigma_T = 6.65245873e-29 * units.m**2

        edens = self.mdl.plasma_array.electron_densities.values

        try:
            edens.to("1/cm^3")
        except AttributeError:
            logger.info("setting electron density units by hand (cm^-3)")
            edens = edens / units.cm**3

        kappa_thom =  edens * sigma_T

        return kappa_thom.to("1/cm")

    def _calc_planck_mean_opacity(self):

        kappa_planck_mean = np.zeros(self.nshells) / units.cm

        for i in xrange(self.nshells):
            delta_nu = (self.nu_bins[1:] - self.nu_bins[:-1])
            T = self.mdl.plasma_array.t_rad[i]

            tmp = (blackbody_nu(self.nu_bins[:-1], T) * delta_nu * self.kappa_tot[:,0]).sum()
            tmp /= (blackbody_nu(self.nu_bins[:-1], T) * delta_nu).sum()

            kappa_planck_mean[i] = tmp

        return kappa_planck_mean.to("1/cm")

    def _calc_planck_optical_depth(self):

        delta_r = self.r_outer - self.r_inner
        delta_tau = delta_r * self.planck_kappa

        return delta_tau.to("")

    def _calc_integrated_planck_optical_depth(self):

        tau = np.zeros(self.nshells)

        tau[-1] = self.planck_delta_tau[-1]
        for i in xrange(self.nshells-2, -1, -1):
            tau[i] = tau[i+1] + self.planck_delta_tau[i]

        return tau
