# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
#  File Name : tardis_opacity.py
#
#  Purpose :
#
#  Creation Date : 28-01-2016
#
#  Last Modified : Thu 28 Jan 2016 16:06:11 CET
#
#  Created By :
#
# _._._._._._._._._._._._._._._._._._._._._.
import numpy as np
import astropy.units as units
import astropy.constants as csts
import pandas as pd
from astropy.analytic_functions import blackbody_nu


def expansion_opacity(mdl, nbins=300, lam_min=100*units.AA,
                      lam_max=2e4*units.AA, bin_scaling="log"):

    try:
        lam_min = lam_min.to("AA")
    except AttributeError:
        lam_min *= units.AA
    try:
        lam_max = lam_max.to("AA")
    except AttributeError:
        lam_max *= units.AA

    nu_max = lam_min.to("Hz", equivalencies=units.spectral())
    nu_min = lam_max.to("Hz", equivalencies=units.spectral())

    if bin_scaling=="log":

        nu_bins = np.logspace(np.log10(nu_min.to("Hz").value),
                              np.log10(nu_max.to("Hz").value), nbins+1) * \
            units.Hz

    elif bin_scaling=="linear":

        nu_bins = np.linspace(nu_min, nu_max, nbins+1)

    line_waves = mdl.atom_data.lines.ix[mdl.plasma_array.tau_sobolevs.index]
    line_waves = line_waves.wavelength.values * units.AA

    nshells = mdl.tardis_config["structure"]["no_of_shells"]
    t_exp = mdl.tardis_config['supernova']['time_explosion']

    kappa = np.zeros((nbins, nshells)) / units.cm

    for i in xrange(nbins):

        lam_low = nu_bins[i+1].to("AA", equivalencies=units.spectral())
        lam_up = nu_bins[i].to("AA", equivalencies=units.spectral())

        mask = np.argwhere((line_waves > lam_low) * (line_waves <
                                                     lam_up)).ravel()
        tmp = (1 -
               np.exp(-mdl.plasma_array.tau_sobolevs.iloc[mask])).sum().values
        kappa[i,:] = tmp * nu_bins[i] / (nu_bins[i+1] - nu_bins[i]) / (csts.c *
                                                                       t_exp)

    return nu_bins, kappa

def thomson_scattering_opacity(mdl):

    kappa = ((mdl.plasma_array.electron_densities.values / units.cm**3) *
             csts.sigma_T)

    return kappa


def total_opacity(mdl, nbins=300, lam_min=100*units.AA, lam_max=2e4*units.AA,
                  bin_scaling="log"):

    nu_bins, kappa_exp = expansion_opacity(mdl, nbins=nbins, lam_min=lam_min,
                                           lam_max=lam_max,
                                           bin_scaling=bin_scaling)

    kappa_thom_tmp = thomson_scattering_opacity(mdl)
    kappa_thom = np.zeros((nbins, len(kappa_thom_tmp))) / units.cm

    for i in xrange(nbins):
        kappa_thom[i,:] = kappa_thom_tmp

    kappa = kappa_exp + kappa_thom

    return nu_bins, kappa, kappa_exp, kappa_thom

def planck_mean_opacity(mdl, nbins=300, lam_min=100*units.AA, lam_max=2e4*units.AA,
                        bin_scaling="log"):

    nu_bins, kappa, kappa_exp, kappa_thom = total_opacity(mdl, nbins=nbins,
                                                          lam_min=lam_min,
                                                          lam_max=lam_max,
                                                          bin_scaling=bin_scaling)

    nshells = mdl.tardis_config["structure"]["no_of_shells"]

    kappa_planck_mean = np.zeros(nshells) / units.cm

    for i in xrange(nshells):
        delta_nu = (nu_bins[1:] - nu_bins[:-1])
        T = mdl.plasma_array.t_rad[i]

        tmp = (blackbody_nu(nu_bins[:-1], T) * delta_nu * kappa[:,0]).sum()
        tmp /= (blackbody_nu(nu_bins[:-1], T) * delta_nu).sum()

        kappa_planck_mean[i] = tmp

    return kappa_planck_mean

def planck_optical_depth(mdl, nbins=300, lam_min=100*units.AA, lam_max=2e4*units.AA,
                         bin_scaling="log"):

    kappa_planck_mean = planck_mean_opacity(mdl, nbins=nbins, lam_min=lam_min,
                                            lam_max=lam_max,
                                            bin_scaling=bin_scaling)

    r_inner = mdl.tardis_config["structure"]["r_inner"]
    r_outer = mdl.tardis_config["structure"]["r_outer"]
    delta_r = r_outer - r_inner

    delta_tau = delta_r * kappa_planck_mean
    tau = np.zeros(len(r_inner))

    tau[-1] = delta_tau[-1]
    for i in xrange(len(tau)-2,-1,-1):
        tau[i] = tau[i+1] + delta_tau[i]

    return r_inner, tau, delta_tau



