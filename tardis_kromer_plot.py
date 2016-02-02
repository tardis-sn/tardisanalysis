"""A simple plotting tool to create spectral diagnostics plots similar to those
originally proposed by M. Kromer (see, for example, Kromer et al. 2013, figure
4).
"""
import tardis.model
import logging
import numpy as np
import pandas as pd
import astropy.units as units
import astropy.constants as csts
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.cm as cm

logger = logging.getLogger(__name__)

elements = {'neut': 0, 'h': 1, 'he': 2, 'li': 3, 'be': 4, 'b': 5, 'c': 6, 'n':
            7, 'o': 8, 'f': 9, 'ne': 10, 'na': 11, 'mg': 12, 'al': 13, 'si': 14,
            'p': 15, 's': 16, 'cl': 17, 'ar': 18, 'k': 19,    'ca': 20, 'sc':
            21, 'ti': 22, 'v': 23, 'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni':
            28, 'cu': 29, 'zn': 30, 'ga': 31, 'ge': 32, 'as': 33, 'se': 34,
            'br': 35, 'kr': 36, 'rb': 37, 'sr': 38, 'y': 39,  'zr': 40, 'nb':
            41, 'mo': 42, 'tc': 43, 'ru': 44, 'rh': 45, 'pd': 46, 'ag': 47,
            'cd': 48}
inv_elements = dict([(v, k) for k, v in elements.items()])


def store_data_for_kromer_plot(mdl, lines_fname="lines.hdf5",
                               model_fname="model.hdf5"):
    """Simple helper routine to dump all information required to generate a
    kromer plot to hdf5 files.

    Parameters
    ----------
    mdl : tardis.model.Radial1DModel
        source tardis model object
    lines_fname : str
        name of the hdf5 file for the line list (default 'lines.hdf5')
    model_fname : str
        name of the hdf5 file for the model data (default 'model.hdf5')
    """

    if mdl.runner.virt_logging != 1:
        logger.warning("Warning: Tardis has not been compiled with the virtual "
                       "packet logging feature, which is required to generate "
                       "Kromer plots. Recompile with the flag "
                       "'--with-vpacket-logging'")

    mdl.to_hdf5(model_fname)
    mdl.atom_data.lines.to_hdf(lines_fname, "lines")


class atom_data_h5(object):
    """Simple representation of the atom_data object of Tardis.

    Used for model_h5.

    Parameters
    ----------
    hdf5 : pandas.HDFStore
        hdf5 file object containing the line list
    """
    def __init__(self, hdf5):
        self.lines = hdf5["lines"]


class runner_h5(object):
    """Simple representation of the runner object of Tardis.

    Used in model_h5.

    Parameters
    ----------
    hdf5 : pandas.HDFStore
        hdf5 file object containing the model data
    """
    def __init__(self, hdf5):
        self.virt_packet_last_interaction_type = \
            hdf5["/runner/virt_packet_last_interaction_type"].values
        self.virt_packet_last_line_interaction_in_id = \
            hdf5["/runner/virt_packet_last_line_interaction_in_id"].values
        self.virt_packet_last_line_interaction_out_id = \
            hdf5["/runner/virt_packet_last_line_interaction_out_id"].values
        self.virt_packet_last_interaction_in_nu = \
            hdf5["/runner/virt_packet_last_interaction_in_nu"].values
        self.virt_packet_nus = hdf5["/runner/virt_packet_nus"].values
        self.virt_packet_energies = \
            hdf5["/runner/virt_packet_energies"].values
        if len(self.virt_packet_nus) == 0:
            logger.warning("Warning: hdf5 files do not contain full information"
                           " about virtual packets. Most likely, Tardis was not"
                           " compiled with the flag '--with-vpacket-logging'.")
            self.virt_logging = 0
        else:
            self.virt_logging = 1


class spectrum_h5(object):
    """Simple representation of the spectrum object of Tardis.

    Used in model_h5.

    Parameters
    ----------
    hdf5 : pandas.HDFStore
        hdf5 file object containing the model data
    """
    def __init__(self, hdf5):
        self.wavelength = (units.Angstrom *
                           hdf5["luminosity_density_virtual"]["wave"].values)
        self.luminosity_density_lambda = units.erg / \
            (units.Angstrom * units.s) * \
            hdf5["luminosity_density_virtual"]["flux"].values


class model_h5(object):
    """Simple representation of the model object of Tardis.

    This helper object holds all information stored in hdf5 files in a similar
    fashion to tardis.model.Radial1DModel. This way, the same plotting routine
    can be used to process the full interactive Tardis model, or the minimal
    information stored in the hdf5 files.
    """
    def __init__(self, model_fname="model.hdf5", lines_fname="lines.hdf5"):

        lhdf5 = pd.HDFStore(lines_fname)
        mhdf5 = pd.HDFStore(model_fname)

        self.atom_data = atom_data_h5(lhdf5)
        self.runner = runner_h5(mhdf5)
        self.spectrum_virtual = spectrum_h5(mhdf5)
        self.time_of_simulation = mhdf5["/configuration/"]["time_of_simulation"]


def make_kromer_plot_virt(mdl, bins=None, cmap=cm.jet, ax=None, xlim=None,
                          ylim=None, twinx=True):

    # Safety checks
    try:
        assert(type(mdl) == tardis.model.Radial1DModel or type(mdl) == model_h5)
    except AssertionError:
        raise ValueError("'mdl' must be either a model_h5 or a"
                         " tardis.model.Radia1DModel instance")

    if mdl.runner.virt_logging == 0:
        raise ValueError("Virtual packet logging was deactivated in the input "
                         "model. Cannot generate Kromer plot")

    zmax = 32

    if ax is None:
        ax = plt.figure().add_subplot(111)

    if bins is None:
        bins = mdl.spectrum_virtual.wavelength[::-1]

    noint_mask = mdl.runner.virt_packet_last_interaction_type == -1
    escat_mask = mdl.runner.virt_packet_last_interaction_type == 1
    escatonly_mask = \
        ((mdl.runner.virt_packet_last_line_interaction_in_id == -1) *
         (escat_mask)).astype(np.bool)

    lam_noint = (csts.c.cgs / (mdl.runner.virt_packet_nus[noint_mask] *
                               units.Hz)).to(units.AA)
    lam_escat = (csts.c.cgs / (mdl.runner.virt_packet_nus[escatonly_mask] *
                               units.Hz)).to(units.AA)

    weights_noint = (mdl.runner.virt_packet_energies[noint_mask] * units.erg /
                     mdl.time_of_simulation)
    weights_escat = (mdl.runner.virt_packet_energies[escatonly_mask] *
                     units.erg / mdl.time_of_simulation)

    line_mask = ((mdl.runner.virt_packet_last_interaction_type > -1) *
                 (mdl.runner.virt_packet_last_line_interaction_in_id > -1))

    lams = [lam_noint, lam_escat]
    weights = [weights_noint, weights_escat]
    colors = ["black", "grey"]

    ids = mdl.runner.virt_packet_last_line_interaction_out_id[line_mask]
    line_infos = mdl.atom_data.lines.iloc[ids]

    line_nu = mdl.runner.virt_packet_nus[line_mask]
    line_L = mdl.runner.virt_packet_energies[line_mask]

    for zi in xrange(1, zmax+1):
        mask = line_infos.atomic_number.values == zi
        lams.append((csts.c.cgs / (line_nu[mask] * units.Hz)).to(units.AA))
        weights.append(line_L[mask] * units.erg / mdl.time_of_simulation)
        colors.append(cmap(float(zi) / float(zmax)))

    Lnorm = 0
    for w in weights:
        Lnorm += np.sum(w)

    ret = ax.hist(lams, bins=bins, stacked=True, histtype="stepfilled",
                  normed=True, weights=weights)

    for i, col in enumerate(ret[-1]):
        for reti in col:
            reti.set_facecolor(colors[i])
            reti.set_edgecolor(colors[i])
            reti.set_linewidth(0)
            reti.xy[:, 1] *= Lnorm

    ax.plot(mdl.spectrum_virtual.wavelength,
            mdl.spectrum_virtual.luminosity_density_lambda, color="blue",
            drawstyle="steps-post", lw=0.5)

    if ylim is None:
        ax.autoscale(axis="y")
    else:
        ax.set_ylim(ylim)

    ylim = ax.get_ylim()

    if xlim is None:
        ax.autoscale(axis="x")
    else:
        ax.set_xlim(xlim)

    xlim = ax.get_xlim()

    custcmap = matplotlib.colors.ListedColormap([cmap(float(i) / float(zmax))
                                                 for i in xrange(1, zmax+1)])
    bounds = np.arange(zmax+1) + 0.5
    norm = matplotlib.colors.Normalize(vmin=1, vmax=zmax+1)
    mappable = cm.ScalarMappable(norm=norm, cmap=custcmap)
    mappable.set_array(np.linspace(1, zmax + 1, 256))
    labels = [inv_elements[zi].capitalize() for zi in xrange(1, zmax+1)]

    mainax = ax
    cbar = plt.colorbar(mappable, ax=mainax)
    cbar.set_ticks(bounds)
    cbar.set_ticklabels(labels)

    if twinx:
        pax = ax.twinx()
    else:
        pax = ax

    nus = mdl.runner.virt_packet_last_interaction_in_nu * units.Hz

    line_L = mdl.runner.virt_packet_energies[line_mask]
    line_nu = nus[line_mask]

    lams = []
    weights = []
    colors = []

    for zi in xrange(1, zmax+1):
        mask = line_infos.atomic_number.values == zi
        lams.append((csts.c.cgs / line_nu[mask]).to(units.AA))
        weights.append(line_L[mask] * units.erg / mdl.time_of_simulation)
        colors.append(cmap(float(zi) / float(zmax)))

    Lnorm = 0
    for w in weights:
        Lnorm -= np.sum(w)

    ret = pax.hist(lams, bins=bins, stacked=True, histtype="stepfilled",
                   normed=True, weights=weights)

    for i, col in enumerate(ret[-1]):
        for reti in col:
            reti.set_facecolor(colors[i])
            reti.set_edgecolor(colors[i])
            reti.set_linewidth(0)
            reti.xy[:, 1] *= Lnorm

    if twinx:
        pax.set_ylim([-ylim[-1], -ylim[0]])
    else:
        pax.set_ylim([-ylim[-1], ylim[-1]])
    pax.set_xlim(xlim)

    if not twinx:
        ax.axhline(y=0, color="white", ls="dashed", dash_capstyle="round", lw=2)

    ax.set_xlabel(r"$\lambda$ [\AA]")
    ax.set_ylabel(r"$L_{\mathrm{\lambda}}$ [$\mathrm{erg\,s^{-1}\,\AA^{-1}}$]")

    bpatch = patches.Patch(color="black", label="photosphere")
    gpatch = patches.Patch(color="grey", label="e-scattering")
    bline = lines.Line2D([], [], color="blue", label="virtual spectrum")
    ax.legend(handles=[bline, gpatch, bpatch])
