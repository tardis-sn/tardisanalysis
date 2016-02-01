"""A simple plotting tool to create spectral diagnostics plots similar to those
originally proposed by M. Kromer (see, for example, Kromer et al. 2013, figure
4).
"""
from tardis import run_tardis
import yaml
import numpy as np
import pandas as pd
import astropy.units as units
import astropy.constants as csts
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.cm as cm
import argparse
import os

elements = { 'neut': 0, 'h': 1, 'he': 2, 'li': 3, 'be': 4, 'b': 5, 'c': 6, 'n': 7, 'o': 8, 'f': 9, 'ne': 10, 'na': 11, 'mg': 12, 'al': 13, 'si': 14, 'p': 15, 's': 16, 'cl': 17, 'ar': 18, 'k': 19,    'ca': 20, 'sc': 21, 'ti': 22, 'v': 23, 'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni': 28, 'cu': 29, 'zn': 30, 'ga': 31, 'ge': 32, 'as': 33, 'se': 34, 'br': 35, 'kr': 36, 'rb': 37, 'sr': 38, 'y': 39,  'zr': 40, 'nb': 41, 'mo': 42, 'tc': 43, 'ru': 44, 'rh': 45, 'pd': 46, 'ag': 47, 'cd': 48}
inv_elements = dict([(v,k) for k, v in elements.items()])

def store_data_for_kromer_plot(mdl, lines_fname = "lines.hdf5", model_fname = "model.hdf5"):

    mdl.to_hdf5(model_fname)
    mdl.atom_data.lines.to_hdf(lines_fname, "lines")


class atom_data_h(object):
    def __init__(self, hdf5):
        self.lines = hdf5["lines"]


class runner_h(object):
    def __init__(self, hdf5):
        self.virt_packet_last_interaction_type = hdf5["/runner/virt_packet_last_interaction_type"].values
        self.virt_packet_last_line_interaction_in_id = hdf5["/runner/virt_packet_last_line_interaction_in_id"].values
        self.virt_packet_last_line_interaction_out_id = hdf5["/runner/virt_packet_last_line_interaction_out_id"].values
        self.virt_packet_last_interaction_in_nu = hdf5["/runner/virt_packet_last_interaction_in_nu"].values
        self.virt_packet_nus = hdf5["/runner/virt_packet_nus"].values
        self.virt_packet_energies = hdf5["/runner/virt_packet_energies"].values

class spectrum_h(object):
    def __init__(self, hdf5):
        self.wavelength = units.Angstrom * hdf5["luminosity_density_virtual"]["wave"].values
        self.luminosity_density_lambda = units.erg / (units.Angstrom * units.s) * hdf5["luminosity_density_virtual"]["flux"].values

class model_h(object):
    def __init__(self, model_fname = "model.hdf5", lines_fname = "lines.hdf5"):

        lhdf5 = pd.HDFStore(lines_fname)
        mhdf5 = pd.HDFStore(model_fname)

        self.atom_data = atom_data_h(lhdf5)
        self.runner = runner_h(mhdf5)
        self.spectrum_virtual = spectrum_h(mhdf5)
        self.time_of_simulation = mhdf5["/configuration/"]["time_of_simulation"]



def make_kromer_plot_virt(mdl, bins = None, cmap = cm.jet, ax = None, xlim = None, ylim = None, twinx = True):

    if ax is None:
        ax = plt.figure().add_subplot(111)

    zmax = 32

    if bins is None:
        bins = mdl.spectrum_virtual.wavelength[::-1]

    noint_mask = mdl.runner.virt_packet_last_interaction_type == -1
    escat_mask = mdl.runner.virt_packet_last_interaction_type == 1
    escatonly_mask = ((mdl.runner.virt_packet_last_line_interaction_in_id == -1) * (escat_mask)).astype(np.bool)

    lam_noint = (csts.c.cgs / (mdl.runner.virt_packet_nus[noint_mask] * units.Hz)).to(units.AA)
    lam_escat = (csts.c.cgs / (mdl.runner.virt_packet_nus[escatonly_mask] * units.Hz)).to(units.AA)

    weights_noint = mdl.runner.virt_packet_energies[noint_mask] * units.erg / mdl.time_of_simulation
    weights_escat = mdl.runner.virt_packet_energies[escatonly_mask] * units.erg / mdl.time_of_simulation

    line_mask = (mdl.runner.virt_packet_last_interaction_type > -1) * (mdl.runner.virt_packet_last_line_interaction_in_id > -1)

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


    ret = ax.hist(lams, bins = bins, stacked = True, histtype = "stepfilled", normed = True, weights = weights)

    for i, col in enumerate(ret[-1]):
        for reti in col:
            reti.set_facecolor(colors[i])
            reti.set_edgecolor(colors[i])
            reti.set_linewidth(0)
            reti.xy[:,1] *= Lnorm

    ax.plot(mdl.spectrum_virtual.wavelength, mdl.spectrum_virtual.luminosity_density_lambda, color = "blue", drawstyle = "steps-post", lw = 0.5)
    if ylim is None:
        ax.autoscale(axis = "y")
    else:
        ax.set_ylim(ylim)

    ylim = ax.get_ylim()

    if xlim is None:
        ax.autoscale(axis = "x")
    else:
        ax.set_xlim(xlim)

    xlim = ax.get_xlim()

    custcmap = matplotlib.colors.ListedColormap([cmap(float(i) / float(zmax)) for i in xrange(1, zmax+1)])
    bounds = np.arange(zmax+1) + 0.5
    norm = matplotlib.colors.Normalize(vmin = 1, vmax = zmax+1)
    mappable = cm.ScalarMappable(norm = norm, cmap = custcmap)
    mappable.set_array(np.linspace(1, zmax + 1, 256))
    labels = [inv_elements[zi].capitalize() for zi in xrange(1, zmax+1)]

    mainax = ax
    cbar = plt.colorbar(mappable, ax = mainax)
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

    ret = pax.hist(lams, bins = bins, stacked = True, histtype = "stepfilled", normed = True, weights = weights)

    for i, col in enumerate(ret[-1]):
        for reti in col:
            reti.set_facecolor(colors[i])
            reti.set_edgecolor(colors[i])
            reti.set_linewidth(0)
            reti.xy[:,1] *= Lnorm

    if twinx:
        pax.set_ylim([-ylim[-1], -ylim[0]])
    else:
        pax.set_ylim([-ylim[-1], ylim[-1]])
    pax.set_xlim(xlim)

    if not twinx:
        ax.axhline(y = 0, color = "white", ls = "dashed", dash_capstyle = "round", lw = 2)

    ax.set_xlabel(r"$\lambda$ [\AA]")
    ax.set_ylabel(r"$L_{\mathrm{\lambda}}$ [$\mathrm{erg\,s^{-1}\,\AA^{-1}}$]")

    bpatch = patches.Patch(color = "black", label = "photosphere")
    gpatch = patches.Patch(color = "grey", label = "e-scattering")
    bline = lines.Line2D([],[], color = "blue", label = "virtual spectrum")
    ax.legend(handles = [bline, gpatch, bpatch])


def test(one_iter = False, yfile = "tardis_example.yml", root = ".", from_file = False):

    yfile = os.path.join(root, yfile)

    config = yaml.load(open(yfile))

    if one_iter:
        config["montecarlo"]["iterations"] = 1

    mdl = run_tardis(config)

    if from_file:

        store_data_for_kromer_plot(mdl, lines_fname = "lines.hdf5", model_fname = "model.hdf5")

        mdl = model_h(lines_fname = "lines.hdf5", model_fname = "model.hdf5")

    make_kromer_plot_virt(mdl, bins = None, cmap = cm.jet, ax = None, ylim = None, twinx = False, xlim = [1000, 6000])


if __name__ == "__main__":

    #run it in the tardis_example directory
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--one", action="store_true", help="run TARDIS with only one iteration (good for a quick demonstration)")
    parser.add_argument("-d", "--dir", action="store", type=str, help="give directory (absolute path) in which TARDIS should be run and where the config file is present", default = ".")
    parser.add_argument("-c", "--config", action="store", type=str, help="give name of TARDIS config file", default = "tardis_example.yml")
    parser.add_argument("-f", "--file", action="store_true", help="store data and produce Kromer plot from file")
    args = parser.parse_args()

    test(one_iter = args.one, yfile = args.config, root = args.dir, from_file = args.file)

    plt.show()

