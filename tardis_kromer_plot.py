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
            7, 'o': 8, 'f': 9, 'ne': 10, 'na': 11, 'mg': 12, 'al': 13, 'si':
            14, 'p': 15, 's': 16, 'cl': 17, 'ar': 18, 'k': 19,    'ca': 20,
            'sc': 21, 'ti': 22, 'v': 23, 'cr': 24, 'mn': 25, 'fe': 26, 'co':
            27, 'ni': 28, 'cu': 29, 'zn': 30, 'ga': 31, 'ge': 32, 'as': 33,
            'se': 34, 'br': 35, 'kr': 36, 'rb': 37, 'sr': 38, 'y': 39,  'zr':
            40, 'nb': 41, 'mo': 42, 'tc': 43, 'ru': 44, 'rh': 45, 'pd': 46,
            'ag': 47, 'cd': 48}
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
        logger.warning("Warning: Tardis has not been compiled with the virtual"
                       " packet logging feature, which is required to generate"
                       " Kromer plots. Recompile with the flag "
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
            logger.warning("Warning: hdf5 files do not contain full "
                           " information" " about virtual packets. Most"
                           " likely, Tardis was not" " compiled with the flag"
                           " '--with-vpacket-logging'.")
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
        self.time_of_simulation = \
            mhdf5["/configuration/"]["time_of_simulation"]


class tardis_kromer_plotter(object):
    """A plotter, generating spectral diagnostics plots as proposed by M.
    Kromer.

    With this tool, a specific visualisation of Tardis spectra may be produced.
    It illustrates which elements predominantly contribute to the emission and
    absorption part of the emergent (virtual) packet spectrum.

    Once a model is defined, a series of queries is performed on the packet
    property arrays. The results are cached and the "Kromer" plot is produced
    with the main method of this class, namely with ~generate_plot.

    Parameters
    ----------
    mdl : tardis.model.Radial1DModel or model_h5
        model (or model_h5) object of the Tardis run

    Notes
    -----
    For this to work, the model must be generated by a Tardis calculation using
    the virtual packet logging capability. This requires a compilation with the
    --with-vpacket-logging flag.

    This way of illustrating the spectral synthesis process was introduced by
    M. Kromer (see e.g. [1]_).

    References
    ----------

    .. [1] Kromer et al. "SN 2010lp - Type Ia Supernova from a Violent Merger
       of Two Carbon-Oxygen White Dwarfs" ApjL, 2013, 778, L18

    """
    def __init__(self, mdl):

        self._mdl = None
        self._zmax = 32
        self._cmap = cm.jet
        self._xlim = None
        self._ylim = None
        self._twinx = False

        self._bins = None
        self._ax = None
        self._pax = None

        self._noint_mask = None
        self._escat_mask = None
        self._escatonly_mask = None
        self._line_mask = None
        self._lam_escat = None
        self._lam_noint = None
        self._weights_escat = None
        self._weights_noint = None
        self._line_in_infos = None
        self._line_in_nu = None
        self._line_in_L = None
        self._line_out_infos = None
        self._line_out_nu = None
        self._line_out_L = None

        self.mdl = mdl

    @property
    def mdl(self):
        """Tardis model object"""
        return self._mdl

    @mdl.setter
    def mdl(self, val):
        try:
            assert(type(val) == tardis.model.Radial1DModel or type(val) ==
                   model_h5)
        except AssertionError:
            raise ValueError("'mdl' must be either a model_h5 or a"
                             " tardis.model.Radia1DModel instance")

        if val.runner.virt_logging == 0:
            raise ValueError("Virtual packet logging was deactivated in the"
                             " input model. Cannot generate Kromer plot")
        self._reset_cache()
        self._mdl = val

    @property
    def zmax(self):
        """Maximum atomic number"""
        return self._zmax

    @property
    def cmap(self):
        """Colour map, used to highlight the different atoms"""
        return self._cmap

    @property
    def ax(self):
        """Main axes, containing the emission part of the Kromer plot"""
        return self._ax

    @property
    def pax(self):
        """Secondary axes, containing the absorption part of the Kromer plot"""
        return self._pax

    @property
    def bins(self):
        """frequency binning for the spectral visualisation"""
        return self._bins

    @property
    def xlim(self):
        """wavelength limits"""
        return self._xlim

    @property
    def ylim(self):
        """Flux limits"""
        return self._ylim

    @property
    def twinx(self):
        """switch to decide where to place the absorption part of the Kromer
        plot"""
        return self._twinx

    @property
    def noint_mask(self):
        """Masking array, highlighting the packets that never interacted"""
        if self._noint_mask is None:
            self._noint_mask = \
                self.mdl.runner.virt_packet_last_interaction_type == -1
        return self._noint_mask

    @property
    def escat_mask(self):
        """Masking array, highlighting the packets that performed Thomson
        scatterings"""
        if self._escat_mask is None:
            self._escat_mask = \
                self.mdl.runner.virt_packet_last_interaction_type == 1
        return self._escat_mask

    @property
    def escatonly_mask(self):
        """Masking array, highlighting the packets that only performed Thomson
        scatterings"""
        if self._escatonly_mask is None:
            tmp = self.mdl.runner
            tmp = ((tmp.virt_packet_last_line_interaction_in_id == -1) *
                   (self.escat_mask)).astype(np.bool)
            self._escatonly_mask = tmp
        return self._escatonly_mask

    @property
    def line_mask(self):
        """Mask array, highlighting packets whose last interaction was with a
        line"""
        if self._line_mask is None:
            tmp = self.mdl.runner
            self._line_mask = \
                ((tmp.virt_packet_last_interaction_type > -1) *
                 (tmp.virt_packet_last_line_interaction_in_id > -1))
        return self._line_mask

    @property
    def lam_noint(self):
        """Wavelength of the non-interacting packets"""
        if self._lam_noint is None:
            self._lam_noint = \
                (csts.c.cgs /
                 (self.mdl.runner.virt_packet_nus[self.noint_mask] *
                  units.Hz)).to(units.AA)
        return self._lam_noint

    @property
    def lam_escat(self):
        """Wavelength of the purely electron scattering packets"""
        if self._lam_escat is None:
            self._lam_escat = \
                (csts.c.cgs /
                 (self.mdl.runner.virt_packet_nus[self.escatonly_mask] *
                  units.Hz)).to(units.AA)
        return self._lam_escat

    @property
    def weights_escat(self):
        """luminosity of the only electron scattering packets"""
        if self._weights_escat is None:
            self._weights_escat = \
                (self.mdl.runner.virt_packet_energies[self.escatonly_mask] *
                 units.erg / self.mdl.time_of_simulation)
        return self._weights_escat

    @property
    def weights_noint(self):
        """luminosity of the non-interacting packets"""
        if self._weights_noint is None:
            self._weights_noint = \
                (self.mdl.runner.virt_packet_energies[self.noint_mask] *
                 units.erg / self.mdl.time_of_simulation)
        return self._weights_noint

    @property
    def line_out_infos(self):
        """Line ids of the transitions packets were emitted last"""
        if self._line_out_infos is None:
            tmp = self.mdl.runner.virt_packet_last_line_interaction_out_id
            ids = tmp[self.line_mask]
            self._line_out_infos = self.mdl.atom_data.lines.iloc[ids]
        return self._line_out_infos

    @property
    def line_out_nu(self):
        """frequency of the transitions packets were emitted last"""
        if self._line_out_nu is None:
            self._line_out_nu = self.mdl.runner.virt_packet_nus[self.line_mask]
        return self._line_out_nu

    @property
    def line_out_L(self):
        """luminosity of the line interaction packets"""
        if self._line_out_L is None:
            tmp = self.mdl.runner.virt_packet_energies
            self._line_out_L = tmp[self.line_mask]
        return self._line_out_L

    @property
    def line_in_infos(self):
        """Line ids of the transitions packets were last absorbed"""
        if self._line_in_infos is None:
            tmp = self.mdl.runner.virt_packet_last_line_interaction_in_id
            ids = tmp[self.line_mask]
            self._line_in_infos = self.mdl.atom_data.lines.iloc[ids]
        return self._line_in_infos

    @property
    def line_in_nu(self):
        """frequencies of the transitions packets were last absorbed"""
        if self._line_in_nu is None:
            nus = self.mdl.runner.virt_packet_last_interaction_in_nu * units.Hz
            self._line_in_nu = nus[self.line_mask]
        return self._line_in_nu

    @property
    def line_in_L(self):
        """luminosity of the line interaction packets"""
        if self._line_in_L is None:
            tmp = self.mdl.runner.virt_packet_energies
            self._line_in_L = tmp[self.line_mask]
        return self._line_in_L

    def _reset_cache(self):
        """Reset cached variables - only needed in case the model is changed
        after initialisation"""

        self._noint_mask = None
        self._escat_mask = None
        self._escatonly_mask = None
        self._line_mask = None
        self._lam_escat = None
        self._lam_noint = None
        self._weights_escat = None
        self._weights_noint = None
        self._line_in_infos = None
        self._line_in_nu = None
        self._line_in_L = None
        self._line_out_infos = None
        self._line_out_nu = None
        self._line_out_L = None

    def generate_plot(self, ax=None, cmap=cm.jet, bins=None, xlim=None,
                      ylim=None, twinx=False):
        """Generate the actual "Kromer" plot

        Parameters
        ----------
        ax : matplotlib.axes or None
            axes object into which the emission part of the Kromer plot should
            be plotted; if None, a new one is generated (default None)
        cmap : matplotlib.cm.ListedColormap or None
            color map object used for the illustration of the different atomic
            contributions (default matplotlib.cm.jet)
        bins : np.ndarray or None
            array of the wavelength bins used for the illustration of the
            atomic contributions; if None, the same binning as for the stored
            virtual spectrum is used (default None)
        xlim : tuple or array-like or None
            wavelength limits for the display; if None, the x-axis is
            automatically scaled (default None)
        ylim : tuple or array-like or None
            flux limits for the display; if None, the y-axis is automatically
            scaled (default None)
        twinx : boolean
            determines where the absorption part of the Kromer plot is placed,
            if True, the absorption part is attached at the top of the main
            axes box, otherwise it is placed below the emission part (default
            False)

        Returns
        -------
        fig : matplotlib.figure
            figure instance containing the plot
        """
        self._ax = None
        self._pax = None

        self._cmap = cmap
        self._ax = ax
        self._xlim = xlim
        self._ylim = ylim
        self._twinx = twinx

        if bins is None:
            self._bins = self.mdl.spectrum_virtual.wavelength[::-1]
        else:
            self._bins = bins

        self._axes_handling_preparation()
        self._generate_emission_part()
        self._generate_and_add_colormap()
        self._generate_and_add_legend()
        self._paxes_handling_preparation()
        self._generate_absorption_part()
        self._axis_handling_label_rescale()

        return plt.gcf()

    def _axes_handling_preparation(self):
        """prepare the main axes; create a new axes if none exists"""

        if self._ax is None:
            self._ax = plt.figure().add_subplot(111)

    def _paxes_handling_preparation(self):
        """prepare the axes for the absorption part of the Kromer plot
        according to the twinx value"""

        if self.twinx:
            self._pax = self._ax.twinx()
        else:
            self._pax = self._ax

    def _generate_emission_part(self):
        """generate the emission part of the Kromer plot"""

        lams = [self.lam_noint, self.lam_escat]
        weights = [self.weights_noint, self.weights_escat]
        colors = ["black", "grey"]

        for zi in xrange(1, self.zmax+1):
            mask = self.line_out_infos.atomic_number.values == zi
            lams.append((csts.c.cgs / (self.line_out_nu[mask] *
                                       units.Hz)).to(units.AA))
            weights.append(self.line_out_L[mask] * units.erg /
                           self.mdl.time_of_simulation)
            colors.append(self.cmap(float(zi) / float(self.zmax)))

        Lnorm = 0
        for w in weights:
            Lnorm += np.sum(w)

        ret = self.ax.hist(lams, bins=self.bins, stacked=True,
                           histtype="stepfilled", normed=True, weights=weights)

        for i, col in enumerate(ret[-1]):
            for reti in col:
                reti.set_facecolor(colors[i])
                reti.set_edgecolor(colors[i])
                reti.set_linewidth(0)
                reti.xy[:, 1] *= Lnorm

        self.ax.plot(self.mdl.spectrum_virtual.wavelength,
                     self.mdl.spectrum_virtual.luminosity_density_lambda,
                     color="blue", drawstyle="steps-post", lw=0.5)

    def _generate_absorption_part(self):
        """generate the absorption part of the Kromer plot"""

        lams = []
        weights = []
        colors = []

        for zi in xrange(1, self.zmax+1):
            mask = self.line_in_infos.atomic_number.values == zi
            lams.append((csts.c.cgs / self.line_in_nu[mask]).to(units.AA))
            weights.append(self.line_in_L[mask] * units.erg /
                           self.mdl.time_of_simulation)
            colors.append(self.cmap(float(zi) / float(self.zmax)))

        Lnorm = 0
        for w in weights:
            Lnorm -= np.sum(w)

        ret = self.pax.hist(lams, bins=self.bins, stacked=True,
                            histtype="stepfilled", normed=True,
                            weights=weights)

        for i, col in enumerate(ret[-1]):
            for reti in col:
                reti.set_facecolor(colors[i])
                reti.set_edgecolor(colors[i])
                reti.set_linewidth(0)
                reti.xy[:, 1] *= Lnorm

    def _generate_and_add_colormap(self):
        """generate the custom color map, linking colours with atomic
        numbers"""

        values = [self.cmap(float(i) / float(self.zmax))
                  for i in xrange(1, self.zmax+1)]

        custcmap = matplotlib.colors.ListedColormap(values)
        bounds = np.arange(self.zmax+1) + 0.5
        norm = matplotlib.colors.Normalize(vmin=1, vmax=self.zmax+1)
        mappable = cm.ScalarMappable(norm=norm, cmap=custcmap)
        mappable.set_array(np.linspace(1, self.zmax + 1, 256))
        labels = [inv_elements[zi].capitalize()
                  for zi in xrange(1, self.zmax+1)]

        mainax = self.ax
        cbar = plt.colorbar(mappable, ax=mainax)
        cbar.set_ticks(bounds)
        cbar.set_ticklabels(labels)

    def _generate_and_add_legend(self):
        """add legend"""

        bpatch = patches.Patch(color="black", label="photosphere")
        gpatch = patches.Patch(color="grey", label="e-scattering")
        bline = lines.Line2D([], [], color="blue", label="virtual spectrum")
        self.ax.legend(handles=[bline, gpatch, bpatch])

    def _axis_handling_label_rescale(self):
        """add axis labels and perform axis scaling"""

        if self.ylim is None:
            self.ax.autoscale(axis="y")
        else:
            self.ax.set_ylim(self.ylim)

        self._ylim = self.ax.get_ylim()

        if self.xlim is None:
            self.ax.autoscale(axis="x")
        else:
            self.ax.set_xlim(self.xlim)

        self._xlim = self.ax.get_xlim()

        if self.twinx:
            self.pax.set_ylim([-self.ylim[-1], -self.ylim[0]])
            self.pax.set_yticklabels([])
        else:
            self.pax.set_ylim([-self.ylim[-1], self.ylim[-1]])
        self.pax.set_xlim(self.xlim)

        self.ax.set_xlabel(r"$\lambda$ [$\mathrm{\AA}$]")
        ylabel = r"$L_{\mathrm{\lambda}}$ [$\mathrm{erg\,s^{-1}\,\AA^{-1}}$]"
        self.ax.set_ylabel(ylabel)
