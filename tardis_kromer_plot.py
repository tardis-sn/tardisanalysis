"""A simple plotting tool to create spectral diagnostics plots similar to those
originally proposed by M. Kromer (see, for example, Kromer et al. 2013, figure
4).
"""
import logging
import os
import numpy as np
import pandas as pd
import astropy.units as units
import astropy.constants as csts
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.cm as cm
from tardis_minimal_model import minimal_model

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


def store_data_for_kromer_plot(mdl, buffer_or_fname="minimal_model.hdf5",
                               path="", mode="virtual"):
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

    def _save_spectrum_real(key, path, hdf_store):

        wave = mdl.spectrum.wavelength.value
        flux = mdl.spectrum.luminosity_density_lambda.value

        luminosity_density = \
            pd.DataFrame.from_dict(dict(wave=wave, flux=flux))
        luminosity_density.to_hdf(hdf_store, os.path.join(path, key))

    def _save_spectrum_virtual(key, path, hdf_store):

        wave = mdl.spectrum_virtual.wavelength.value
        flux = mdl.spectrum_virtual.luminosity_density_lambda.value

        luminosity_density_virtual = pd.DataFrame.from_dict(dict(wave=wave,
                                                                 flux=flux))
        luminosity_density_virtual.to_hdf(hdf_store, os.path.join(path, key))

    def _save_configuration_dict(key, path, hdf_store):
        configuration_dict = dict(time_of_simulation=mdl.time_of_simulation)
        configuration_dict_path = os.path.join(path, 'configuration')
        pd.Series(configuration_dict).to_hdf(hdf_store,
                                             configuration_dict_path)

    if mode == "virtual":
        include_from_runner_ = \
            {'virt_packet_last_interaction_type': None,
             'virt_packet_last_line_interaction_in_id': None,
             'virt_packet_last_line_interaction_out_id': None,
             'virt_packet_last_interaction_in_nu': None,
             'virt_packet_nus': None,
             'virt_packet_energies': None}
        include_from_spectrum_ = \
            {'luminosity_density_virtual': _save_spectrum_virtual}
    elif mode == "real":
        include_from_runner_ = \
            {'last_interaction_type': None,
             'last_line_interaction_in_id': None,
             'last_line_interaction_out_id': None,
             'last_interaction_in_nu': None,
             'output_nu': None,
             'output_energy': None}
        include_from_spectrum_ = \
            {'luminosity_density': _save_spectrum_real}

    else:
        raise ValueError

    include_from_atom_data_ = {'lines': None}
    include_from_model_in_hdf5 = {'runner': include_from_runner_,
                                  'atom_data': include_from_atom_data_,
                                  'spectrum': include_from_spectrum_,
                                  'configuration_dict':
                                  _save_configuration_dict,
                                  }

    if isinstance(buffer_or_fname, basestring):
        hdf_store = pd.HDFStore(buffer_or_fname)
    elif isinstance(buffer_or_fname, pd.HDFStore):
        hdf_store = buffer_or_fname
    else:
        raise IOError('Please specify either a filename or an HDFStore')
    logger.info('Writing to path %s', path)

    def _get_hdf5_path(path, property_name):
        return os.path.join(path, property_name)

    def _to_smallest_pandas(object):
        try:
            return pd.Series(object)
        except Exception:
            return pd.DataFrame(object)

    def _save_model_property(object, property_name, path, hdf_store):
        property_path = _get_hdf5_path(path, property_name)

        try:
            object.to_hdf(hdf_store, property_path)
        except AttributeError:
            _to_smallest_pandas(object).to_hdf(hdf_store, property_path)

    for key in include_from_model_in_hdf5:
        if include_from_model_in_hdf5[key] is None:
            _save_model_property(getattr(mdl, key), key, path, hdf_store)
        elif callable(include_from_model_in_hdf5[key]):
            include_from_model_in_hdf5[key](key, path, hdf_store)
        else:
            try:
                for subkey in include_from_model_in_hdf5[key]:
                    if include_from_model_in_hdf5[key][subkey] is None:
                        _save_model_property(getattr(getattr(mdl, key),
                                                     subkey), subkey,
                                             os.path.join(path, key),
                                             hdf_store)
                    elif callable(include_from_model_in_hdf5[key][subkey]):
                        include_from_model_in_hdf5[key][subkey](subkey, os.path.join(path, key), hdf_store)
                    else:
                        logger.critical('Can not save %s',
                                        str(os.path.join(path, key, subkey)))
            except:
                logger.critical('An error occurred while dumping %s to HDF.',
                                str(os.path.join(path, key)))

    hdf_store.flush()
    hdf_store.close()


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
    def __init__(self, mdl, mode="real"):

        self._mode = None
        self.mode = mode

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
    def mode(self):
        """packet mode - use real or virtual packets for plotting"""
        return self._mode

    @mode.setter
    def mode(self, val):
        known_modes = ["real", "virtual"]
        try:
            assert(val in known_modes)
        except AssertionError:
            raise ValueError("unknown mode")
        self._mode = val

    @property
    def mdl(self):
        """Tardis model object"""
        return self._mdl

    @mdl.setter
    def mdl(self, val):
        try:
            assert(type(val) == minimal_model)
        except AssertionError:
            raise ValueError("'mdl' must be either a minimal_model")

        if val.mode != self.mode:
            raise ValueError("packet mode of minimal_model doesn't"
                             " match requested mode")
        if not val.readin:
            raise ValueError("passing empty minimal_model; read in data first")

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
                self.mdl.last_interaction_type == -1
        return self._noint_mask

    @property
    def escat_mask(self):
        """Masking array, highlighting the packets that performed Thomson
        scatterings"""
        if self._escat_mask is None:
            self._escat_mask = \
                self.mdl.last_interaction_type == 1
        return self._escat_mask

    @property
    def escatonly_mask(self):
        """Masking array, highlighting the packets that only performed Thomson
        scatterings"""
        if self._escatonly_mask is None:
            tmp = ((self.mdl.last_line_interaction_in_id == -1) *
                   (self.escat_mask)).astype(np.bool)
            self._escatonly_mask = tmp
        return self._escatonly_mask

    @property
    def line_mask(self):
        """Mask array, highlighting packets whose last interaction was with a
        line"""
        if self._line_mask is None:
            self._line_mask = \
                ((self.mdl.last_interaction_type > -1) *
                 (self.mdl.last_line_interaction_in_id > -1))
        return self._line_mask

    @property
    def lam_noint(self):
        """Wavelength of the non-interacting packets"""
        if self._lam_noint is None:
            self._lam_noint = \
                (csts.c.cgs /
                 (self.mdl.packet_nus[self.noint_mask])).to(units.AA)
        return self._lam_noint

    @property
    def lam_escat(self):
        """Wavelength of the purely electron scattering packets"""
        if self._lam_escat is None:
            self._lam_escat = \
                (csts.c.cgs /
                 (self.mdl.packet_nus[self.escatonly_mask])).to(units.AA)
        return self._lam_escat

    @property
    def weights_escat(self):
        """luminosity of the only electron scattering packets"""
        if self._weights_escat is None:
            self._weights_escat = \
                (self.mdl.packet_energies[self.escatonly_mask] /
                 self.mdl.time_of_simulation)
        return self._weights_escat

    @property
    def weights_noint(self):
        """luminosity of the non-interacting packets"""
        if self._weights_noint is None:
            self._weights_noint = \
                (self.mdl.packet_energies[self.noint_mask] /
                 self.mdl.time_of_simulation)
        return self._weights_noint

    @property
    def line_out_infos(self):
        """Line ids of the transitions packets were emitted last"""
        if self._line_out_infos is None:
            tmp = self.mdl.last_line_interaction_out_id
            ids = tmp[self.line_mask]
            self._line_out_infos = self.mdl.lines.iloc[ids]
        return self._line_out_infos

    @property
    def line_out_nu(self):
        """frequency of the transitions packets were emitted last"""
        if self._line_out_nu is None:
            self._line_out_nu = self.mdl.packet_nus[self.line_mask]
        return self._line_out_nu

    @property
    def line_out_L(self):
        """luminosity of the line interaction packets"""
        if self._line_out_L is None:
            tmp = self.mdl.packet_energies
            self._line_out_L = tmp[self.line_mask]
        return self._line_out_L

    @property
    def line_in_infos(self):
        """Line ids of the transitions packets were last absorbed"""
        if self._line_in_infos is None:
            tmp = self.mdl.last_line_interaction_in_id
            ids = tmp[self.line_mask]
            self._line_in_infos = self.mdl.lines.iloc[ids]
        return self._line_in_infos

    @property
    def line_in_nu(self):
        """frequencies of the transitions packets were last absorbed"""
        if self._line_in_nu is None:
            nus = self.mdl.last_interaction_in_nu
            self._line_in_nu = nus[self.line_mask]
        return self._line_in_nu

    @property
    def line_in_L(self):
        """luminosity of the line interaction packets"""
        if self._line_in_L is None:
            tmp = self.mdl.packet_energies
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
            self._bins = self.mdl.spectrum_wave[::-1]
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
            lams.append((csts.c.cgs / (self.line_out_nu[mask])).to(units.AA))
            weights.append(self.line_out_L[mask] /
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

        self.ax.plot(self.mdl.spectrum_wave,
                     self.mdl.spectrum_luminosity,
                     color="blue", drawstyle="steps-post", lw=0.5)

    def _generate_absorption_part(self):
        """generate the absorption part of the Kromer plot"""

        lams = []
        weights = []
        colors = []

        for zi in xrange(1, self.zmax+1):
            mask = self.line_in_infos.atomic_number.values == zi
            lams.append((csts.c.cgs / self.line_in_nu[mask]).to(units.AA))
            weights.append(self.line_in_L[mask] /
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
