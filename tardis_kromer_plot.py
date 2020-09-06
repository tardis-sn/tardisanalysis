"""A simple plotting tool to create spectral diagnostics plots similar to those
originally proposed by M. Kromer (see, for example, Kromer et al. 2013, figure
4).
"""
import logging
import numpy as np
import astropy.units as units
import astropy.constants as csts
import pandas as pd
import csv

try:
    import astropy.modeling.blackbody as abb
except ImportError:  # for astropy version < 2.0
    import astropy.analytic_functions as abb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.cm as cm
from tardis_minimal_model import minimal_model

plt.rcdefaults()

logger = logging.getLogger(__name__)

with open("elements.csv") as f:
    reader = csv.reader(f, skipinitialspace=True)
    elements = dict(reader)

inv_elements = dict([(int(v), k) for k, v in elements.items()])


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
    mdl : minimal_model
        a minimal_model object containing the Tardis run
    mode : str, optional
        'real' (default) or 'virtual'; determines which packet population is
        used to generate the Kromer plot.

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
        self._zmax = 100
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
            assert val in known_modes
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
            assert type(val) == minimal_model
        except AssertionError:
            raise ValueError("'mdl' must be either a minimal_model")

        if val.mode != self.mode:
            raise ValueError(
                "packet mode of minimal_model doesn't" " match requested mode"
            )
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
            self._noint_mask = self.mdl.last_interaction_type == -1
        return self._noint_mask

    @property
    def escat_mask(self):
        """Masking array, highlighting the packets that performed Thomson
        scatterings"""
        if self._escat_mask is None:
            self._escat_mask = self.mdl.last_interaction_type == 1
        return self._escat_mask

    @property
    def escatonly_mask(self):
        """Masking array, highlighting the packets that only performed Thomson
        scatterings"""
        if self._escatonly_mask is None:
            tmp = (
                (self.mdl.last_line_interaction_in_id == -1) * (self.escat_mask)
            ).astype(np.bool)
            self._escatonly_mask = tmp
        return self._escatonly_mask

    @property
    def line_mask(self):
        """Mask array, highlighting packets whose last interaction was with a
        line"""
        if self._line_mask is None:
            self._line_mask = (self.mdl.last_interaction_type > -1) * (
                self.mdl.last_line_interaction_in_id > -1
            )
        return self._line_mask

    @property
    def lam_noint(self):
        """Wavelength of the non-interacting packets"""
        if self._lam_noint is None:
            self._lam_noint = (
                csts.c.cgs / (self.mdl.packet_nus[self.noint_mask])
            ).to(units.AA)
        return self._lam_noint

    @property
    def lam_escat(self):
        """Wavelength of the purely electron scattering packets"""
        if self._lam_escat is None:
            self._lam_escat = (
                csts.c.cgs / (self.mdl.packet_nus[self.escatonly_mask])
            ).to(units.AA)
        return self._lam_escat

    @property
    def weights_escat(self):
        """luminosity of the only electron scattering packets"""
        if self._weights_escat is None:
            self._weights_escat = (
                self.mdl.packet_energies[self.escatonly_mask]
                / self.mdl.time_of_simulation
            )
        return self._weights_escat

    @property
    def weights_noint(self):
        """luminosity of the non-interacting packets"""
        if self._weights_noint is None:
            self._weights_noint = (
                self.mdl.packet_energies[self.noint_mask]
                / self.mdl.time_of_simulation
            )
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

    @property
    def line_info(self):
        """produces list of elements to be included in the kromer plot"""
        self.line_out_infos_within_xlims = self.line_out_infos.loc[
            (self.line_out_infos.wavelength >= self._xlim[0])
            & (self.line_out_infos.wavelength <= self._xlim[1])
        ]
        
        self.line_out_infos_within_xlims['ion_id'] = self.line_out_infos_within_xlims['atomic_number'] * 1000 + self.line_out_infos_within_xlims['ion_number']
                
        if self._species_list != None:
            ids = [species_string_to_tuple(species)[0] * 1000 + species_string_to_tuple(species)[1] for species in self._species_list]

        
        self._elements_in_kromer_plot = np.c_[
            np.unique(
                line_out_infos_within_xlims.ion_id.values,
                return_counts=True,
            )
        ]
        
        if len(self._elements_in_kromer_plot) > self._nelements:
            if self._species_list == None:
                self._elements_in_kromer_plot = self._elements_in_kromer_plot[
                    np.argsort(self._elements_in_kromer_plot[:, 1])[::-1]
                ]
                self._elements_in_kromer_plot = self._elements_in_kromer_plot[
                    : self._nelements
                ]
                self._elements_in_kromer_plot = self._elements_in_kromer_plot[
                    np.argsort(self._elements_in_kromer_plot[:, 0])
                ]
            else:
                ids = [species_string_to_tuple(species)[0] * 1000 + species_string_to_tuple(species)[1] for species in self._species_list]
                mask = np.in1d(self._elements_in_kromer_plot[:, 0], ids)
                self._elements_in_kromer_plot = self._elements_in_kromer_plot[mask]
        else:
            self._nelements = len(self._elements_in_kromer_plot)
        return self._elements_in_kromer_plot

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

    def generate_plot(
        self,
        ax=None,
        cmap=cm.jet,
        bins=None,
        xlim=None,
        ylim=None,
        nelements=None,
        twinx=False,
        species_list=None,
    ):
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
        species_list: list of strings or None
            list of strings containing the names of species that should be included in the Kromer plots,
            e.g. ['Si II', 'Ca II']

        Returns
        -------
        fig : matplotlib.figure
            figure instance containing the plot
        """
        self._ax = None
        self._pax = None

        self._cmap = cmap
        self._ax = ax
        self._ylim = ylim
        self._twinx = twinx

        if nelements == None and species_list == None:
            self._nelements = len(
                np.unique(self.line_out_infos.atomic_number.values)
            )
        elif nelements == None and species_list != None:
            self._nelements = len(species_list)
        else:
            self._nelements = nelements
        
        self._species_list = species_list

        if xlim == None:
            self._xlim = [
                np.min(self.mdl.spectrum_wave).value,
                np.max(self.mdl.spectrum_wave).value,
            ]
        else:
            self._xlim = xlim

        if bins is None:
            self._bins = self.mdl.spectrum_wave[::-1]
        else:
            self._bins = bins

        self._axes_handling_preparation()
        self._generate_emission_part()
        self._generate_photosphere_part()
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

        self.elements_in_kromer_plot = self.line_info

        for zi in np.unique(self.line_out_infos_within_xlims.ion_id.values, return_counts=False,):
            
            ion_number = zi % 1000
            atomic_number = (zi - ion_number) / 1000
            
            if zi not in self.elements_in_kromer_plot[:, 0]:
                
                mask = ((self.line_out_infos.atomic_number.values == atomic_number) & (self.line_out_infos.ion_number.values == ion_number))
                lams.append((csts.c.cgs / (self.line_out_nu[mask])).to(units.AA))
                weights.append(self.line_out_L[mask] / self.mdl.time_of_simulation)
                colors.append("silver")
        ii = 0
        for zi in np.unique(self.line_out_infos_within_xlims.ion_id.values, return_counts=False,):
            
            ion_number = zi % 1000
            atomic_number = (zi - ion_number) / 1000
            
            if zi in self.elements_in_kromer_plot[:, 0]:
                
                mask = ((self.line_out_infos.atomic_number.values == atomic_number) & (self.line_out_infos.ion_number.values == ion_number))
                lams.append((csts.c.cgs / (self.line_out_nu[mask])).to(units.AA))
                weights.append(self.line_out_L[mask] / self.mdl.time_of_simulation)
                colors.append(self.cmap(float(ii) / float(self._nelements)))
                ii = ii + 1

        Lnorm = 0
        for w, lam in zip(weights, lams):
            Lnorm += np.sum(w[(lam >= self.bins[0]) * (lam <= self.bins[-1])])

        lams = [tmp_lam.value for tmp_lam in lams]
        weights = [tmp_wt.value for tmp_wt in weights]
        ret = self.ax.hist(
            lams,
            bins=self.bins.value,
            stacked=True,
            histtype="stepfilled",
            density=True,
            weights=weights,
        )

        for i, col in enumerate(ret[-1]):
            for reti in col:
                reti.set_facecolor(colors[i])
                reti.set_edgecolor(colors[i])
                reti.set_linewidth(0)
                reti.xy[:, 1] *= Lnorm.to("erg / s").value

        self.ax.plot(
            self.mdl.spectrum_wave,
            self.mdl.spectrum_luminosity,
            color="blue",
            drawstyle="steps-post",
            lw=0.5,
        )

    def _generate_photosphere_part(self):
        """generate the photospheric input spectrum part of the Kromer plot"""

        Lph = (
            abb.blackbody_lambda(self.mdl.spectrum_wave, self.mdl.t_inner)
            * 4
            * np.pi ** 2
            * self.mdl.R_phot ** 2
            * units.sr
        ).to("erg / (AA s)")

        self.ax.plot(self.mdl.spectrum_wave, Lph, color="red", ls="dashed")

    def _generate_absorption_part(self):
        """generate the absorption part of the Kromer plot"""

        lams = []
        weights = []
        colors = []

        self.elements_in_kromer_plot = self.line_info

        for zi in np.unique(self.line_out_infos_within_xlims.ion_id.values, return_counts=False,):
            
            ion_number = zi % 1000
            atomic_number = (zi - ion_number) / 1000
            
            if zi not in self.elements_in_kromer_plot[:, 0]:
                
                mask = ((self.line_out_infos.atomic_number.values == atomic_number) & (self.line_out_infos.ion_number.values == ion_number))
                lams.append((csts.c.cgs / (self.line_in_nu[mask])).to(units.AA))
                weights.append(self.line_in_L[mask] / self.mdl.time_of_simulation)
                colors.append("silver")
        ii = 0
        for zi in np.unique(self.line_out_infos_within_xlims.ion_id.values, return_counts=False,):
            
            ion_number = zi % 1000
            atomic_number = (zi - ion_number) / 1000
            
            if zi in self.elements_in_kromer_plot[:, 0]:
                
                mask = ((self.line_out_infos.atomic_number.values == atomic_number) & (self.line_out_infos.ion_number.values == ion_number))
                lams.append((csts.c.cgs / (self.line_in_nu[mask])).to(units.AA))
                weights.append(self.line_in_L[mask] / self.mdl.time_of_simulation)
                colors.append(self.cmap(float(ii) / float(self._nelements)))
                ii = ii + 1


        Lnorm = 0
        for w, lam in zip(weights, lams):
            Lnorm -= np.sum(w[(lam >= self.bins[0]) * (lam <= self.bins[-1])])

        lams = [tmp_l.value for tmp_l in lams]
        weights = [tmp_wt.value for tmp_wt in weights]
        ret = self.pax.hist(
            lams,
            bins=self.bins.value,
            stacked=True,
            histtype="stepfilled",
            density=True,
            weights=weights,
        )

        for i, col in enumerate(ret[-1]):
            for reti in col:
                reti.set_facecolor(colors[i])
                reti.set_edgecolor(colors[i])
                reti.set_linewidth(0)
                reti.xy[:, 1] *= Lnorm.to("erg / s").value

    def _generate_and_add_colormap(self):
        """generate the custom color map, linking colours with atomic
        numbers"""

        self.elements_in_kromer_plot = self.line_info

        values = [
            self.cmap(float(i) / float(self._nelements))
            for i in range(self._nelements)
        ]

        custcmap = matplotlib.colors.ListedColormap(values)
        bounds = np.arange(self._nelements) + 0.5
        norm = matplotlib.colors.Normalize(vmin=0, vmax=self._nelements)
        mappable = cm.ScalarMappable(norm=norm, cmap=custcmap)
        mappable.set_array(np.linspace(1, self.zmax + 1, 256))
        if self._species_list != None:
            labels = [
                      species_tuple_to_string(species_string_to_tuple(zi))
                      for zi in self._species_list
                      ]
        else:
            labels = [
                      inv_elements[zi].capitalize()
                      for zi in self.elements_in_kromer_plot[:, 0]
                      ]

        mainax = self.ax
        cbar = plt.colorbar(mappable, ax=mainax)
        cbar.set_ticks(bounds)
        cbar.set_ticklabels(labels)

    def _generate_and_add_legend(self):
        """add legend"""

        bpatch = patches.Patch(color="black", label="photosphere")
        gpatch = patches.Patch(color="grey", label="e-scattering")
        spatch = patches.Patch(color="silver", label="Other species")

        bline = lines.Line2D([], [], color="blue", label="virtual spectrum")
        phline = lines.Line2D(
            [], [], color="red", ls="dashed", label="L at photosphere"
        )

        self.ax.legend(handles=[phline, bline, gpatch, bpatch, spatch])

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
