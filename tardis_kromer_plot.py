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
from tardis.util.base import (
    species_string_to_tuple,
    species_tuple_to_string,
    roman_to_int,
    int_to_roman,
)

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
            self._lam_noint = (csts.c.cgs / (self.mdl.packet_nus[self.noint_mask])).to(
                units.AA
            )
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
                self.mdl.packet_energies[self.noint_mask] / self.mdl.time_of_simulation
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

        # this generates the 4-digit ID for all transitions in the model (e.g. Fe III line --> 2602)
        self.line_out_infos_within_xlims["ion_id"] = (
            self.line_out_infos_within_xlims["atomic_number"] * 100
            + self.line_out_infos_within_xlims["ion_number"]
        )

        # this is a list that will hold which elements should all be in the same colour.
        # This is used if the user requests a mix of ions and elements.
        self.keep_colour = []
        # this reads in the species specified by user and generates the 4-digit ID keys for them
        if self._species_list is not None:
            # create a list of the ions ids requested by species_list
            requested_species_ids = []
            # check if there are any digits in the species list. If there are then exit
            # species_list should only contain species in the Roman numeral format, e.g. Si II, and each ion must contain a space
            if any(char.isdigit() for char in " ".join(self._species_list)) == True:
                raise ValueError(
                    "All species must be in Roman numeral form, e.g. Si II"
                )
            else:
                # go through each of the request species. Check whether it is an element or ion (ions have spaces).
                # If it is an element, add all possible ions to the ions list. Otherwise just add the requested ion
                for species in self._species_list:
                    if " " in species:
                        requested_species_ids.append(
                            [
                                species_string_to_tuple(species)[0] * 100
                                + species_string_to_tuple(species)[1]
                            ]
                        )
                    else:
                        atomic_number = int(elements[species.lower()])
                        requested_species_ids.append(
                            [atomic_number * 100 + i for i in np.arange(atomic_number)]
                        )
                        self.keep_colour.append(atomic_number)
                self.requested_species_ids = [
                    species_id for list in requested_species_ids for species_id in list
                ]

        # now we are getting the list of unique values for 'ion_id' if we would like to use species. Otherwise we get unique atomic numbers
        if self._species_list is not None:
            self._elements_in_kromer_plot = np.c_[
                np.unique(
                    self.line_out_infos_within_xlims.ion_id.values,
                    return_counts=True,
                )
            ]
        else:
            self._elements_in_kromer_plot = np.c_[
                np.unique(
                    self.line_out_infos_within_xlims.atomic_number.values,
                    return_counts=True,
                )
            ]

        # if the length of self._elements_in_kromer_plot exceeds the requested number
        # of elements to be included in the colourbar, then this if statement applies
        if self._species_list is not None:
            # if we have specified a species list then only take those species that are requested
            mask = np.in1d(
                self._elements_in_kromer_plot[:, 0], self.requested_species_ids
            )
            self._elements_in_kromer_plot = self._elements_in_kromer_plot[mask]
        elif len(self._elements_in_kromer_plot) > self._nelements:
            # if nelements is specified, then sort to find the top contributing elements, pick the top nelements, and sort back by atomic number
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
            # if the length of self._elements_in_kromer_plot is less than the requested number of elements in the model,
            # then this requested length is updated to be the length of length of self._elements_in_kromer_plot
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
        nelements: int or None
            number of elements that should be included in the Kromer plots.
            The top nelements are determined based on those with the most packet 
            interactions
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

        # the species list can contain either a specific element, a specific ion, a range of ions, or any combination of these
        # if the list contains a range of ions, separate each one into a new entry in the species list
        full_species_list = []
        if species_list is not None:
            for species in species_list:
                # check if a hyphen is present. If it is, then it indicates a range of ions. Add each ion in that range to the list
                if "-" in species:
                    element = species.split(" ")[0]
                    first_ion_numeral = roman_to_int(
                        species.split(" ")[-1].split("-")[0]
                    )
                    second_ion_numeral = roman_to_int(
                        species.split(" ")[-1].split("-")[-1]
                    )
                    for i in np.arange(first_ion_numeral, second_ion_numeral + 1):
                        full_species_list.append(element + " " + int_to_roman(i))
                else:
                    full_species_list.append(species)
            self._species_list = full_species_list
        else:
            self._species_list = None

        # if no nelements and no species list is specified, then the number of elements to be included
        # in the colourbar is determined from the list of unique elements that appear in the model
        if nelements is None and species_list is None:
            self._nelements = len(np.unique(self.line_out_infos.atomic_number.values))
        elif nelements is None and species_list is not None:
            # if species_list has been specified, then the number of elements to be included is set to the length of that list
            self._nelements = len(self._species_list)
        else:
            # if nelements has been specified, then the number of elements to be included is set to the length of that list
            self._nelements = nelements

        if xlim is None:
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

        # get the elements/species to be included in the plot
        self._elements_in_kromer_plot = self.line_info

        # this will reset nelements if species_list is turned on
        # it's possible to request a species that doesn't appear in the plot
        # this will ensure that species isn't counted when determining labels and colours
        if self._species_list is not None:
            labels = []
            for species in self._species_list:
                if " " in species:
                    atomic_number = species_string_to_tuple(species)[0]
                    ion_number = species_string_to_tuple(species)[1]

                    species_id = atomic_number * 100 + ion_number
                    if species_id in self._elements_in_kromer_plot:
                        labels.append(species)
                else:
                    labels.append(species)
            self._nelements = len(labels)

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

        # if species_list is entered, the ion_id will be used to determine the colours, etc
        if self._species_list is not None:
            values_to_compare = np.unique(
                self.line_out_infos_within_xlims.ion_id.values,
                return_counts=False,
            )
        else:
            # otherwise, if there is no species_list, then the atomic_number is used for colours, etc.
            values_to_compare = np.unique(
                self.line_out_infos_within_xlims.atomic_number.values,
                return_counts=False,
            )

        # this first for loop is to go through all elements and colour all elements as 'Other' if they weren't requested
        # or among the top nelements. The reason to do it twice is to ensure that the colours are stacked appropriately,
        # e.g. all 'other' are together
        for zi in values_to_compare:
            # zi is the unique 4-digit code for the species in the model
            # determining the atomic and ion numbers for all ions in our model
            if self._species_list is not None:
                ion_number = zi % 100
                atomic_number = (zi - ion_number) / 100
            else:
                atomic_number = zi

            # if the ion is not included in our list for the colourbar, then its contribution
            # is added here to the miscellaneous grey shaded region of the plot
            if zi not in self._elements_in_kromer_plot[:, 0]:
                # if species_list is given then use the atomic number and ion_number to peforming masking
                if self._species_list is not None:
                    mask = (
                        self.line_out_infos.atomic_number.values == atomic_number
                    ) & (self.line_out_infos.ion_number.values == ion_number)
                else:
                    # otherwise only elements are plotted, so only use the atomic number
                    mask = self.line_out_infos.atomic_number.values == atomic_number

                lams.append((csts.c.cgs / (self.line_out_nu[mask])).to(units.AA))
                weights.append(self.line_out_L[mask] / self.mdl.time_of_simulation)
                colors.append("silver")

        ii = 0
        # this is a variable that will allow for situations where elements and ions are requested in the same list
        # this will ensure that any ions for a requested element will all be coloured the same
        previous_atomic_number = 0
        for zi in values_to_compare:
            # zi is the unique 4-digit code for the species in the model
            # determining the atomic and ion numbers for all ions in our model
            if self._species_list is not None:
                ion_number = zi % 100
                atomic_number = (zi - ion_number) / 100
            else:
                atomic_number = zi

            # if the ion is included in our list for the colourbar, then its
            # contribution is added here as a colour to the plot
            if zi in self._elements_in_kromer_plot[:, 0]:
                # if this is the first ion, don't update the colour
                if (previous_atomic_number == 0):
                    ii = ii
                    previous_atomic_number = atomic_number
                elif atomic_number in self.keep_colour:
                    # if this ion is grouped into an element, check whether this is the first ion of that element to occur
                    # if it is, then update the colour. If it isn't then don't update the colour
                    if previous_atomic_number == atomic_number:
                        ii = ii
                        previous_atomic_number = atomic_number
                    else:
                        ii = ii +1
                        previous_atomic_number = atomic_number
                else:
                    ii = ii + 1
                    previous_atomic_number = atomic_number
                if self._species_list is not None:
                    mask = (
                        self.line_out_infos.atomic_number.values == atomic_number
                    ) & (self.line_out_infos.ion_number.values == ion_number)
                else:
                    mask = self.line_out_infos.atomic_number.values == atomic_number

                lams.append((csts.c.cgs / (self.line_out_nu[mask])).to(units.AA))
                weights.append(self.line_out_L[mask] / self.mdl.time_of_simulation)
                colors.append(self.cmap(float(ii) / float(self._nelements)))

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

        if self._species_list is not None:
            values_to_compare = np.unique(
                self.line_out_infos_within_xlims.ion_id.values,
                return_counts=False,
            )
        else:
            values_to_compare = np.unique(
                self.line_out_infos_within_xlims.atomic_number.values,
                return_counts=False,
            )

        for zi in values_to_compare:
            # zi is the unique 4-digit code for the species in the model
            # determining the atomic and ion numbers for all ions in our model
            if self._species_list is not None:
                ion_number = zi % 100
                atomic_number = (zi - ion_number) / 100
            else:
                atomic_number = zi

            # if the ion is not included in our list for the colourbar, then its contribution
            # is added here to the miscellaneous grey shaded region of the plot
            if zi not in self._elements_in_kromer_plot[:, 0]:

                if self._species_list is not None:
                    mask = (
                        self.line_out_infos.atomic_number.values == atomic_number
                    ) & (self.line_out_infos.ion_number.values == ion_number)
                else:
                    mask = self.line_out_infos.atomic_number.values == atomic_number

                lams.append((csts.c.cgs / (self.line_in_nu[mask])).to(units.AA))
                weights.append(self.line_in_L[mask] / self.mdl.time_of_simulation)
                colors.append("silver")
        ii = 0
        previous_atomic_number = 0
        for zi in values_to_compare:
            # zi is the unique 4-digit code for the species in the model
            # determining the atomic and ion numbers for all ions in our model
            if self._species_list is not None:
                ion_number = zi % 100
                atomic_number = (zi - ion_number) / 100
            else:
                atomic_number = zi

            # if the ion is included in our list for the colourbar, then its
            # contribution is added here as a unique colour to the plot
            if zi in self._elements_in_kromer_plot[:, 0]:
                # if this is the first ion, don't update the colour
                if (previous_atomic_number == 0):
                    ii = ii
                    previous_atomic_number = atomic_number
                elif atomic_number in self.keep_colour:
                    # if this ion is grouped into an element, check whether this is the first ion of that element to occur
                    # if it is, then update the colour. If it isn't then don't update the colour
                    if previous_atomic_number == atomic_number:
                        ii = ii
                        previous_atomic_number = atomic_number
                    else:
                        ii = ii +1
                        previous_atomic_number = atomic_number
                else:
                    ii = ii + 1
                    previous_atomic_number = atomic_number
                if self._species_list is not None:
                    mask = (
                        self.line_out_infos.atomic_number.values == atomic_number
                    ) & (self.line_out_infos.ion_number.values == ion_number)
                else:
                    mask = self.line_out_infos.atomic_number.values == atomic_number

                lams.append((csts.c.cgs / (self.line_in_nu[mask])).to(units.AA))
                weights.append(self.line_in_L[mask] / self.mdl.time_of_simulation)
                colors.append(self.cmap(float(ii) / float(self._nelements)))

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

        values = [
            self.cmap(float(i) / float(self._nelements)) for i in range(self._nelements)
        ]

        custcmap = matplotlib.colors.ListedColormap(values)
        bounds = np.arange(self._nelements) + 0.5
        norm = matplotlib.colors.Normalize(vmin=0, vmax=self._nelements)
        mappable = cm.ScalarMappable(norm=norm, cmap=custcmap)
        mappable.set_array(np.linspace(1, self.zmax + 1, 256))

        # if a species_list has been specified...
        if self._species_list is not None:
            labels = []
            for zi in self._elements_in_kromer_plot:

                ion_number = zi[0] % 100
                atomic_number = (zi[0] - ion_number) / 100

                ion_numeral = int_to_roman(ion_number + 1)
                # using elements dictionary to get atomic symbol for the species
                atomic_symbol = inv_elements[atomic_number].capitalize()

                # if the element was requested, and not a specific ion, then add the element symbol to the label list
                if (atomic_number in self.keep_colour) & (atomic_symbol not in labels):
                    # compiling the label, and adding it to the list
                    label = f"{atomic_symbol}"
                    labels.append(label)
                elif atomic_number not in self.keep_colour:
                    # otherwise add the ion to the label list
                    label = f"{atomic_symbol}$\,${ion_numeral}"
                    labels.append(label)

        else:
            # if no species_list specified, generate the labels this way
            labels = [
                inv_elements[zi].capitalize()
                for zi in self._elements_in_kromer_plot[:, 0]
            ]

        mainax = self.ax
        cbar = plt.colorbar(mappable, ax=mainax)
        cbar.set_ticks(bounds)
        cbar.set_ticklabels(labels)

    def _generate_and_add_legend(self):
        """add legend"""

        spatch = patches.Patch(color="silver", label="Other species")
        gpatch = patches.Patch(color="grey", label="e-scattering")
        bpatch = patches.Patch(color="black", label="Photosphere")

        bline = lines.Line2D([], [], color="blue", label="Virtual spectrum")
        phline = lines.Line2D(
            [], [], color="red", ls="dashed", label="L at photosphere"
        )

        self.ax.legend(handles=[phline, bline, spatch, gpatch, bpatch])

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
