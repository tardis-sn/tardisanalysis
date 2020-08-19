import pandas as pd
import numpy as np
import astropy.units as u
import astropy.constants as uconst
import pyne
import matplotlib.cm as cm
import matplotlib



try:
    import astropy.modeling.blackbody as abb
except ImportError:  # for astropy version < 2.0
    import astropy.analytic_functions as abb
import matplotlib.pyplot as plt



def constructKromerPacketFrame(sim):
    """
    Constructs Pandas DataFrames from a TARDIS
    Simulation object containing logged packet
    properties used for Kromer analysis.

    Parameters
    ----------
    sim : tardis.simulation
        TARDIS simulation object to analyze

    Returns
    -------
    kromer_packet_df : pd.DataFrame
        DataFrame containing tardis packet logged properties

    """
    packet_nus = sim.runner.virt_packet_nus * u.Hz
    packet_nus = packet_nus.to('Hz', u.spectral())
    last_line_interaction_in_nu = sim.runner.virt_packet_last_interaction_in_nu

    packet_lambdas = packet_nus.to('angstrom', u.spectral())
    packet_energies = sim.runner.virt_packet_energies * u.erg

    last_interaction_type = sim.runner.virt_packet_last_interaction_type
    last_line_interaction_out_id = sim.runner.virt_packet_last_line_interaction_out_id
    last_line_interaction_in_id = sim.runner.virt_packet_last_line_interaction_in_id

    kromer_packet_df = pd.DataFrame(
        np.column_stack(
            (packet_nus,
             packet_lambdas,
             packet_energies,
             last_interaction_type,
             last_line_interaction_out_id,
             last_line_interaction_in_id,
             last_line_interaction_in_nu)),

        columns=['nus',
                 'lambdas',
                 'energies',
                 'last_interaction_type',
                 'last_line_interaction_out_id',
                 'last_line_interaction_in_id',
                 'last_line_interaction_in_nu']
    )
    return kromer_packet_df



class KromerPlotLib:
    """
    A new implementation of Kromer plots using pandas and
    more sophisticated plotting.
    """

    @property
    def line_mask(self):
        """
        Mask for selecting packets that have experienced line interaction
        (i.e. filtering out packets with no interactions or only electron
        scattering).
        """
        line_mask = np.logical_and(self.kromer_packet_df['last_interaction_type'] > -1,
                                   self.kromer_packet_df['last_line_interaction_in_id'] > -1)
        return line_mask



    def __init__(self, sim, distance=None):

        # Save sim
        self.sim = sim

        # Set up a dataframe with all the useful
        # sim attributes for making a kromer plot.
        # These are properties of the packets.
        self.kromer_packet_df = constructKromerPacketFrame(sim)

        # Set up similar dataframe with
        # packet properties of packets
        # that have experienced line interaction.
        # We will track atomic number of last
        # interaction here. I could not come up
        # with a good way of doing this in the main
        # dataframe because this requires using the
        # last line interaction id, which is set to -1
        # for electron scattering.

        # Atomic data of lines in sim
        self.lines_df = sim.plasma.atomic_data.lines.reset_index().set_index('line_id')

        # Dataframe of packets that experience line interaction.
        self.kromer_packet_df_line_interaction = self.kromer_packet_df.loc[self.line_mask]

        # Add columns for atomic number of last interaction in/out
        # I think these should be the same in all circumstances
        # but for now I include both since original kromer code
        # does.
        self.kromer_packet_df_line_interaction['last_line_interaction_out_atom'] = \
        self.lines_df.iloc[self.kromer_packet_df_line_interaction['last_line_interaction_out_id']][
            'atomic_number'].values
        self.kromer_packet_df_line_interaction['last_line_interaction_in_atom'] = \
        self.lines_df.iloc[self.kromer_packet_df_line_interaction['last_line_interaction_in_id']][
            'atomic_number'].values

        if distance is None:
            self.isflux = False
            self.distance = None
        else:
            lum_to_flux = 4.0 * np.pi * (distance.to('cm'))**2
            self.kromer_packet_df['energies'] = self.kromer_packet_df['energies'] / lum_to_flux
            self.kromer_packet_df_line_interaction['energies'] = self.kromer_packet_df_line_interaction['energies'] / lum_to_flux
            self.isflux = True
            self.lum_to_flux = lum_to_flux

        return


    def generateKromerPlot(self, packet_wvl_range=None, ax=None, figsize=(10,10),
                           cmapname='jet', show_virtspec=True):
        """
        Generates a Kromer plot. Colors between the x-axis and
        the virtual spectrum show the relative contributions of
        different atoms for the escaped packets. Colors for the
        absorption spectrum show atoms that absorbed packets in
        the given wavelength bin.

        Parameters
        ----------
        packet_wvl_range : [lower_lambda, upper_lambda]*u.angstrom
            Wavelength range to restrict the kromer analysis of escaped
            packets. Emission component will be bounded by these values.

        ax : matplotlib.axes._subplots.AxesSubplot
            Axis on which to create plot.

        figsize : tuple
            Figure size if no axis is specified.

        cmapname : str
            Name of matplotlib color map

        show_virtspec : Boolean
            Set to False to hide the virtual spectrum.

        Returns
        -------

        f, ax :
            The current figure and the axis

        """

        if ax is None:
            self.ax = plt.figure(figsize=figsize).add_subplot(111)
        else:
            self.ax = ax

        # Bin edges
        bins = self.sim.runner.spectrum_virtual._frequency

        # Wavelengths
        wvl = self.sim.runner.spectrum_virtual.wavelength

        self._plotEmission(bins=bins, wvl=wvl,
                           packet_wvl_range=packet_wvl_range,
                           cmapname=cmapname, show_virtspec=show_virtspec)
        self._plotAbsorption(bins=bins, wvl=wvl,
                             packet_wvl_range=packet_wvl_range)
        self._plotPhotosphere()
        self.ax.legend(fontsize=20)
        if self.isflux:
            self.ax.set_xlabel('Wavelength $(\AA)$', fontsize=20)
            self.ax.set_ylabel('$F_{\lambda}$ (erg/s/$cm^{2}/\AA$)', fontsize=20)
        else:
            self.ax.set_xlabel('Wavelength $(\AA)$', fontsize=20)
            self.ax.set_ylabel('$L$ (erg/s/$\AA$)', fontsize=20)

        return plt.gcf(), self.ax


    def _plotEmission(self, bins, wvl, packet_wvl_range, cmapname, show_virtspec):
        if packet_wvl_range is None:
            packet_nu_range_mask = np.array(len(self.kromer_packet_df['nus']) * [True])
            packet_nu_line_range_mask = np.array(len(self.kromer_packet_df_line_interaction['nus']) * [True])
        else:
            packet_nu_range = packet_wvl_range.to('Hz', u.spectral())
            print(packet_nu_range)
            packet_nu_range_mask = np.logical_and(self.kromer_packet_df['nus'].values < packet_nu_range[0].value,
                                                  self.kromer_packet_df['nus'].values > packet_nu_range[1].value)
            packet_nu_line_range_mask = np.logical_and(
                self.kromer_packet_df_line_interaction['nus'].values < packet_nu_range[0].value,
                self.kromer_packet_df_line_interaction['nus'].values > packet_nu_range[1].value)

        # weights are packet luminosities in erg/s
        weights = self.kromer_packet_df.loc[packet_nu_range_mask]['energies'] / self.sim.runner.time_of_simulation
        hist = np.histogram(self.kromer_packet_df.loc[packet_nu_range_mask]['nus'], bins=bins, weights=weights,
                            density=False)

        # No interaction contribution
        # mask_noint selects packets with no interaction
        mask_noint = self.kromer_packet_df.loc[packet_nu_range_mask]['last_interaction_type'] == -1
        hist_noint = np.histogram(self.kromer_packet_df.loc[packet_nu_range_mask]['nus'][mask_noint], bins=bins,
                                  weights=weights[mask_noint], density=False)

        # Electron scattering contribution
        # mask_escatter selects packets that ONLY experience
        # electron scattering.
        mask_escatter = np.logical_and(self.kromer_packet_df.loc[packet_nu_range_mask]['last_interaction_type'] == 1,
                                       self.kromer_packet_df.loc[packet_nu_range_mask][
                                           'last_line_interaction_in_id'] == -1)
        hist_escatter = np.histogram(self.kromer_packet_df.loc[packet_nu_range_mask]['nus'][mask_escatter], bins=bins,
                                     weights=weights[mask_escatter], density=False)

        # Plot virtual spectrum
        if show_virtspec:
            if self.isflux:
                self.ax.plot(self.sim.runner.spectrum_virtual.wavelength,
                             self.sim.runner.spectrum_virtual.luminosity_density_lambda / self.lum_to_flux,
                             '--b', label='Virtual Spectrum', ds='steps-pre', linewidth=1)
            else:
                self.ax.plot(self.sim.runner.spectrum_virtual.wavelength,
                         self.sim.runner.spectrum_virtual.luminosity_density_lambda,
                         '--b', label='Virtual Spectrum', ds='steps-pre', linewidth=1)

        # No Scattering Contribution
        # We convert histogram values to luminosity density lambda.
        lower_level = np.zeros(len(hist_noint[1][1:]))
        L_nu_noint = hist_noint[0] * u.erg / u.s / self.sim.runner.spectrum_virtual.delta_frequency
        L_lambda_noint = L_nu_noint * self.sim.runner.spectrum_virtual.frequency / self.sim.runner.spectrum_virtual.wavelength

        self.ax.fill_between(wvl, lower_level, L_lambda_noint.value, step='pre', color='k', label='No interaction')
        lower_level = L_lambda_noint.value

        # Only Electron Scattering
        L_nu_escatter = hist_escatter[0] * u.erg / u.s / self.sim.runner.spectrum_virtual.delta_frequency
        L_lambda_escatter = L_nu_escatter * self.sim.runner.spectrum_virtual.frequency / self.sim.runner.spectrum_virtual.wavelength

        self.ax.fill_between(wvl, lower_level, lower_level + L_lambda_escatter.value, step='pre', color='grey',
                             label='Electron Scatter Only')
        lower_level = lower_level + L_lambda_escatter.value

        # Groupby packet dataframe by last atom interaction out
        g = self.kromer_packet_df_line_interaction.loc[packet_nu_line_range_mask].groupby(
            by='last_line_interaction_out_atom')

        # Set up color map
        elements_z = g.groups.keys()
        elements_name = [pyne.nucname.name(el) for el in elements_z]
        nelements = len(elements_z)

        # Save color map for later use
        self.cmap = cm.get_cmap(cmapname, nelements)

        values = [self.cmap(i / nelements) for i in range(nelements)]
        custcmap = matplotlib.colors.ListedColormap(values)
        bounds = np.arange(nelements) + 0.5
        norm = matplotlib.colors.Normalize(vmin=0, vmax=nelements)
        mappable = cm.ScalarMappable(norm=norm, cmap=custcmap)

        mappable.set_array(np.linspace(1, nelements + 1, 256))
        labels = elements_name

        # Contribution from each element
        for ind, groupkey in enumerate(elements_z):
            # select subgroup of packet dataframe for specific element.
            group = g.get_group(groupkey)

            # histogram specific element.
            hist_el = np.histogram(group['nus'], bins=bins,
                                   weights=group['energies'] / self.sim.runner.time_of_simulation)

            # Convert to luminosity density lambda
            L_nu_el = hist_el[0] * u.erg / u.s / self.sim.runner.spectrum_virtual.delta_frequency
            L_lambda_el = L_nu_el * self.sim.runner.spectrum_virtual.frequency / self.sim.runner.spectrum_virtual.wavelength

            self.ax.fill_between(wvl, lower_level, lower_level + L_lambda_el.value, step='pre',
                                 color=self.cmap(ind / len(elements_z)), cmap=self.cmap, linewidth=0)
            lower_level = lower_level + L_lambda_el.value

        # Colorbar and Legend
        # self.ax.legend(fontsize=20)
        #self.ax.set_xlabel('Wavelength $(\AA)$', fontsize=20)
        #self.ax.set_ylabel('$L$ (erg/s/$\AA$)', fontsize=20)

        cbar = plt.colorbar(mappable, ax=self.ax)
        cbar.set_ticks(bounds)
        cbar.set_ticklabels(labels)

        return

    def _plotAbsorption(self, bins, wvl, packet_wvl_range):

        # Set up a wavelength range to only analyze emitted packets in that range.
        # packet_wvl_range should be a list [lower_lambda, upper_lambda]*u.angstrom
        # Notice that we convert to frequency, where the lower_lambda is a larger
        # frequency.
        if packet_wvl_range is None:
            packet_nu_line_range_mask = np.array(len(self.kromer_packet_df_line_interaction['nus']) * [True])
        else:
            packet_nu_range = packet_wvl_range.to('Hz', u.spectral())
            packet_nu_line_range_mask = np.logical_and(
                self.kromer_packet_df_line_interaction['nus'].values < packet_nu_range[0].value,
                self.kromer_packet_df_line_interaction['nus'].values > packet_nu_range[1].value)

        abs_lower_level = np.zeros(len(wvl))

        # Groupby packet dataframe by last atom interaction in
        g_abs = self.kromer_packet_df_line_interaction.loc[packet_nu_line_range_mask].groupby(
            by='last_line_interaction_in_atom')
        elements_z = g_abs.groups.keys()
        for ind, groupkey in enumerate(elements_z):
            group = g_abs.get_group(groupkey)
            hist_el = np.histogram(group['last_line_interaction_in_nu'], bins=bins,
                                   weights=group['energies'] / self.sim.runner.time_of_simulation)
            L_nu_el = hist_el[0] * u.erg / u.s / self.sim.runner.spectrum_virtual.delta_frequency
            L_lambda_el = L_nu_el * self.sim.runner.spectrum_virtual.frequency / self.sim.runner.spectrum_virtual.wavelength

            self.ax.fill_between(wvl, abs_lower_level, abs_lower_level - L_lambda_el.value, step='pre',
                                 color=self.cmap(ind / len(elements_z)), cmap=self.cmap, linewidth=0)
            abs_lower_level = abs_lower_level - L_lambda_el.value

        return

    def _plotPhotosphere(self):
        """
        Plots the blackbody luminosity density from the inner
        boundary of the TARDIS simulation.

        """
        Lph = (abb.blackbody_lambda(self.sim.runner.spectrum_virtual.wavelength, self.sim.model.t_inner)
               * 4
               * np.pi ** 2
               * self.sim.model.r_inner[0] ** 2
               * u.sr
               ).to("erg / (AA s)")
        if self.isflux:
            self.ax.plot(self.sim.runner.spectrum_virtual.wavelength, Lph / self.lum_to_flux, '--r', label="Blackbody Photosphere")
        else:
            self.ax.plot(self.sim.runner.spectrum_virtual.wavelength, Lph, '--r', label="Blackbody Photosphere")
        return


