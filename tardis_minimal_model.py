# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
#  File Name : tardis_minimal_model.py
#
#  Purpose :
#
#  Creation Date : 18-02-2016
#
#  Last Modified : Thu 18 Feb 2016 15:06:09 CET
#
#  Created By :
#
# _._._._._._._._._._._._._._._._._._._._._.
import pandas as pd
import astropy.units as units


class minimal_model(object):
    def __init__(self, mode="real"):

        allowed_modes = ["real", "virtual"]
        try:
            assert(mode in allowed_modes)
        except AssertionError:
            msg = "unknown mode '{:s}';".format(mode) + \
                "allowed modes are {:s}".format(",".join(allowed_modes))
            raise ValueError(msg)

        self.readin = False
        self.mode = mode
        self.lines = None
        self.last_interaction_type = None
        self.last_line_interaction_in_id = None
        self.last_line_interaction_out_id = None
        self.last_interaction_in_nu = None
        self.packet_nus = None
        self.packet_energies = None
        self.spectrum_wave = None
        self.spectrum_luminosity = None
        self.time_of_simulation = None

    def from_interactive(self, mdl):

        self.time_of_simulation = mdl.time_of_simulation
        self.lines = mdl.atom_data.lines
        if self.mode == "virtual":

            if mdl.runner.virt_logging != 1:
                raise ValueError("Tardis must be compiled with the virtual"
                                 " packet logging feature if Kromer plots are"
                                 " to be generated for the virtual packet"
                                 " population")

            self.last_interaction_type = \
                mdl.runner.virt_packet_last_interaction_type
            self.last_line_interaction_in_id = \
                mdl.runner.virt_packet_last_line_interaction_in_id
            self.last_line_interaction_out_id = \
                mdl.runner.virt_packet_last_line_interaction_out_id
            self.last_interaction_in_nu = \
                mdl.runner.virt_packet_last_interaction_in_nu
            self.packet_nus = \
                mdl.runner.virt_packet_nus * units.Hz
            self.packet_energies = \
                mdl.runner.virt_packet_energies * units.erg
            self.spectrum_wave = \
                mdl.spectrum_virtual.wavelength
            self.spectrum_luminosity = \
                mdl.spectrum_virtual.luminosity_density_lambda
        elif self.mode == "real":

            esc_mask = mdl.runner.output_energy >= 0

            self.last_interaction_type = \
                mdl.runner.last_interaction_type[esc_mask]
            self.last_line_interaction_in_id = \
                mdl.runner.last_line_interaction_in_id[esc_mask]
            self.last_line_interaction_out_id = \
                mdl.runner.last_line_interaction_out_id[esc_mask]
            self.last_interaction_in_nu = \
                mdl.runner.last_interaction_in_nu[esc_mask]
            self.packet_nus = \
                mdl.runner.output_nu[esc_mask]
            self.packet_energies = \
                mdl.runner.output_energy[esc_mask]
            self.spectrum_wave = \
                mdl.spectrum.wavelength
            self.spectrum_luminosity = \
                mdl.spectrum.luminosity_density_lambda
        else:
            raise ValueError
        self.last_interaction_in_nu = self.last_interaction_in_nu * units.Hz
        self.readin = True

    def from_hdf5(self, buffer_or_fname):

        if isinstance(buffer_or_fname, basestring):
            hdf_store = pd.HDFStore(buffer_or_fname)
        elif isinstance(buffer_or_fname, pd.HDFStore):
            hdf_store = buffer_or_fname
        else:
            raise IOError('Please specify either a filename or an HDFStore')

        self.time_of_simulation = \
            hdf_store["/configuration"].time_of_simulation
        self.lines = hdf_store["/atom_data/lines"]

        if self.mode == "virtual":

            self.last_interaction_type = \
                hdf_store["/runner/virt_packet_last_interaction_type"]
            self.last_line_interaction_in_id = \
                hdf_store["/runner/virt_packet_last_line_interaction_in_id"]
            self.last_line_interaction_out_id = \
                hdf_store["/runner/virt_packet_last_line_interaction_out_id"]
            self.last_interaction_in_nu = \
                hdf_store["/runner/virt_packet_last_interaction_in_nu"]
            self.packet_nus = \
                hdf_store["/runner/virt_packet_nus"]
            self.packet_energies = \
                hdf_store["/runner/virt_packet_energies"]
            self.spectrum_wave = \
                hdf_store["/spectrum/luminosity_density_virtual"]["wave"]
            self.spectrum_luminosity = \
                hdf_store["/spectrum/luminosity_density_virtual"]["flux"]

        elif self.mode == "real":
            esc_mask = hdf_store["/runner/output_energy"] >= 0

            self.last_interaction_type = \
                hdf_store["/runner/last_interaction_type"][esc_mask]
            self.last_line_interaction_in_id = \
                hdf_store["/runner/last_line_interaction_in_id"][esc_mask]
            self.last_line_interaction_out_id = \
                hdf_store["/runner/last_line_interaction_out_id"][esc_mask]
            self.last_interaction_in_nu = \
                hdf_store["/runner/last_interaction_in_nu"][esc_mask]
            self.packet_nus = \
                hdf_store["/runner/output_nu"][esc_mask]
            self.packet_energies = \
                hdf_store["/runner/output_energy"][esc_mask]
            self.spectrum_wave = \
                hdf_store["/spectrum/luminosity_density"]["wave"]
            self.spectrum_luminosity = \
                hdf_store["/spectrum/luminosity_density"]["flux"]
        else:
            raise ValueError

        self.last_interaction_type = self.last_interaction_type.values
        self.last_line_interaction_in_id = \
            self.last_line_interaction_in_id.values
        self.last_line_interaction_out_id = \
            self.last_line_interaction_out_id.values
        self.last_interaction_in_nu = \
            self.last_interaction_in_nu.values * units.Hz
        self.packet_nus = self.packet_nus.values * units.Hz
        self.packet_energies = self.packet_energies.values * units.erg
        self.spectrum_wave = self.spectrum_wave.values
        self.spectrum_luminosity = self.spectrum_luminosity.values
        self.readin = True
