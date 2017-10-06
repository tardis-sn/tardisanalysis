# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
#  File Name : tardis_minimal_model.py
#
#  Purpose :
#
#  Creation Date : 18-02-2016
#
#  Last Modified : Tue 02 Aug 2016 15:09:26 CEST
#
#  Created By :
#
# _._._._._._._._._._._._._._._._._._._._._.
"""This module provides an interface object which holds the essential
information of a Tardis run to do all the diagnostic tasks for which the
tardisanalysis repository provides tools. Relying on this interface model
objects is a temporary solution until the model storage capability of Tardis
has reached a mature state.
"""
from __future__ import print_function
import pandas as pd
import astropy.units as units
import os
from tardis.model import Radial1DModel


def store_data_for_minimal_model(simulation, buffer_or_fname="minimal_model.hdf5",
                                 path="", mode="virtual"):
    """Simple helper routine to dump all information which are required to
    perform extensive diagnostics with the tardisanalysis tools to an HDF5
    file.

    Parameters
    ----------
    mdl : tardis.model.Radial1DModel
        source tardis model object
    buffer_or_fname : str or file stream
        name of the hdf5 file or file handler (default 'minimal_model.hdf5')
    path : str
        location of the date within the HDF5 file (default '', i.e. its root)
    mode : str
        "virtual" (default), "real" or "both"; store the properties of the
        virtual or the real packet population
    """

    def _save_atom_data(key, path, hdf_store):

        lines = simulation.plasma.atomic_data.lines
        property_path = _get_hdf5_path(path, "atom_data/lines")

        try:
            lines.to_hdf(hdf_store, property_path)
        except AttributeError:
            _to_smallest_pandas(lines).to_hdf(hdf_store, property_path)

    def _save_spectrum_real(key, path, hdf_store):
        """save the real packet spectrum"""

        wave = simulation.runner.spectrum.wavelength.value
        flux = simulation.runner.spectrum.luminosity_density_lambda.value

        luminosity_density = \
            pd.DataFrame.from_dict(dict(wave=wave, flux=flux))
        luminosity_density.to_hdf(hdf_store, os.path.join(path, key))

    def _save_spectrum_virtual(key, path, hdf_store):
        """save the virtual packet spectrum"""

        wave = simulation.runner.spectrum_virtual.wavelength.value
        flux = \
            simulation.runner.spectrum_virtual.luminosity_density_lambda.value

        luminosity_density_virtual = pd.DataFrame.from_dict(dict(wave=wave,
                                                                 flux=flux))
        luminosity_density_virtual.to_hdf(hdf_store, os.path.join(path, key))

    def _save_configuration_dict(key, path, hdf_store):
        """save some information from the basic configuration of the run. For
        now only the time of the simulation is stored
        """
        configuration_dict = dict(
            time_of_simulation=simulation.runner.time_of_simulation,
            R_photosphere=(simulation.model.time_explosion *
                           simulation.model._velocity[0]).to("cm"),
            t_inner=simulation.model.t_inner)
        configuration_dict_path = os.path.join(path, 'configuration')
        pd.Series(configuration_dict).to_hdf(hdf_store,
                                             configuration_dict_path)

    possible_modes = ["real", "virtual", "both"]
    try:
        assert(mode in possible_modes)
    except AssertionError:
        raise ValueError(
            "Wrong mode - possible_modes are {:s}".format(
                ", ".join(possible_modes)))

    if mode == "virtual" and simulation.runner.virt_logging == 0:
        raise ValueError(
            "Virtual packet logging is switched off - cannot store the "
            "properties of the virtual packet population")

    include_from_runner_ = {}
    include_from_spectrum_ = {}
    if mode == "virtual" or mode == "both":
        include_from_runner_.update(
            {'virt_packet_last_interaction_type': None,
             'virt_packet_last_line_interaction_in_id': None,
             'virt_packet_last_line_interaction_out_id': None,
             'virt_packet_last_interaction_in_nu': None,
             'virt_packet_nus': None,
             'virt_packet_energies': None})
        include_from_spectrum_.update(
            {'luminosity_density_virtual': _save_spectrum_virtual})
    if mode == "real" or mode == "both":
        include_from_runner_.update(
            {'last_interaction_type': None,
             'last_line_interaction_in_id': None,
             'last_line_interaction_out_id': None,
             'last_interaction_in_nu': None,
             'output_nu': None,
             'output_energy': None})
        include_from_spectrum_.update(
            {'luminosity_density': _save_spectrum_real})

    include_from_atom_data_ = {'lines': None}
    include_from_model_in_hdf5 = {'runner': include_from_runner_,
                                  'atom_data': _save_atom_data,
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
    print('Writing to path %s' % path)

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
            _save_model_property(getattr(simulation, key), key, path, hdf_store)
        elif callable(include_from_model_in_hdf5[key]):
            include_from_model_in_hdf5[key](key, path, hdf_store)
        else:
            try:
                for subkey in include_from_model_in_hdf5[key]:
                    print(subkey)
                    if include_from_model_in_hdf5[key][subkey] is None:
                        _save_model_property(getattr(getattr(simulation, key),
                                                     subkey), subkey,
                                             os.path.join(path, key),
                                             hdf_store)
                    elif callable(include_from_model_in_hdf5[key][subkey]):
                        include_from_model_in_hdf5[key][subkey](
                            subkey, os.path.join(path, key), hdf_store)
                    else:
                        print('Can not save %s' % str(os.path.join(path, key, subkey)))
            except:
                print('An error occurred while dumping %s to HDF.' % str(os.path.join(path, key)))

    hdf_store.flush()
    hdf_store.close()


class minimal_model(object):
    """Interface object used in many tardisanalysis tools. It holds the
    essential diagnostics information for either the real or the virtual packet
    population of a run.

    This interface object may be filled from an existing Tardis radial1dmodel
    object (for example during the interactive ipython use of Tardis), or
    filled from an HDF5 file (generated by store_data_for_minimal_model).

    Parameters
    ----------
    mode : str
        "real" (default) or "virtual"; defines which packet population is
        stored in the interface object.
    """
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

    def from_interactive(self, simulation):
        """fill the minimal_model from an existing simulation object

        Parameters
        ----------
        simulation : Simulation
            Tardis simulation object holding the run
        """

        self.time_of_simulation = simulation.runner.time_of_simulation
        self.lines = \
            simulation.plasma.atomic_data.lines.reset_index().set_index(
                'line_id')
        self.R_phot = (simulation.model._velocity[0] *
                       simulation.model.time_explosion).to("cm")
        self.t_inner = simulation.model.t_inner

        if self.mode == "virtual":

            self.last_interaction_type = \
                simulation.runner.virt_packet_last_interaction_type
            self.last_line_interaction_in_id = \
                simulation.runner.virt_packet_last_line_interaction_in_id
            self.last_line_interaction_out_id = \
                simulation.runner.virt_packet_last_line_interaction_out_id
            self.last_interaction_in_nu = \
                simulation.runner.virt_packet_last_interaction_in_nu
            self.packet_nus = \
                simulation.runner.virt_packet_nus * units.Hz
            self.packet_energies = \
                simulation.runner.virt_packet_energies * units.erg
            self.spectrum_wave = \
                simulation.runner.spectrum_virtual.wavelength
            self.spectrum_luminosity = \
                simulation.runner.spectrum_virtual.luminosity_density_lambda
        elif self.mode == "real":

            esc_mask = simulation.runner.output_energy >= 0

            self.last_interaction_type = \
                simulation.runner.last_interaction_type[esc_mask]
            self.last_line_interaction_in_id = \
                simulation.runner.last_line_interaction_in_id[esc_mask]
            self.last_line_interaction_out_id = \
                simulation.runner.last_line_interaction_out_id[esc_mask]
            self.last_interaction_in_nu = \
                simulation.runner.last_interaction_in_nu[esc_mask]
            self.packet_nus = \
                simulation.runner.output_nu[esc_mask]
            self.packet_energies = \
                simulation.runner.output_energy[esc_mask]
            self.spectrum_wave = \
                simulation.runner.spectrum.wavelength
            self.spectrum_luminosity = \
                simulation.runner.spectrum.luminosity_density_lambda
        else:
            raise ValueError
        self.last_interaction_in_nu = self.last_interaction_in_nu * units.Hz
        self.readin = True

    def from_hdf5(self, buffer_or_fname):
        """Fill minimal_model from an HDF5 file, which was created by using
        store_data_for_minimal_model.

        Parameters
        ----------
        buffer_or_fname : str, file stream
            name of file object containing the essential Tardis run information
        """
        if isinstance(buffer_or_fname, basestring):
            hdf_store = pd.HDFStore(buffer_or_fname)
        elif isinstance(buffer_or_fname, pd.HDFStore):
            hdf_store = buffer_or_fname
        else:
            raise IOError('Please specify either a filename or an HDFStore')

        self.time_of_simulation = \
            hdf_store["/configuration"].time_of_simulation
        self.lines = hdf_store["/atom_data/lines"].reset_index().set_index(
            'line_id')
        self.R_phot = hdf_store["/configuration"].R_photosphere
        self.t_inner = hdf_store["/configuration"].t_inner

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
