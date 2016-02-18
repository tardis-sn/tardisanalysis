# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
#  File Name : tardis_minimal_model.py
#
#  Purpose :
#
#  Creation Date : 18-02-2016
#
#  Last Modified : Thu 18 Feb 2016 15:09:45 CET
#
#  Created By :
#
# _._._._._._._._._._._._._._._._._._._._._.
import pandas as pd
import astropy.units as units
import os
import logging

logger = logging.getLogger(__name__)

def store_data_for_minimal_model(mdl, buffer_or_fname="minimal_model.hdf5",
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
