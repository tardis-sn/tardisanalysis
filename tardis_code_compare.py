import os
import pandas as pd
import astropy.units as units


class CodeComparisonOutputFile(object):
    first_column_name = 'VEL'

    def __init__(self, times, data_table, model_name, data_first_column):
        self.times = times
        self.data_table = data_table
        self.data_table.insert(0, 'wav', data_first_column)
        self.model_name = model_name

    @property
    def times_str(self):
        return ' '.join([str(time) for time in self.times])

    @property
    def fname(self):
        return self.data_type + '_{}_tardis.txt'.format(self.model_name)

    def write(self, dest='.'):
        path = os.path.join(dest, self.fname)
        with open(path, mode='w+') as f:
            f.write('#NTIMES: {}\n'.format(len(self.times)))
            f.write('#N{}: {}\n'.format(self.first_column_name,
                                        len(self.data_table)))
            f.write('#TIMES[d]: ' + self.times_str + '\n')
            f.write(self.column_description + '\n')
            self.data_table.to_csv(f, index=False, float_format='%.6E',
                                   sep=' ', header=False)

    @staticmethod
    def get_times_from_simulations(simulations):
        times = [
            sim.model.time_explosion.to(units.day).value for sim in simulations
        ]
        return times

    @classmethod
    def from_simulations(cls, simulations, model_name):
        times = cls.get_times_from_simulations(simulations)
        data_table = cls.get_data_table(simulations)
        data_first_column = cls.get_data_first_column(simulations)
        return cls(times, data_table, model_name, data_first_column)

    @staticmethod
    def get_data_first_column(simulations):
        pass

    @staticmethod
    def get_data_table(simulations):
        pass


class SpectralOutputFile(CodeComparisonOutputFile):
    data_type = 'spectra'
    first_column_name = 'WAVE'
    column_description = ('#wavelength[Ang] flux_t0[erg/s/Ang] '
                          'flux_t1[erg/s/Ang] ... flux_tn[erg/s/Ang]')

    @staticmethod
    def get_data_first_column(simulations):
        return simulations[0].runner.spectrum.wavelength.value

    @staticmethod
    def get_data_table(simulations):
        spectra = [
            sim.runner.spectrum_integrated.luminosity_density_lambda.value
            for sim in simulations
        ]
        return pd.DataFrame(spectra).T


class TGasOutputFile(CodeComparisonOutputFile):
    data_type = 'tgas'
    column_description = '#vel_mid[km/s] Tgas_t0[K] Tgas_t1[K] ... Tgas_tn[K]'
