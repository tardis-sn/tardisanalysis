from __future__ import print_function
import numpy as np
import os
import tardis
import astropy.units as units
import matplotlib.pyplot as plt
from tardis_minimal_model import minimal_model

elements = {'neut': 0, 'h': 1, 'he': 2, 'li': 3, 'be': 4, 'b': 5, 'c': 6, 'n': 7, 'o': 8, 'f': 9, 'ne': 10,
            'na': 11, 'mg': 12, 'al': 13, 'si': 14, 'p': 15, 's': 16, 'cl': 17, 'ar': 18, 'k':  19, 'ca':20,
            'sc': 21, 'ti': 22, 'v': 23, 'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni': 28, 'cu': 29, 'zn': 30,
            'ga': 31, 'ge': 32, 'as': 33, 'se': 34, 'br': 35, 'kr': 36, 'rb': 37, 'sr': 38, 'y': 39, 'zr': 40,
            'nb': 41, 'mo': 42, 'tc': 43, 'ru': 44, 'rh': 45, 'pd': 46, 'ag': 47, 'cd': 48, 'in': 49, 'sn': 50,
            'sb': 51, 'te': 52, 'i': 53, 'xe': 54, 'cs': 55, 'ba': 56, 'la': 57, 'ce': 58, 'pr': 59, 'nd': 60,
            'pm': 61, 'sm': 62, 'eu': 63, 'gd': 64, 'tb': 65, 'dy': 66, 'ho': 67, 'er': 68, 'tm': 69, 'yb': 70,
            'lu': 71, 'hf': 72, 'ta': 73, 'w': 74, 're': 75, 'os': 76, 'ir': 77, 'pt': 78, 'au': 79, 'hg': 80,
            'tl': 81, 'pb': 82, 'bi': 83, 'po': 84, 'at': 85, 'rn': 86, 'fr': 87, 'ra': 88, 'ac': 89, 'th': 90,
            'pa': 91, 'u': 92}
inv_elements = dict([(v, k) for k, v in elements.items()])

class line_identifier(object):

    def __init__(self, mdl):
        self._reset_cache()
        self.mdl = mdl

    def _reset_cache(self):
        self._reset_lam_min()
        self._reset_lam_max()
        self._reset_derived_quantities()

    def _reset_lam_min(self):
        self._lam_min = None

    def _reset_lam_max(self):
        self._lam_max = None

    def _reset_derived_quantities(self):
        self._line_mask = None
        self._lam_in = None
        self._lines_info_unique = None
        self._lines_count = None
        self._lines_ids = None
        self._lines_ids_unique = None

    @property
    def mdl(self):
        return self._mdl

    @mdl.setter
    def mdl(self, val):
        self._reset_cache()

        try:
            assert(type(val) == minimal_model)
        except AssertionError:
            raise ValueError("mdl must be a minimal_model instance")
        if not val.readin:
            raise ValueError("empty minimal_model; use from_interactive or "
                             "from_hdf5 to fill the model")
        self._mdl = val

    @property
    def lam_min(self):
        if self._lam_min is None:
            raise ValueError("lam_min not set")
        return self._lam_min

    @lam_min.setter
    def lam_min(self, val):
        self._reset_derived_quantities()
        try:
            self._lam_min = val.to(units.AA)
        except AttributeError:
            self._lam_min = val * units.AA

    @property
    def lam_max(self):
        if self._lam_max is None:
            raise ValueError("lam_max is not set")
        return self._lam_max

    @lam_max.setter
    def lam_max(self, val):
        self._reset_derived_quantities()
        try:
            self._lam_max = val.to(units.AA)
        except AttributeError:
            self._lam_max = val * units.AA

    @property
    def lam_in(self):
        if self._lam_in is None:
            self._lam_in = ((self.mdl.last_interaction_in_nu).to(units.AA,equivalencies=units.spectral())) # should modify to allow v-packets
        return self._lam_in

    @property
    def line_mask(self):
        if self._line_mask is None:
            self._line_mask = ((self.lam_in >= self.lam_min) * (self.lam_in <= self.lam_max))
        return self._line_mask

    @property
    def lines_ids(self):
        if self._lines_ids is None:
            ids = self.mdl.last_line_interaction_in_id[self.line_mask]
            self._lines_ids = self.mdl.lines.iloc[ids].index
        return self._lines_ids

    @property
    def lines_ids_unique(self):
        if self._lines_ids_unique is None:
            self._lines_ids_unique = np.unique(self.lines_ids)
        return self._lines_ids_unique

    @property
    def lines_info_unique(self):
        if self._lines_info_unique is None:
            self._lines_info_unique = \
                self.mdl.lines.ix[self.lines_ids_unique]
        return self._lines_info_unique

    @property
    def lines_count(self):
        if self._lines_count is None:
            counts = np.bincount(self.lines_ids)
            self._lines_count = counts[counts > 0]
        return self._lines_count

    def identify(self, lam_min, lam_max):
        self.lam_min = lam_min
        self.lam_max = lam_max

    def plot_summary(self, nlines=None, lam_min=None, lam_max=None, output_filename=None):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.2)

        sym_fname = os.path.join(tardis.__path__[0], "data", "atomic_symbols.dat")

        if lam_min == None: # this sets lam_min to 500 Angstroms - unless otherwise specified
            self.lam_min = 500
        else:
            self.lam_min = lam_min
        
        if lam_max == None: # this sets lam_max to 20000 Angstroms - unless otherwise specified
            self.lam_max = 20000
        else:
            self.lam_max = lam_max
        
        if nlines == None: # this caps no. of lines in bar chart at 20 - unless otherwise specified
            self.nlines = 20
        else:
            self.nlines = nlines

        sorting_mask = np.argsort(self.lines_count)
        _lines_count = self.lines_count[sorting_mask][-self.nlines:]
        _freq_lines_in_range  = _lines_count
        _lines_count = _lines_count / float(self.lines_count.sum()) # this is better - ratio of line transitions won't change if more/less bars included now
        _lines_ids = self.lines_ids_unique[sorting_mask][-self.nlines:]

        # function to convert ionisation level into roman numeral notation
        def ion2roman(ion_value):
            roman_numerals = {1:"I", 4:"IV", 5:"V", 9:"IX", 10:"X"}
            result = ""
            for value, roman_numeral in sorted(roman_numerals.items(), reverse=True):
                while int(ion_value) >= int(value):
                    result += roman_numeral
                    ion_value -= value
            return result

        labels = []
        syms_ions = []
        lams = []
        for lid in _lines_ids:
            sym = self.lines_info_unique.ix[lid].atomic_number
            sym = inv_elements[sym].capitalize()
            ion = self.lines_info_unique.ix[lid].ion_number
            ion = ion2roman(int(ion) + 1)
            lam = self.lines_info_unique.ix[lid].wavelength
            label = (f'{sym} {ion}: {lam:.3f}$\AA$')
            syms_ions.append(f'{sym} {ion}')
            lams.append(lam)
            labels.append(label)

        title = f'Line Transitions in Range {self.lam_min.value}$\AA$ $\leq \lambda \leq$ {self.lam_max.value}$\AA$'

        info = f"{len(self.lines_ids)} interacting and\nescaping packets\n({np.sum(_freq_lines_in_range)} shown)\n{len(_lines_count)} of {len(self.lines_count)} lines displayed"

        # if a filename has been specified, then all lines in the region of interest are exported to a file
        if output_filename != None:
            complete_sorting_mask = np.argsort(self.lines_count)
            _complete_lines_count = self.lines_count[complete_sorting_mask]
            _complete_freq_lines_in_range = _complete_lines_count
            _complete_lines_count = _complete_lines_count / float(self.lines_count.sum()) # this is better - ratio of line transitions won't change if more/less bars included now
            _complete_lines_ids = self.lines_ids_unique[complete_sorting_mask]
            complete_labels = []
            complete_syms_ions = []
            complete_lams = []
            for lid in _complete_lines_ids:
                sym = self.lines_info_unique.ix[lid].atomic_number
                sym = inv_elements[sym].capitalize()
                ion = self.lines_info_unique.ix[lid].ion_number
                ion = ion2roman(int(ion) + 1)
                lam = self.lines_info_unique.ix[lid].wavelength
                label = (f'{sym} {ion}: {lam:.3f}$\AA$')
                complete_syms_ions.append(f'{sym} {ion}')
                complete_lams.append(lam)
                complete_labels.append(label)
            output_info = np.flip(np.c_[complete_syms_ions, complete_lams, _complete_freq_lines_in_range, _complete_lines_count], 0)
#             print(output_info)
            with open(output_filename, 'w') as f:
                f.write(f'# Line Transitions in Wavelength Range {self.lam_min.value} - {self.lam_max.value} Angstroms\n')
                f.write(f'# Species\tWavelength[AA]\tNo. of line transitions\tFraction of line transitions\n')
                for item in output_info:
                    f.write(f'{item[0]}\t')
                    f.write(f'{float(item[1]):.3f}\t') # needs to be float to truncate to 3dp
                    f.write(f'{item[2]}\t')
                    f.write(f'{item[3]}\n')
                f.close()

        # parameters for the output plot
        ax.set_title(title)
        ax.barh(np.arange(len(_lines_count)), _lines_count)
        ax.set_xlabel('Fraction of Total Line Transitions in Wavelength Range')
        ax.set_yticks(np.arange(len(_lines_count)))
        ax.set_yticklabels(labels, size="medium")
        ax.annotate(info, xy=(0.95, 0.05), xycoords="axes fraction", horizontalalignment="right", verticalalignment="bottom")

