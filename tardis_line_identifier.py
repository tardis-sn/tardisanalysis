# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
#  File Name : tardis_line_identifier.py
#
#  Purpose :
#
#  Creation Date : 18-02-2016
#
#  Last Modified : Thu 18 Feb 2016 15:16:05 CET
#
#  Created By :
#
# _._._._._._._._._._._._._._._._._._._._._.
from __future__ import print_function
import numpy as np
import roman
import util.elements as elem
import astropy.units as units
import matplotlib.pyplot as plt
from tardis_minimal_model import minimal_model


class line_identifier(object):
    def __init__(self, mdl):

        self._reset_cache()

        self.mdl = mdl

    def _reset_cache(self):

        self._line_mask = None
        self._lam_min = None
        self._lam_max = None
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
        self._mdl = val

    @property
    def lam_min(self):
        if self._lam_min is None:
            raise ValueError("lam_min not set")
        return self._lam_min

    @lam_min.setter
    def lam_min(self, val):
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
        try:
            self._lam_max = val.to(units.AA)
        except AttributeError:
            self._lam_max = val * units.AA

    @property
    def lam_in(self):
        if self._lam_in is None:
            self._lam_in = \
                ((self.mdl.last_interaction_in_nu).to(units.AA,
                                                      equivalencies=units.spectral()))
        return self._lam_in

    @property
    def line_mask(self):
        if self._line_mask is None:
            self._line_mask = ((self.lam_in >= self.lam_min) *
                               (self.lam_in <= self.lam_max))
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

    def print_summary(self):

        print("Not yet implemented")

        pass

    def plot_summary(self, nlines=20):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ehelper = elem.elements_helper()
        ehelper.capitalize()

        sorting_mask = np.argsort(self.lines_count)
        _lines_count = self.lines_count[sorting_mask][-nlines:]
        _lines_count = _lines_count / float(_lines_count.sum())
        _lines_ids = self.lines_ids_unique[sorting_mask][-nlines:]

        labels = []
        for lid in _lines_ids:
            sym = self.lines_info_unique.ix[lid].atomic_number
            sym = ehelper.inv_elements[sym]
            ion = self.lines_info_unique.ix[lid].ion_number
            ion = roman.toRoman(int(ion) + 1)
            lam = self.lines_info_unique.ix[lid].wavelength
            label = \
                r"{:<2s} {:>3s}: {:.2f}$\,\mathrm{{\AA}}$".format(sym,
                                                                  ion, lam)
            labels.append(label)

        title = \
            r"${:.2f}\,\mathrm{{\AA}}\leq\lambda\leq{:.2f}\,\mathrm{{\AA}}$"
        title = title.format(self.lam_min.value, self.lam_max.value)

        info = "{:d} interacting and escaping packets\n{:d} of {:d} lines displayed"
        info = info.format(len(self.lines_ids), len(_lines_count),
                           len(self.lines_count))

        ax.set_title(title)
        ax.barh(np.arange(len(_lines_count)), _lines_count)
        ax.set_yticks(np.arange(len(_lines_count)) + 0.4)
        ax.set_yticklabels(labels)
        ax.annotate(info, xy=(0.95, 0.05), xycoords="axes fraction",
                    horizontalalignment="right", verticalalignment="bottom")
