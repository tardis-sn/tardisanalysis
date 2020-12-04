from __future__ import print_function
import numpy as np
import os
import tardis
import astropy.units as units
import matplotlib.pyplot as plt
from tardis_minimal_model import minimal_model
import pandas as pd

elements = pd.read_csv("elements.csv", names=["chem_symbol", "atomic_no"])
inv_elements = pd.Series(
    elements["chem_symbol"], index=elements["atomic_no"]
).to_dict()


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
            assert type(val) == minimal_model
        except AssertionError:
            raise ValueError("mdl must be a minimal_model instance")
        if not val.readin:
            raise ValueError(
                "empty minimal_model; use from_interactive or "
                "from_hdf5 to fill the model"
            )
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
            self._lam_in = (self.mdl.last_interaction_in_nu).to(
                units.AA, equivalencies=units.spectral()
            )
        return self._lam_in

    @property
    def line_mask(self):
        if self._line_mask is None:
            self._line_mask = (self.lam_in >= self.lam_min) * (
                self.lam_in <= self.lam_max
            )
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
            self._lines_info_unique = self.mdl.lines.ix[self.lines_ids_unique]
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

    def plot_summary(
        self, nlines=None, lam_min=None, lam_max=None, output_filename=None
    ):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.2)

        sym_fname = os.path.join(
            tardis.__path__[0], "data", "atomic_symbols.dat"
        )

        if lam_min is None:
            self.lam_min = np.min(self.mdl.spectrum_wave).value
        else:
            self.lam_min = lam_min

        if lam_max is None:
            self.lam_max = np.max(self.mdl.spectrum_wave).value
        else:
            self.lam_max = lam_max

        _lines_count = self.lines_count[np.argsort(self.lines_count)][::-1]
        _lines_fraction = self.lines_count[np.argsort(self.lines_count)][
            ::-1
        ] / float(self.lines_count.sum())
        _lines_ids = self.lines_ids_unique[np.argsort(self.lines_count)][::-1]

        if nlines is None:
            if len(_lines_count) > 20:
                self.nlines = 20
            else:
                self.nlines = len(_lines_count)
        else:
            if len(_lines_count) > nlines:
                self.nlines = nlines
            else:
                self.nlines = len(_lines_count)

        def ion2roman(ion_value):
            """function to convert ionisation level into roman numeral
            notation"""
            roman_numerals = {1: "I", 4: "IV", 5: "V", 9: "IX", 10: "X"}
            result = ""
            for value, roman_numeral in sorted(
                roman_numerals.items(), reverse=True
            ):
                while int(ion_value) >= int(value):
                    result += roman_numeral
                    ion_value -= value
            return result

        species = []
        wavelengths = []
        labels = []
        angstrom = "$\mathrm{\AA}$"  # included as f-strings cannot have \ in {}
        for line_id in _lines_ids:
            chemical_symbol = inv_elements[
                self.lines_info_unique.ix[line_id].atomic_number
            ].capitalize()
            ionisation_level = ion2roman(
                int(self.lines_info_unique.ix[line_id].ion_number) + 1
            )
            species.append(f"{chemical_symbol} {ionisation_level}")
            wavelengths.append(
                f"{self.lines_info_unique.ix[line_id].wavelength:.3f}"
            )
            labels.append(
                f"{chemical_symbol} {ionisation_level}: {self.lines_info_unique.ix[line_id].wavelength:.3f}{angstrom}"
            )

        # parameters for the output plot
        ax.set_title(
            f"Line Transitions in Range {self.lam_min.value:.1f}{angstrom}$\leq \lambda \leq${self.lam_max.value:.1f}{angstrom}"
        )
        ax.barh(np.arange(self.nlines), _lines_fraction[: self.nlines][::-1])
        ax.set_xlabel("Fraction of Total Line Transitions in Wavelength Range")
        ax.set_yticks(np.arange(len(_lines_fraction[: self.nlines][::-1])))
        ax.set_yticklabels(labels[: self.nlines][::-1], size="medium")
        ax.annotate(
            f"{len(self.lines_ids)} interacting and\nescaping packets\n({np.sum(_lines_count[:self.nlines])} shown)\n{self.nlines} of {len(self.lines_count)} lines displayed",
            xy=(0.95, 0.05),
            xycoords="axes fraction",
            horizontalalignment="right",
            verticalalignment="bottom",
        )

        """if a filename has been specified, then all lines in the region of
        interest are exported to a file"""
        if output_filename != None:
            dataframe = pd.DataFrame(
                {
                    "Species": species,
                    "Wavelength(Angstroms)": wavelengths,
                    "Total no. of transitions": _lines_count,
                    "Fraction of total transitions": _lines_fraction,
                }
            )

            f = open(output_filename, "w")
            f.write(
                f"# Line Transitions in Wavelength Range {self.lam_min.value:.1f} - {self.lam_max.value:.1f} Angstroms\n"
            )
            dataframe.to_csv(f, sep="\t", index=False)
            f.close()
