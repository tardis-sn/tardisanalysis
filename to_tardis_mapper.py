# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
#  File Name : to_tardis_mapper.py
#
#  Purpose : Generate Tardis input from generic explosion models
#
#  Creation Date : 29-02-2016
#
#  Last Modified : Fri 04 Mar 2016 15:23:22 CET
#
#  Created By : unoebauer
#
# _._._._._._._._._._._._._._._._._._._._._.
"""A simple tool to map the output of SN explosion calculations or any ejecta
model into Tardis, using its capability to work with specific density and
abundance files.
"""
import numpy as np
import logging
import astropy.units as units
from pyne import nucname, material

logger = logging.getLogger(__name__)

try:
    material.Material().decay
except AttributeError:
    logger.critical("PyNe module outdated: version >= 0.5 is required")
    raise ImportError("No recent PyNe module found")

# maximum atomic number
zmax = 30


class original_model(object):
    """Simple model interface. It and its derived classes should provide a
    common interface for all possible explosion model formats.

    This class only requires a minimal set of information, from which all the
    remaining quantities are constructed (under the assumption of homologous
    expansion). For each radial shell, the inner and the outer shell radius
    have to be provided, in addition with the shell density, the mass fractions
    of all stable elements and of all radioactive elements. The final
    information to be passed is the time since explosion.

    Velocity and shell masses are generated from these input data, by assuming
    perfect homologous expansion, (i.e. $u = r/t$) and spherical symmetry.

    Notes
    -----

    The to_tardis_mapper tool expects that stable_abundances + radio_abundance
    = 1. In other words, the radioactive isotopes should not be contained in
    the mass fractions stores in stable_abundances.

    Parameters
    ----------
    """
    def __init__(self):

        self._ro = None
        self._ri = None
        self._t = None
        self._rho = None

        self._nzones = None
        self._vo = None
        self._vi = None
        self._dm = None
        self._mr = None

        self._stable_abundances = None
        self._radio_abundances = None

    def _reset_cached_variables(self):
        """Resets the cached derived quantities. This routine should be called
        every time one of the principle inputs (such as density) is reset.
        """

        self._nzones = None
        self._vo = None
        self._vi = None
        self._dm = None
        self._mr = None

    @property
    def ro(self):
        """outer shell edge for each radial cell"""
        if self._ro is None:
            raise ValueError(
                "You have to read-in or assign the outer cell radii first")
        return self._ro

    @ro.setter
    def ro(self, val):
        try:
            val.to("cm")
        except (AttributeError, units.UnitConversionError):
            raise ValueError(
                "ro must be a valid length astropy.units.Quantity")
        self._ro = val
        self._reset_cached_variables()

    @property
    def ri(self):
        """inner shell edge for each radial cell"""
        if self._ri is None:
            raise ValueError(
                "You have to read-in or assign the inner cell radii first")
        return self._ri

    @ri.setter
    def ri(self, val):
        try:
            val.to("cm")
        except (AttributeError, units.UnitConversionError):
            raise ValueError(
                "ri must be a valid length astropy.units.Quantity")
        self._ri = val
        self._reset_cached_variables()

    @property
    def rho(self):
        """cell density"""
        if self._rho is None:
            raise ValueError(
                "You have to read-in or assign the cell density first")
        return self._rho

    @rho.setter
    def rho(self, val):
        try:
            val.to("g/cm^3")
        except (AttributeError, units.UnitConversionError):
            raise ValueError(
                "rho must be a valid mass density astropy.units.Quantity")
        self._rho = val
        self._reset_cached_variables()

    @property
    def t(self):
        """time since explosion"""
        if self._t is None:
            raise ValueError(
                "You have to read-in or assign the time since explosion first")
        return self._t

    @t.setter
    def t(self, val):
        try:
            val.to("s")
        except (AttributeError, units.UnitConversionError):
            raise ValueError(
                "t must be a valid time astropy.units.Quantity")
        self._t = val
        self._reset_cached_variables()

    @property
    def nzones(self):
        """number of shells in the model"""
        if self._nzones is None:
            self._nzones = len(self.ro)

        return self._nzones

    @property
    def vo(self):
        """fluid velocity at outer cell edge"""
        if self._vo is None:
            self._vo = self.ro / self.t
        return self._vo

    @property
    def vi(self):
        """fluid velocity at inner cell edge"""
        if self._vi is None:
            self._vi = self.ri / self.t
        return self._vi

    @property
    def dm(self):
        """mass contained within a cell"""
        if self._dm is None:
            self._dm = 4. * np.pi / 3. * (self.ro**3 - self.ri**3) * self.rho
        return self._dm

    @property
    def mr(self):
        """mass enclosed by shells outer radius"""
        if self._mr is None:
            _mr = np.zeros(self.nzones) * self.dm.unit
            _mr[0] = self.dm[0]
            for i in xrange(1, self.nzones):
                _mr[i] = _mr[i-1] + self.dm[i]
            self._mr = _mr
        return self._mr

    @property
    def stable_abundances(self):
        """a dictionary holding the mass fractions of all stable elements up to
        Z = zmax. The proton number serves as dictionary key
        """
        if self._stable_abundances is None:
            tmp = {i+1: np.zeros(self.nzones) for i in xrange(zmax)}
            self._stable_abundances = tmp

        return self._stable_abundances

    @property
    def radio_abundances(self):
        """a dictionary holding the mass fractions of all radioactive isotopes
        at time = t. Strings such as 'Ni56', i.e. containing the Element symbol
        and the mass number, serve as dictionary keys
        """
        if self._radio_abundances is None:
            self._radio_abundances = {}
        return self._radio_abundances

    @property
    def complete(self):
        """A simple flag determining whether all quantities necessary for the
        mapping process have been set/read-in"""

        _complete = False
        try:
            self.t
            self.rho
            self.ro
            self.ri
            _complete = True
        except ValueError:
            pass

        try:
            X = np.array(
                [self.stable_abundances[z] for z in xrange(1, zmax+1)])
            _complete = _complete * (X > 0).any()
        except ValueError:
            _complete = False

        return _complete

    def read_density(self, fname):
        """A prototype for reading the radius and density information from
        file"""

        pass

    def read_abundances(self, fname):
        """A prototype for reading the stable and radioactive elemental
        abundances from file"""

        pass


class w7_model(original_model):
    """A simple interface class for the W7 model.

    Notes
    -----
    The original W7 model has been presented by [1]_. This reader is designed
    particular version calculated by [2]_.

    References
    ----------

    .. [1] Nomoto et al. "Accreting white dwarf models of Type I supernovae.
       III - Carbon deflagration supernovae" ApJ, 1984, 286, 644-658
    .. [2] Iwamoto et al. "Nucleosynthesis in Chandrasekhar Mass Models for
       Type IA Supernovae and Constraints on Progenitor Systems and
       Burning-Front Propagation" ApJS, 1999, 125, 439-462
    """
    def __init__(self):
        super(w7_model, self).__init__()

    def read_density(self, fname):
        """Read the radius and density of the W7 model from file.

        Parameters
        ----------
        fname : str
            Name of the hydrodynamics file of the W7 model
        """

        with open(fname, "r") as f:
            # read header
            buffer = f.readline().rsplit()
            self.t = float(buffer[2]) * units.s
            buffer = f.readline()

            # read main data block
            data = np.loadtxt(f)
        self.ro = data[:, 2] * units.cm
        self.ri = np.insert(self.ro, 0, 0 * units.cm)[:-1]
        self.rho = data[:, 3] * units.g / units.cm**3

    def read_abundances(self, fname):
        """Read elemental abundances of the W7 model from file.

        Parameters
        ----------
        fname : str
            Name of the nucleosynthesis file of the W7 model
        """

        with open(fname, "r") as f:
            data = np.loadtxt(f, skiprows=1)
        # the file contains abundances for elements from Z=1 to Z=32 (Ge).
        for i in xrange(np.max([zmax, 32])):
            self.stable_abundances[i+1] = data[:, i]

        # the last three columns contain the mass fractions of the radioactive
        # isotopes nickel-56, cobalt-56, nickel-57
        self.radio_abundances["ni56"] = data[:, 32]
        self.radio_abundances["co56"] = data[:, 33]
        self.radio_abundances["ni57"] = data[:, 34]


class to_tardis_mapper(object):
    """A simple mapper object, transforming explosion models, stored in the
    original_model interface classes, into Tardis specific structure input
    files

    Parameters
    ----------
    orig_model : original_model or derived classes
        interface object holding the essential data of the original explosion
        model
    """
    def __init__(self, orig_model):

        if not orig_model.complete:
            logger.critical("Not all necessary information have been"
                            " set/read-in in the original model")
            raise ValueError("Check original_model for completeness")
        self.orig = orig_model

    def remap(self, v, t, decay=True, write_density=True,
              density_fname="tardis_densities.dat", write_abundances=True,
              abundance_fname="tardis_abundances.dat", be_fix=True, be_to_z=6):
        """Perform the remapping of the original model onto a specified Tardis
        velocity grid. A homologous expansion from the time of the model to the
        time since explosion used in the Tardis calculation is automatically
        performed. Optionally, the decay of the radioactive isotopes can also
        be performed. The remapped model can then be written to Tardis files.

        Notes
        -----
        When supplying specific structure files to Tardis, it expects to find
        also the density and composition of the photosphere. Thus, if the
        ejecta above the photosphere is supposed to be described by 20 shells,
        the data files have to contain 21 rows. The specific density and
        composition of the photosphere is not important since this information
        is not used within Tardis - the photospheric velocity (i.e. the
        velocity of the first row) is used!

        Parameters
        ----------
        v : array-like astropy.units.Quantity
            Tardis velocity grid, interpreted as the velocity at the outer
            shell edges. Mind the photospheric shell!
        t : scalar astropy.units.Quantity
            start time (time since explosion) of the Tardis calculation
        decay : bool
            perform decay of the radioactive isotopes (default True)
        write_density : bool
            write remapped density to a Tardis specific structure file (default
            True)
        density_fname : str
            name of the Tardis specific structure file (default
            'tardis_densities.dat')
        write_abundances : bool
            write remapped abundances to a Tardis specific abundance file
            (default True)
        abundance_fname : str
            name of the Tardis specific structure file (default
            'tardis_abundances.dat')
        be_fix : bool
            perform the Beryllium fix (see #438), i.e. set Be to zero and add
            its abundance to a different element (default True)
        be_to_z : int
            proton number of the destination element for the Be fix (default 6,
            i.e. carbon)
        """

        self._remap_density(v, t)
        self._remap_abundances()

        if decay:
            self._decay_abundances()
        else:
            self._copy_radio_abundances()
        if be_fix:
            self._be_fix(to_z=be_to_z)
        if write_density:
            self._write_tardis_density_file(fname=density_fname)
        if write_abundances:
            self._write_tardis_abundance_file(fname=abundance_fname)

    def _remap_density(self, v, t):
        """Remap density of the original_model onto the provided velocity
        grid and perform a homologous expansion of the density until the
        defined start time of Tardis

        Notes
        -----

        For the remapping, we interpolate mr in the v**3 space (corresponds to
        volume space).

        Parameters
        ----------
        v : array-like astropy.units.Quantity
            Tardis velocity grid, interpreted as the velocity at the outer
            shell edges. Mind the photospheric shell!
        t : scalar astropy.units.Quantity
            start time (time since explosion) of the Tardis calculation
        """

        self.t = t
        self.N_interp = len(v) - 1
        self.v_interp_r = v[1:].to("cm/s")
        self.v_interp_l = v[:-1].to("cm/s")

        V_interp = 4. / 3. * np.pi * ((self.v_interp_r * t).to("cm")**3 -
                                      (self.v_interp_l * t).to("cm")**3)

        vrorig = self.orig.vo.to("cm/s")
        mrorig = self.orig.mr.to("solMass")

        mr_interp = np.interp(
            np.insert(self.v_interp_r, 0, self.v_interp_l[0])**3,
            np.append(0 * vrorig.unit, vrorig)**3,
            np.append(0 * mrorig.unit, mrorig)) * mrorig.unit

        self.dm_interp = (mr_interp[1:] - mr_interp[:-1])
        self.rho_interp = (self.dm_interp / V_interp).to("g/cm^3")

    def _remap_abundances(self):
        """Remap abundances for the original model onto the grid defined in
        _remap_density. This routine has be called after _remap_density is
        performed.

        Notes
        -----
        Must be called after _remap_density.

        Raises
        ------
        AttributeError if called before _remap_density
        """
        try:
            self.v_interp_l
            self.v_interp_r
        except AttributeError:
            logger.critical("Density must be remapped before abundances are"
                            " addressed")
            raise AttributeError("no v_interp_r; call _remap_density first")

        self.abundances_interp = {}
        self.radio_abundances_interp = {}

        def remap_species(Xorig):
            """helper routine to remap one specific elemental species onto the
            new velocity grid

            Parameters
            ----------
            Xorig : numpy.ndarray
                original mass fraction of the original model

            Returns
            -------
            X_interp : numpy.ndarray
                remapped mass fractions, corresponding to the Tardis model
            """

            vrorig = self.orig.vo.to("cm/s")
            Xrorig = np.zeros(len(Xorig)) * self.orig.dm.unit

            Xrorig[0] = self.orig.dm[0] * Xorig[0]
            for i in xrange(1, self.orig.nzones):
                Xrorig[i] += Xrorig[i-1] + self.orig.dm[i] * Xorig[i]

            X_interp = np.interp(
                np.insert(self.v_interp_r, 0, self.v_interp_l[0])**3,
                np.append(0 * vrorig.unit, vrorig)**3,
                np.append(0 * Xrorig.unit, Xrorig)) * Xrorig.unit

            X_interp = X_interp[1:] - X_interp[:-1]

            return (X_interp / self.dm_interp).to("").value

        # remap stable elements
        for z in xrange(1, zmax+1):

            Xorig = self.orig.stable_abundances[z]
            X_interp = remap_species(Xorig)

            self.abundances_interp[z] = X_interp

        # remap radioactive isotopes
        for ident in self.orig.radio_abundances.keys():

            Xorig = self.orig.radio_abundances[ident]

            X_interp = remap_species(Xorig)

            self.radio_abundances_interp[ident] = X_interp

    def _copy_radio_abundances(self):
        """Copies the radioactive isotopes onto the corresponding stable
        elements. This routine is intended for uses of the mapper during which
        the decay is neglected

        Notes
        -----
        Must be called after _remap_abundances.

        Raises
        ------
        AttributeError if called before _remap_abundances
        """
        try:
            self.radio_abundances_interp
        except AttributeError:
            logger.critical("Abundances must be remapped before radioactive:"
                            " isotopes are copied onto the stable elements")
            raise AttributeError("no radio_abundances_interp; call"
                                 " _remap_abundances first")

        for ident in self.radio_abundances_interp.keys():
            elemid = nucname.id(ident)
            z = nucname.znum(elemid)
            self.abundances_interp[z] = \
                self.abundances_interp[z] + self.radio_abundances_interp[ident]

    def _decay_abundances(self):
        """Determines the decay of all radioactive isotopes. Afterwards their
        mass fractions are added to the stable elements.

        Notes
        -----
        Must be called after _remap_abundances.

        Raises
        ------
        AttributeError if called before _remap_abundances
        """
        try:
            self.radio_abundances_interp
        except AttributeError:
            logger.critical("Abundances must be remapped before decay is"
                            " handled")
            raise AttributeError("no radio_abundances_interp; call "
                                 "_remap_abundances first")

        for i in xrange(self.N_interp):
            comp = {}
            mass = 0
            for ident in self.radio_abundances_interp.keys():
                Xi = self.radio_abundances_interp[ident][i]
                mass += Xi
                comp[nucname.id(ident)] = Xi
            inp = material.Material(comp, mass=mass)
            res = inp.decay(
                (self.t - self.orig.t).to("s").value).mult_by_mass()

            for item in res.items():
                z = nucname.znum(item[0])
                self.abundances_interp[z][i] = \
                    self.abundances_interp[z][i] + item[1]

    def _be_fix(self, to_z=6):
        """Set Beryllium abundance to zero and add its mass fraction to another
        (specified) element. This is done to avoid issue #438.

        Notes
        -----
        Must be called after _remap_abundances.

        Parameters
        ---------
        to_z : int
            proton number of the destination element for the Be fix (default 6,
            i.e. carbon)

        Raises
        ------
        AttributeError if called before _remap_abundances
        """
        try:
            self.abundances_interp
        except AttributeError:
            logger.critical("Abundances must be remapped before the Be fix can"
                            " be applied")
            raise AttributeError("no abundances_interp; call"
                                 " _remap_abundances first")

        logger.info("Total relative Be mass in model: {:e}\n".format(
            (self.abundances_interp[4] * self.dm_interp).sum() /
            self.dm_interp.sum()))

        self.abundances_interp[to_z] = (
            self.abundances_interp[to_z] + self.abundances_interp[4])
        self.abundances_interp[4] = np.zeros(self.N_interp)

    def _write_tardis_abundance_file(self, fname="tardis_abundances.dat"):
        """Write specific density file for Tardis

        Parameters
        ----------
        fname : str
            name of the Tardis specific structure file (default
            'tardis_abundances.dat')
        """
        with open(fname, "w") as f:
            f.write("# index Z=1 - Z={:d}\n".format(zmax))
            X = np.zeros((zmax+1, self.N_interp+1))

            X[0, :] = np.arange(self.N_interp+1)
            X[1:, 1:] = np.array(
                [self.abundances_interp[z] for z in xrange(1, zmax+1)])
            X[1:, 0] = X[1:, 1]

            np.savetxt(f, X.T, fmt=["% 4d"] + ["%.7e" for _ in xrange(1, zmax+1)])

    def _write_tardis_density_file(self, fname="tardis_densities.dat"):
        """Write specific abundance file for Tardis

        Parameters
        ----------
        fname : str
            name of the Tardis specific structure file (default
            'tardis_densities.dat')
        """
        with open(fname, "w") as f:
            f.write("{:f} {:s}\n".format(self.t.to("day").value, "day"))
            f.write("# index velocity (km/s) density (g/cm^3)\n")
            X = np.array([np.arange(self.N_interp+1),
                         np.insert(self.v_interp_r.to("km/s"), 0,
                                   self.v_interp_l.to("km/s")[0]).value,
                         np.insert(self.rho_interp.to("g/cm^3"), 0,
                                   self.rho_interp.to("g/cm^3")[0]).value]).T
            np.savetxt(f, X, fmt=["% 4d", "% 9.3f", "%.7e"])
