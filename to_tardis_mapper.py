# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
#  File Name : to_tardis_mapper.py
#
#  Purpose :
#
#  Creation Date : 29-02-2016
#
#  Last Modified : Tue 01 Mar 2016 18:58:39 CET
#
#  Created By :
#
# _._._._._._._._._._._._._._._._._._._._._.
import numpy as np
import astropy.units as units
from pyne import nucname, material

# maximum atomic number
zmax = 30


class original_model(object):
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
        if self._stable_abundances is None:
            tmp = {}

            for i in xrange(zmax):
                z = i+1
                tmp[z] = np.zeros(self.nzones)
            self._stable_abundances = tmp

        return self._stable_abundances

    @property
    def radio_abundances(self):
        if self._radio_abundances is None:
            self._radio_abundances = {}
        return self._radio_abundances

    def read_w7_density(self, fname):

        f = open(fname, "r")

        # read header
        buffer = f.readline().rsplit()
        self.t= float(buffer[2]) * units.s
        buffer = f.readline()

        # read main data block
        data = np.loadtxt(f)
        f.close()

        self.ro = data[:, 2] * units.cm
        self.ri = np.insert(self.ro, 0, 0 * units.cm)[:-1]
        self.rho = data[:, 3] * units.g / units.cm**3

    def read_w7_abundances(self, fname):

        f = open(fname, "r")
        data = np.loadtxt(f, skiprows=1)
        f.close()

        for i in xrange(zmax):
            self.stable_abundances[i+1] = data[:, i]

        self.radio_abundances["ni56"] = data[:, 32]
        self.radio_abundances["co56"] = data[:, 33]
        self.radio_abundances["ni57"] = data[:, 34]


class to_tardis_mapper(object):
    def __init__(self, orig_model):

        self.orig = orig_model

    def remap(self, v, t, decay=True, write_density=True,
              density_fname="tardis_density.dat", write_abundances=True,
              abundance_fname="tardis_abundances.dat", be_fix=True, to_z=6):

        self._remap_density(v, t)
        self._remap_abundances(t)
        if decay:
            self._decay_abundances(t)
        if be_fix:
            self._be_fix(to_z=to_z)
        if write_density:
            self._write_tardis_density_file(fname=density_fname)
        if write_abundances:
            self._write_tardis_abundance_file(fname=abundance_fname)

    def _remap_density(self, v, t):

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

    def _remap_abundances(self, t):
        self.abundances_interp = {}
        self.radio_abundances_interp = {}

        def remap_species(vrorig, Xorig):
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

        for z in xrange(1, zmax+1):

            vrorig = self.orig.vo.to("cm/s")
            Xorig = self.orig.stable_abundances[z]

            X_interp = remap_species(vrorig, Xorig)

            self.abundances_interp[z] = X_interp

        for ident in self.orig.radio_abundances.keys():

            vrorig = self.orig.vo.to("cm/s")
            Xorig = self.orig.radio_abundances[ident]

            X_interp = remap_species(vrorig, Xorig)

            self.radio_abundances_interp[ident] = X_interp

    def _decay_abundances(self, t):

        for i in xrange(self.N_interp):
            comp = {}
            mass = 0
            for ident in self.radio_abundances_interp.keys():
                Xi = self.radio_abundances_interp[ident][i]
                mass += Xi
                comp[nucname.id(ident)] = Xi
            inp = material.Material(comp, mass=mass)
            res = inp.decay(t.to("s").value).mult_by_mass()

            for item in res.items():
                z = nucname.znum(item[0])
                self.abundances_interp[z][i] = \
                    self.abundances_interp[z][i] + item[1]

    def _be_fix(self, to_z=6):

        print("Total relative Be mass in model: {:e}\n".format(
            (self.abundances_interp[4] * self.dm_interp).sum() /
            self.dm_interp.sum()))

        self.abundances_interp[to_z] = (
            self.abundances_interp[to_z] + self.abundances_interp[4])
        self.abundances_interp[4] = np.zeros(self.N_interp)

    def _write_tardis_abundance_file(self, fname="tardis_abundances.dat"):
        f = open(fname, "w")
        f.write("# index Z=1 - Z={:d}\n".format(zmax))
        X = np.zeros((zmax+1, self.N_interp+1))

        X[0, :] = np.arange(self.N_interp+1)
        X[1:, 1:] = np.array([self.abundances_interp[z] for z in xrange(1, zmax+1)])
        X[1:, 0] = X[1:, 1]

        np.savetxt(f, X.T, fmt=["% 4d"] + ["%.7e" for i in xrange(1, zmax+1)])
        f.close()

    def _write_tardis_density_file(self, fname="tardis_density.dat"):

        f = open(fname, "w")

        f.write("{:f} {:s}\n".format(self.t.to("day").value, "day"))
        f.write("# index velocity (km/s) density (g/cm^3)\n")
        X = np.array([np.arange(self.N_interp+1),
                     np.insert(self.v_interp_r.to("km/s"), 0,
                               self.v_interp_l.to("km/s")[0]).value,
                     np.insert(self.rho_interp.to("g/cm^3"), 0,
                               self.rho_interp.to("g/cm^3")[0]).value]).T
        np.savetxt(f, X, fmt=["% 4d", "% 9.3f", "%.7e"])

        f.close()
