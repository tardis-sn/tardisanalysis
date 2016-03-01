# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
#  File Name : to_tardis_mapper.py
#
#  Purpose :
#
#  Creation Date : 29-02-2016
#
#  Last Modified : Tue 01 Mar 2016 13:58:41 CET
#
#  Created By :
#
# _._._._._._._._._._._._._._._._._._._._._.
import numpy as np
import astropy.units as units
from pyne import nucname, material

zmax = 30


class original_model(object):
    def __init__(self):

        self.stable_abundances = {}
        self.radio_abundances = {}

    def read_w7_density(self, fname):

        f = open(fname, "r")
        buffer = f.readline().rsplit()
        self.nzones = int(buffer[0])
        self.time = float(buffer[2]) * units.s
        buffer = f.readline()

        data = np.loadtxt(f)
        f.close()
        self.rr = data[:, 2] * units.cm
        self.rl = np.insert(self.rr, 0, 0 * units.cm)[:-1]
        self.rho = data[:, 3] * units.g / units.cm**3

        self.vr = self.rr / self.time
        self.vl = self.rl / self.time

        self.dm = 4. * np.pi / 3. * (self.rr**3 - self.rl**3) * self.rho
        self.mr = np.zeros(self.nzones) * units.g
        self.mr[0] = self.dm[0]
        for i in xrange(1, self.nzones):
            self.mr[i] = self.mr[i-1] + self.dm[i]

    def read_w7_abundances(self, fname):

        f = open(fname, "r")
        data = np.loadtxt(f, skiprows=1)
        f.close()

        for i in xrange(32):
            self.stable_abundances[i+1] = data[:, i]

        self.radio_abundances["ni56"] = data[:, 32]
        self.radio_abundances["co56"] = data[:, 33]
        self.radio_abundances["ni57"] = data[:, 34]


class to_tardis_mapper(object):
    def __init__(self, orig_model):

        self.orig = orig_model

    def remap_density(self, v, t):

        self.t = t
        self.N_interp = len(v) - 1
        self.v_interp_r = v[1:].to("cm/s")
        self.v_interp_l = v[:-1].to("cm/s")

        V_interp = 4. / 3. * np.pi * ((self.v_interp_r * t).to("cm")**3 -
                                      (self.v_interp_l * t).to("cm")**3)

        vrorig = self.orig.vr.to("cm/s")
        mrorig = self.orig.mr.to("solMass")

        mr_interp = np.interp(
            np.insert(self.v_interp_r, 0, self.v_interp_l[0])**3,
            np.append(0 * vrorig.unit, vrorig)**3,
            np.append(0 * mrorig.unit, mrorig)) * mrorig.unit

        self.dm_interp = (mr_interp[1:] - mr_interp[:-1])
        self.rho_interp = (self.dm_interp / V_interp).to("g/cm^3")

    def remap_abundances(self, v, t):
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

            vrorig = self.orig.vr.to("cm/s")
            Xorig = self.orig.stable_abundances[z]

            X_interp = remap_species(vrorig, Xorig)

            self.abundances_interp[z] = X_interp

        for ident in self.orig.radio_abundances.keys():

            vrorig = self.orig.vr.to("cm/s")
            Xorig = self.orig.radio_abundances[ident]

            X_interp = remap_species(vrorig, Xorig)

            self.radio_abundances_interp[ident] = X_interp

    def decay_abundances(self, v, t):

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

    def write_tardis_abundance_file(self, fname="tardis_abundances.dat"):
        f = open(fname, "w")
        f.write("# index Z=1 - Z={:d}\n".format(zmax))
        X = np.zeros((zmax+1, self.N_interp+1))

        X[0, :] = np.arange(self.N_interp+1)
        X[1:, 1:] = np.array([self.abundances_interp[z] for z in xrange(1, zmax+1)])
        X[1:, 0] = X[1:, 1]

        np.savetxt(f, X.T, fmt=["% 4d"] + ["%.7e" for i in xrange(1, zmax+1)])
        f.close()

    def write_tardis_density_file(self, fname="tardis_density.dat"):

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
