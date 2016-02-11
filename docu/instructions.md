# Instructions

in the following, brief instructions on how to use the tools of the repository
are provided.

## tardis_opacity.py

With ``tardis_opacity.py``, the opacity within a Tardis model may be examined in
detail. To determine the line opacity, the expansion opacity formalism is used.

The following figure shows the total opacity in the first and last radial shell
in the Tardis calculation using the ``tardis_example`` setup:

![image](images/tardis_opacity_example.png)

It has been produced with the following instructions (some plotting related
instructions are omitted):

```
import tardis
import yaml
import tardis_opacity as top

config = yaml.safe_load(open("tardis_example.yml", "r"))
mdl = tardis.run_tardis(config)

# initialise opacity calculator
opacity_diag = top.opacity_calculator(mdl)
# get total opacity on a standard frequency grid for each cell
chi_tot = opacity_diag.kappa_tot
# midpoints of the frequency grid:
nu_grid = 0.5 * (opacity_diag.nu_bins[1:] + opacity_diag.nu_bins[:-1])

plt.loglog(nu_grid, chi_tot[:, 0], label = "cell #0")
plt.loglog(nu_grid, chi_tot[:, -1], label = "cell #19")
```

With the opacity_calculator, the following quantities may also be calculated:

* pure bound-bound opacity per shell
* pure Thomson opacity per shell
* Planck-mean opacity per shell
* Planck-mean optical depth of each shell
* Planck-mean optical depth, integrated from the ejecta surface

## tardis_kromer_plot.py

With ``tardis_kromer_plot.py``, the importance of the different chemical
elements for the formation of the synthetic spectra may be illustrated. This
type of illustration has been developed by M. Kromer.

**Important:** The virtual packet logging capability must be active in order to
use this tool. Thus, Tardis must be compiled with the flag
``--with-vpacket-logging``.

The following figure shows such a "Kromer-type" plot for a Tardis calculation
using the ``tardis_example`` setup:

![image](images/kromer_plot_example.png)

It has been produced with the following instructions:

```
import tardis
import yaml
import tardis_kromer_plot as tkp

config = yaml.safe_load(open("tardis_example.yml", "r"))
mdl = tardis.run_tardis(config)

# initialise the plotting tool
plotter = tkp.tardis_kromer_plotter(mdl)
# generate plot
plotter.generate_plot(xlim=(3000,1e4), twinx=True)
```
