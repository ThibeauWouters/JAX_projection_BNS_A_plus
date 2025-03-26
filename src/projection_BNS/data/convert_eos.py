"""
We have downloaded the EOS files from the bilby github repo (see, e.g., https://github.com/bilby-dev/bilby/tree/main/bilby/gw/eos/eos_tables)

Now, check them out with the bilby functionalities.

Note to self: use the bilby_pipe conda environment for the transformation. This does not give a scipy import error.
"""

import numpy as np
import matplotlib.pyplot as plt
import bilby

from bilby.gw.eos import TabularEOS, EOSFamily
from bilby.gw.eos.eos import conversion_dict
from bilby.gw.eos.tov_solver import IntegrateTOV

params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

EOS_NAMES = ["HQC18" ,"SLY230A", "MPA1"]
COLORS = ["b", "g", "r"]

eos_dict = {}

plt.subplots(nrows = 1, ncols = 2, figsize=(12, 6))
for col, eos_name in zip(COLORS, EOS_NAMES):
    print(f"Checking out the EOS and NS for {eos_name}")
    
    # Load the EOS and get the NS family
    tabular_eos = TabularEOS(eos_name)
    family = EOSFamily(tabular_eos, npts=1_000)
    
    mass = np.array(family.mass) * conversion_dict["mass"]["m_sol"]
    radius = np.array(family.radius) * conversion_dict["radius"]["km"]
    tidal_deformability = np.array(family.tidal_deformability)
    
    eos_dict[eos_name] = {"mass_EOS": mass, 
                          "radii_EOS": radius,
                          "Lambdas_EOS": tidal_deformability}
    
    # Save it as npz file
    np.savez(f"{eos_name}.npz", masses_EOS=mass, radii_EOS=radius, Lambdas_EOS=tidal_deformability)
    
    # For plotting, mask to make the plot more readable
    
    mask = mass > 0.99
    plt.subplot(121)
    plt.plot(radius[mask], mass[mask], label=eos_name, color=col)
    
    plt.subplot(122)
    plt.plot(mass[mask], tidal_deformability[mask], label=eos_name, color=col)

plt.subplot(121)
plt.xlabel(r"$R$ [km]")
plt.ylabel(r"$M$ [M$_{\odot}$]")

plt.subplot(122)
plt.xlabel(r"$M$ [M$_{\odot}$]")
plt.ylabel(r"$\Lambda$")
plt.yscale("log")
plt.legend()
    
plt.savefig("./figures/check_eos.png", bbox_inches = "tight")
plt.close()
    