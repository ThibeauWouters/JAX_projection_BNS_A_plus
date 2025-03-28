"""
We have downloaded the EOS files from the bilby github repo (see, e.g., https://github.com/bilby-dev/bilby/tree/main/bilby/gw/eos/eos_tables)

Now, check them out with the bilby functionalities.

Note to self: use the bilby_pipe conda environment for the transformation. This does not give a scipy import error.
"""

import numpy as np
import matplotlib.pyplot as plt
import lalsimulation as lalsim

M_SUN = 1.988409870698050731911960804878414216e30 # LAL_MSUN_SI
G = 6.67430e-11
c = 299792458e0

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
    
    eos = lalsim.SimNeutronStarEOSByName(eos_name)
    family = lalsim.CreateSimNeutronStarFamily(eos)
    
    # Construct the array of masses
    m_max = lalsim.SimNeutronStarMaximumMass(family)
    print(f"M_Max is {m_max / M_SUN}")
    masses_array = np.linspace(0.25 * M_SUN, m_max, 1_000)
    
    # Calculate the radius and the Love number
    radius = []
    k2 = []
    for m in masses_array:
        radius.append(lalsim.SimNeutronStarRadius(m, family))
        k2.append(lalsim.SimNeutronStarLoveNumberK2(m, family))
    radius = np.array(radius)
    k2 = np.array(k2)
    
    # Calculate the tidal deformability -- note the annoying units
    m_meter = masses_array * G / c / c
    lambdas = [2. / 3. * k2_ * r ** 5. / m ** 5. for k2_, r, m in zip(k2, radius, m_meter)]
    
    # Convert units
    masses_array /= M_SUN
    radius /= 1e3 # convert to km
    lambdas = np.array(lambdas)
    
    # Save it as npz file
    np.savez(f"{eos_name}.npz", masses_EOS=masses_array, radii_EOS=radius, Lambdas_EOS=lambdas)
    
    # For plotting, mask to make the plot more readable
    mask = masses_array > 0.99
    plt.subplot(121)
    plt.plot(radius[mask], masses_array[mask], label=eos_name, color=col)
    
    plt.subplot(122)
    plt.plot(masses_array[mask], lambdas[mask], label=eos_name, color=col)

plt.subplot(121)
plt.xlabel(r"$R$ [km]")
plt.ylabel(r"$M$ [M$_{\odot}$]")

plt.subplot(122)
plt.xlabel(r"$M$ [M$_{\odot}$]")
plt.ylabel(r"$\Lambda$")
plt.yscale("log")
plt.legend()
    
plt.savefig("./figures/check_eos.pdf", bbox_inches = "tight")
plt.close()
    