"""
Convert ASD to PSD and save as new files for our own convenience.

The noise curves are taken from LIGO Document T2000012-v2
"""

import numpy as np
import matplotlib.pyplot as plt

PSD_FILENAMES = ["AplusDesign.txt",
                 "avirgo_O5high_NEW.txt",
                #  "avirgo_O5low_NEW.txt",
                 "kagra_80Mpc.txt"]

NAMES = ["LIGO", 
         "VIRGO",
         "KAGRA"]

psd_dict = {}
for name, filename in zip(NAMES, PSD_FILENAMES):
    # Open it and read the data
    f, psd = np.loadtxt(filename, unpack=True)
    psd = psd ** 2
    psd_dict[name] = {"f": f, "psd": psd}
    
    # Save it to new file:
    new_filename = filename.replace(".txt", "_PSD.txt")
    np.savetxt(new_filename, np.array([f, psd]).T)
    
    # Test loading it again:
    test_f, test_psd = np.loadtxt(new_filename, unpack=True)
    assert np.allclose(f, test_f)
    assert np.allclose(psd, test_psd)
    
# Plot it:
plt.figure(figsize = (8, 6))
for name in NAMES:
    plt.loglog(psd_dict[name]["f"], psd_dict[name]["psd"], label=name)
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [Hz^-1]")
plt.legend()
plt.grid()
plt.savefig("./figures/check_psd.png", bbox_inches = "tight")
plt.close()