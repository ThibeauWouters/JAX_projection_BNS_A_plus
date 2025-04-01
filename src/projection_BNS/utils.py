import os
import copy
import numpy as np 

c = 299_792.458 # km/s
H0 = 67.47 # km/s/Mpc

# Get the location of this file:
this_file_location = os.path.abspath(__file__)
this_dir = os.path.dirname(this_file_location)

GW_PATH = os.path.join(this_dir, "GW")
NF_PATH = os.path.join(this_dir, "NF")
EOS_PATH = os.path.join(this_dir, "EOS")
DATA_PATH = os.path.join(this_dir, "data")

HQC18_EOS_FILENAME = os.path.join(DATA_PATH, "HQC18.npz")
SLY230A_EOS_FILENAME = os.path.join(DATA_PATH, "SLY230A.npz")
MPA1_EOS_FILENAME = os.path.join(DATA_PATH, "MPA1.npz")

EOS_FILENAMES_DICT = {"HQC18": HQC18_EOS_FILENAME,
                      "SLY230A": SLY230A_EOS_FILENAME,
                      "MPA1": MPA1_EOS_FILENAME}

# These are the colors of the jax logo (order J, A, X)
TARGET_COLORS_DICT = {"HQC18": "#5e97f6", 
                      "SLY230A": "#26a69a",
                      "MPA1": "#9c27b0"}

def get_eos_name_from_dirname(dirname: str):
    if "HQC18" in dirname:
        eos_name = "HQC18"
    elif "SLY230A" in dirname:
        eos_name = "SLY230A"
    elif "MPA1" in dirname:
        eos_name = "MPA1"
    elif "jester_soft" in dirname:
        eos_name = "jester_soft"
    elif "jester_middle" in dirname:
        eos_name = "jester_middle"
    elif "jester_hard" in dirname:
        eos_name = "jester_middle"
    else:
        raise ValueError("EOS name not found for target directory {}".format(dirname))
    return eos_name

def get_eos_file_from_dirname(dirname: str):
    eos_name = get_eos_name_from_dirname(dirname)
    return EOS_FILENAMES_DICT[eos_name]

def load_eos(eos_filename):
    eos = np.load(eos_filename)
    m, r, l = eos["masses_EOS"], eos["radii_EOS"], eos["Lambdas_EOS"]
    return m, r, l

def distance_to_redshift(dL: float):
    z = (H0/c) * dL
    return z

def redshift_to_distance(z: float):
    dL = (c/H0) * z
    return dL