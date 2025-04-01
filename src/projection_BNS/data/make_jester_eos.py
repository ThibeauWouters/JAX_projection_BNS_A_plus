"""
Create an EOS file that is made from the Jester code so that we ensure the same parametrization
This code is inspired by EOS/inference.py, using the transform as defined there to create the target.
"""

import os 
import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

from jimgw.prior import UniformPrior, CombinePrior
import projection_BNS.EOS.inference_utils as inference_utils
import projection_BNS.utils as utils


def make_targets():
    
    NB_CSE = 8
    NMAX_NSAT = 10
    
    E_sym_prior = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
    L_sym_prior = UniformPrior(10.0, 120.0, parameter_names=["L_sym"]) # note: upper bound is smaller than jester paper
    K_sym_prior = UniformPrior(-300.0, 100.0, parameter_names=["K_sym"])
    Q_sym_prior = UniformPrior(-800.0, 800.0, parameter_names=["Q_sym"])
    Z_sym_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sym"])

    K_sat_prior = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
    Q_sat_prior = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
    Z_sat_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])

    prior_list = [
        E_sym_prior,
        L_sym_prior, 
        K_sym_prior,

        K_sat_prior,
    ]

    ### CSE priors
    print(f"Using CSE grid with {NB_CSE} points")
    nbreak_prior = UniformPrior(1.0 * 0.16, 2.0 * 0.16, parameter_names=[f"nbreak"])
    prior_list.append(nbreak_prior)
    for i in range(NB_CSE):
        # NOTE: the density parameters are sampled from U[0, 1], so we need to scale it, but it depends on break so will be done internally
        prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"n_CSE_{i}_u"]))
        prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"]))

    # Final point to end
    prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{NB_CSE}"]))

    # Construct the EOS prior and a transform here which can be used down below for creating the EOS plots after inference is completed
    eos_prior = CombinePrior(prior_list)
    eos_param_names = eos_prior.parameter_names
    all_output_keys = ["logpc_EOS", "masses_EOS", "radii_EOS", "Lambdas_EOS", "n", "p", "e","cs2"]
    all_output_keys += eos_param_names
    name_mapping = (eos_param_names, all_output_keys)

    # This transform will be the same as my_transform, but with different output keys, namely, all EOS related quantities, for postprocessing
    my_transform_eos = inference_utils.MicroToMacroTransform(name_mapping,
                                                            nmax_nsat=NMAX_NSAT,
                                                            nb_CSE=NB_CSE
                                                            )

    names = ["jester_soft", "jester_middle", "jester_hard"]
    def check_valid(target: dict, name: str):
        m, r, l = target["masses_EOS"], target["radii_EOS"], target["Lambdas_EOS"]
        mtov_valid = np.max(m) > 2.0
        if not mtov_valid:
            return False
        
        r14 = np.interp(1.4, m, r)
        print(r14)
        if name == "jester_soft":
            return r14 < 11.5
        elif name == "jester_middle":
            return (r14 > 12.0) and (r14 < 12.50)
        elif name == "jester_hard":
            return (r14 > 13.00) * (r14 < 13.5)
        else:
            raise ValueError("Unknown name")

    key = jax.random.PRNGKey(1)
    for name in names:
        print(f"Creating target {name}")
        
        # Sample until we have a valid EOS
        valid = False
        counter = 0
        while not valid and counter < 1_000:
            key, subkey = jax.random.split(key)
            target_param = eos_prior.sample(subkey, 1)
            target_param = {k: float(v.at[0].get()) for k, v in target_param.items()}
            
            target = my_transform_eos.forward(target_param)
            valid = check_valid(target, name)
            counter += 1
        
        if not valid:
            raise ValueError("Could not create a valid EOS")
        
        print(target.keys())
        
        eos_filename = os.path.join(utils.DATA_PATH, "eos", f"{name}.npz")
        np.savez(eos_filename, **target)
        print(f"Saved target to {eos_filename}")
    

def plot_targets():
    # Make a plot of the MR and ML
    names = ["jester_soft", "jester_middle", "jester_hard"]
    plt.subplots(figsize = (12, 8), nrows = 1, ncols = 2)
    labels = ["Soft", "Middle", "Hard"]
    for name, label in zip(names, labels):
        eos_filename = os.path.join(utils.DATA_PATH, "eos", f"{name}.npz")
        eos = np.load(eos_filename)
        m, r, l = eos["masses_EOS"], eos["radii_EOS"], eos["Lambdas_EOS"]
        
        mask = m > 0.75
        m, r, l = m[mask], r[mask], l[mask]
        
        plt.subplot(1, 2, 1)
        plt.plot(r, m, label=label)
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M$ [$M_\odot$]")
        
        plt.subplot(1, 2, 2)
        plt.plot(m, l, label=label)
        plt.xlabel(r"$M$ [$M_\odot$]")
        plt.ylabel(r"$\Lambda$")
        plt.yscale("log")
        plt.legend()
        
    plt.savefig("./figures/jester_eos.pdf", bbox_inches = "tight")
    plt.close()
    
if __name__ == "__main__":
    make_targets()
    plot_targets()