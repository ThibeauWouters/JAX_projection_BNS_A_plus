"""
EOS inference with trained flows from GW posteriors
"""

################
### PREAMBLE ###
################
import os 
import time
import shutil
import numpy as np
np.random.seed(43) # for reproducibility
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jimgw.prior import UniformPrior, CombinePrior
from jimgw.jim import Jim
import argparse

print(f"GPU found?")
print(jax.devices())

import projection_BNS.EOS.inference_utils as utils

################
### Argparse ###
################

def parse_arguments():
    parser = argparse.ArgumentParser(description="Full-scale inference script with customizable options.")
    parser.add_argument("--eos", 
                        type=str, 
                        help="Name of the EOS. Choose from [HQC18, MPA1, SLY230A].")
    parser.add_argument("--ifo-network", 
                        type=str, 
                        help="Name of the network of detectors. Choose from [Aplus, Asharp, ET].")
    parser.add_argument("--id-list", 
                        type=str, 
                        nargs='+',
                        default=None,
                        help="List of identifier of the GW injection for that EOS.")
    parser.add_argument("--id-begin", 
                        type=int, 
                        default=1,
                        help="Starting point of indices")
    parser.add_argument("--id-end", 
                        type=int, 
                        default=30,
                        help="Ending point of indices")
    parser.add_argument("--N-masses-evaluation", 
                        type=int, 
                        default=1,
                        help="This is the N_masses_evaluation argument passed on to the GW likelihood function, which essentially determines how many samples are used to marginalize over the masses.")
    parser.add_argument("--local-sampler-name", 
                        type=str, 
                        default="MALA", 
                        help="Name of the local sampler to use. Choose from [MALA, GaussianRandomWalk].")
    parser.add_argument("--ignore-Q-Z", 
                        type=bool,
                        default=True, 
                        help="Whether to sample the higher-order metamodel parameters Q_sym, Q_sat, Z_sym, Z_sat. Recommended to set to True due to the MM degeneracy (see Jester paper).")
    parser.add_argument("--sample-radio", 
                        type=bool, 
                        default=False,
                        help="Whether to sample the radio timing mass measurement pulsars. Do all of them at once.")
    parser.add_argument("--sample-chiEFT", 
                        type=bool, 
                        default=False, 
                        help="Whether to sample chiEFT data")
    parser.add_argument("--use-zero-likelihood", 
                        type=bool, 
                        default=False, 
                        help="Whether to use a mock log-likelihood which constantly returns 0")
    parser.add_argument("--outdir",
                        type=str, 
                        default="./outdir/", 
                        help="Directory to save output files (default: './outdir/')")
    parser.add_argument("--N-samples-EOS", 
                        type=int, 
                        default=10_000,
                        help="Number of samples for which the TOV equations are solved at the end of the script to output the final EOS samples and their NS families.")
    parser.add_argument("--nb-cse", 
                        type=int, 
                        default=8, 
                        help="Number of CSE grid points (excluding the last one at the end, since its density value is fixed, we do add the cs2 prior separately.)")
    parser.add_argument("--sampling-seed", 
                        type=int, 
                        default=11,
                        help="Number of CSE grid points (excluding the last one at the end, since its density value is fixed, we do add the cs2 prior separately.)")
    ### flowMC/Jim hyperparameters
    parser.add_argument("--n-loop-training", 
                        type=int, 
                        default=20,
                        help="Number of flowMC training loops.)")
    parser.add_argument("--n-loop-production", 
                        type=int, 
                        default=20,
                        help="Number of flowMC production loops.)")
    parser.add_argument("--eps-mass-matrix", 
                        type=float, 
                        default=1e-3,
                        help="Overall scaling factor for the step size matrix for MALA.")
    parser.add_argument("--n-local-steps", 
                        type=int, 
                        default=2,
                        help="Number of local steps to perform.")
    parser.add_argument("--n-global-steps", 
                        type=int, 
                        default=100,
                        help="Number of global steps to perform.")
    parser.add_argument("--n-epochs", 
                        type=int, 
                        default=20,
                        help="Number of epochs for NF training.")
    parser.add_argument("--n-chains", 
                        type=int, 
                        default=1000,
                        help="Number of MCMC chains to evolve.")
    parser.add_argument("--train-thinning", 
                        type=int, 
                        default=1,
                        help="Thinning factor before feeding samples to NF for training.")
    parser.add_argument("--output-thinning", 
                        type=int, 
                        default=5,
                        help="Thinning factor before saving samples.")
    return parser.parse_args()

def main(args):
    
    NMAX_NSAT = 10
    NB_CSE = args.nb_cse

    E_sym_prior = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
    L_sym_prior = UniformPrior(10.0, 120.0, parameter_names=["L_sym"]) # note: upper bound is smaller than jester paper
    K_sym_prior = UniformPrior(-300.0, 100.0, parameter_names=["K_sym"])
    Q_sym_prior = UniformPrior(-800.0, 800.0, parameter_names=["Q_sym"])
    Z_sym_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sym"])
    
    K_sat_prior = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
    Q_sat_prior = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
    Z_sat_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])

    if args.ignore_Q_Z:
        print(f"Ignoring the Q and Z NEP parameters")
        prior_list = [
            E_sym_prior,
            L_sym_prior, 
            K_sym_prior,

            K_sat_prior,
        ]
    else:
        prior_list = [
            E_sym_prior,
            L_sym_prior, 
            K_sym_prior,
            Q_sym_prior,
            Z_sym_prior,

            K_sat_prior,
            Q_sat_prior,
            Z_sat_prior,
    ]

    ### CSE priors
    if NB_CSE > 0:
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
    name_mapping = (eos_param_names, all_output_keys)
    
    # This transform will be the same as my_transform, but with different output keys, namely, all EOS related quantities, for postprocessing
    if args.nb_cse > 0:
        keep_names = ["E_sym", "L_sym", "nbreak"]
    else:
        keep_names = ["E_sym", "L_sym"]
    my_transform_eos = utils.MicroToMacroTransform(name_mapping,
                                                   keep_names=keep_names,
                                                   nmax_nsat=NMAX_NSAT,
                                                   nb_CSE=NB_CSE
                                                )
    
    # Create the output directory if it does not exist
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    # Copy this script to the output directory, for reproducibility later on
    shutil.copy(__file__, os.path.join(outdir, "backup_inference.py"))
    
    ##################
    ### LIKELIHOOD ###
    ##################

    # Likelihood: choose which PSR(s) to perform inference on:
    if args.use_zero_likelihood:
        print("Using the zero likelihood:")
        likelihood = utils.ZeroLikelihood(my_transform)
    
    else:
        # Likelihoods from GW posteriors
        if args.id_list is None:
            id_list = np.arange(args.id_begin, args.id_end + 1)
            print(f"Given id_list was None, so created id list {id_list}")
        else:
            id_list = args.id_list
        
        likelihoods_list_GW = []
        for idx in id_list:
            try:
                new_likelihood = utils.GWlikelihood_with_masses(args.eos, 
                                                                args.ifo_network, 
                                                                idx, 
                                                                N_masses_evaluation=args.N_masses_evaluation)
                likelihoods_list_GW.append(new_likelihood)
            except Exception as e:
                print(f"Could not load the likelihood for id {idx}, because of the following error: {e}")
                print(f"Moving on")
                
        print(f"There are {len(likelihoods_list_GW)} GW likelihoods used now")
        keep_names += ["key"]
        
        # Radio timing mass measurement pulsars
        likelihoods_list_radio = []
        if args.sample_radio:
            print(f"We are also sampling the radio timing mass measurement pulsars for MTOV constraints")
            likelihoods_list_radio += [utils.RadioTimingLikelihood("J1614", 1.94, 0.06)]
            likelihoods_list_radio += [utils.RadioTimingLikelihood("J0348", 2.01, 0.08)]
            likelihoods_list_radio += [utils.RadioTimingLikelihood("J0740", 2.08, 0.14)]

        # Chiral EFT
        likelihoods_list_chiEFT = []
        if args.sample_chiEFT and args.nb_cse > 0:
            # FIXME: decide whether to remove this permanently
            # keep_names += ["nbreak"]
            # print(f"Loading data necessary for the Chiral EFT")
            # likelihoods_list_chiEFT += [utils.ChiEFTLikelihood()]
            
            raise ValueError("Chiral EFT likelihood is no longer supported in this project (at least for now)")

        # Total likelihoods list:
        likelihoods_list = likelihoods_list_GW + likelihoods_list_radio + likelihoods_list_chiEFT
        for l in likelihoods_list:
            print(l)
            
        # Combine into a full likelihood
        if len(likelihoods_list) > 1:
            print(f"Combining likelihoods into one final likelihood . . .")
            likelihood = utils.CombinedLikelihood(likelihoods_list)
        else:
            print(f"There is only one likelihood so we will not combine them")
            likelihood = likelihoods_list[0]
        
    # Construct the transform object
    TOV_output_keys = ["masses_EOS", "Lambdas_EOS"]
    prior_keys = [p.parameter_names[0] for p in prior_list]
    print("prior_keys")
    print(prior_keys)
    
    # full_prior_list = prior_list + mass_priors # TODO: remove me, this is for the old implementation with the masses
    key_prior = utils.KeyPrior()
    full_prior_list = prior_list + [key_prior]
    prior = CombinePrior(full_prior_list)
    all_prior_keys = prior.parameter_names
    
    print("all_prior_keys")
    print(all_prior_keys)
    
    for i in range(len(prior.parameter_names)):
        print(f"Prior parameter {i}: {prior.parameter_names[i]}")
    sampled_param_names = prior.parameter_names
    name_mapping = (sampled_param_names, TOV_output_keys)
    my_transform = utils.MicroToMacroTransform(name_mapping,
                                               keep_names = keep_names,
                                               nmax_nsat = NMAX_NSAT,
                                               nb_CSE = NB_CSE,
                                               )
    
    mass_matrix = jnp.eye(prior.n_dim)
    local_sampler_arg = {"step_size": mass_matrix * args.eps_mass_matrix}
    kwargs = {"n_loop_training": args.n_loop_training,
            "n_loop_production": args.n_loop_production,
            "n_chains": args.n_chains,
            "n_local_steps": args.n_local_steps,
            "n_global_steps": args.n_global_steps,
            "n_epochs": args.n_epochs,
            "train_thinning": args.train_thinning,
            "output_thinning": args.output_thinning,
    }
    
    print("We are going to give these kwargs to Jim:")
    print(kwargs)
    
    print("We are going to sample the following parameters:")
    print(prior.parameter_names)
    
    # Pass on the local sampler to Jim
    kwargs["local_sampler_name"] = args.local_sampler_name

    # Define the Jim object here
    jim = Jim(likelihood,
              prior,
              local_sampler_arg = local_sampler_arg,
              likelihood_transforms = [my_transform],
              **kwargs)

    # Test case
    samples = prior.sample(jax.random.PRNGKey(0), 3)
    samples_transformed = jax.vmap(my_transform.forward)(samples)
    log_prob = jax.vmap(likelihood.evaluate)(samples_transformed, {})
    
    print("log_prob")
    print(log_prob)
    
    # Do the sampling
    print(f"Sampling seed is set to: {args.sampling_seed}")
    start = time.time()
    jim.sample(jax.random.PRNGKey(args.sampling_seed))
    jim.print_summary()
    end = time.time()
    runtime = end - start

    print(f"Sampling has been successful, now we will do some postprocessing. Sampling time: roughly {int(runtime / 60)} mins")

    ### POSTPROCESSING ###
        
    # Training (just to count number of samples)
    sampler_state = jim.sampler.get_sampler_state(training=True)
    log_prob = sampler_state["log_prob"].flatten()
    nb_samples_training = len(log_prob)

    # Production (also for postprocessing plotting)
    sampler_state = jim.sampler.get_sampler_state(training=False)

    # Get the samples, and also get them as a dictionary
    samples_named = jim.get_samples()
    samples_named_for_saving = {k: np.array(v) for k, v in samples_named.items()}
    samples_named = {k: np.array(v).flatten() for k, v in samples_named.items()}
    keys, samples = list(samples_named.keys()), np.array(list(samples_named.values()))

    # Get the log prob, also count number of samples from it
    log_prob = np.array(sampler_state["log_prob"])
    log_prob = log_prob.flatten()
    nb_samples_production = len(log_prob)
    total_nb_samples = nb_samples_training + nb_samples_production
    
    # Save the final results
    print(f"Saving the final results")
    np.savez(os.path.join(outdir, "results_production.npz"), log_prob=log_prob, **samples_named_for_saving)

    print(f"Number of samples generated in training: {nb_samples_training}")
    print(f"Number of samples generated in production: {nb_samples_production}")
    print(f"Number of samples generated: {total_nb_samples}")
    
    # Save the runtime to a file as well
    with open(os.path.join(outdir, "runtime.txt"), "w") as f:
        f.write(f"{runtime}")

    # Generate the final EOS + TOV samples from the EOS parameter samples
    idx_1 = np.random.choice(np.arange(len(log_prob)), size=args.N_samples_EOS, replace=False)
    idx_2 = np.random.choice(np.arange(len(log_prob)), size=args.N_samples_EOS, replace=False)
    
    chosen_samples_test = {k: jnp.array(v[idx_1]) for k, v in samples_named.items()}
    chosen_samples = {k: jnp.array(v[idx_2]) for k, v in samples_named.items()}
    # NOTE: jax lax map helps us deal with batching, but a batch size multiple of 10 gives errors, therefore this weird number
    # transformed_samples = jax.lax.map(jax.jit(my_transform_eos.forward), chosen_samples, batch_size = 4_999)
    
    # First do a single batch to jit compile, then do compiled vmap to get the timing right
    my_forward = jax.jit(my_transform_eos.forward)
    transformed_samples_test = jax.vmap(my_forward)(chosen_samples_test)
    
    TOV_start = time.time()
    transformed_samples = jax.vmap(my_forward)(chosen_samples)
    TOV_end = time.time()
    print(f"Time taken for TOV map: {TOV_end - TOV_start} s")
    chosen_samples.update(transformed_samples)

    log_prob = log_prob[idx_2]
    np.savez(os.path.join(args.outdir, "eos_samples.npz"), log_prob=log_prob, **chosen_samples)
    
    samples_for_corner = {k: v.flatten() for k, v in chosen_samples.items()}
    samples_for_corner = {k: v for k, v in samples_for_corner.items() if k in prior_keys}
    keys_to_plot = list(samples_for_corner.keys())
    print(f"The corner plot will plot the parameters: {keys_to_plot}")
    samples_for_corner_values = np.array(list(samples_for_corner.values())).T
    
    print("DONE entire script")
    
if __name__ == "__main__":
    args = parse_arguments()  # Get command-line arguments
    main(args)