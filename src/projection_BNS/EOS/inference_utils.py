import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Float
import copy
import os
import json
import arviz

from jimgw.base import LikelihoodBase
from jimgw.transforms import NtoMTransform

import equinox as eqx
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.distributions import Normal, Transformed

from joseTOV.eos import MetaModel_with_CSE_EOS_model, MetaModel_EOS_model, construct_family
import joseTOV.utils as jose_utils

from projection_BNS.NF.NFTrainer import get_source_masses, make_flow, NF_PATH

NEP_CONSTANTS_DICT = {
    "E_sat": -16.0 # TODO: this is it, right?
}

##################
### TRANSFORMS ###
##################

class MicroToMacroTransform(NtoMTransform):
    
    def __init__(self,
                 name_mapping: tuple[list[str], list[str]],
                 keep_names: list[str] = None,
                 # metamodel kwargs:
                 ndat_metamodel: int = 100,
                 # CSE kwargs
                 nmax_nsat: float = 25,
                 nb_CSE: int = 8,
                 # TOV kwargs
                 min_nsat_TOV: float = 0.75,
                 ndat_TOV: int = 100,
                 ndat_CSE: int = 100,
                 nb_masses: int = 100,
                 fixed_params: dict[str, float] = None,
                ):
    
        # By default, keep all names
        if keep_names is None:
            keep_names = name_mapping[0]
        super().__init__(name_mapping, keep_names=keep_names)
    
        # Save as attributes
        self.ndat_metamodel = ndat_metamodel
        self.nmax_nsat = nmax_nsat
        self.nmax = nmax_nsat * 0.16
        self.nb_CSE = nb_CSE
        self.min_nsat_TOV = min_nsat_TOV
        self.ndat_TOV = ndat_TOV
        self.ndat_CSE = ndat_CSE
        self.nb_masses = nb_masses
        
        # Create the EOS object -- there are several choices for the parametrizations
        if nb_CSE > 0:
            eos = MetaModel_with_CSE_EOS_model(nmax_nsat=self.nmax_nsat,
                                               ndat_metamodel=self.ndat_metamodel,
                                               ndat_CSE=self.ndat_CSE,
                    )
            self.transform_func = self.transform_func_MM_CSE
        else:
            print(f"WARNING: This is a metamodel run with no CSE parameters!")
            eos = MetaModel_EOS_model(nmax_nsat = self.nmax_nsat,
                                      ndat = self.ndat_metamodel)
        
            self.transform_func = self.transform_func_MM
        
        self.eos = eos
        
        # Remove those NEPs from the fixed values that we sample over
        if fixed_params is None:
            fixed_params = copy.deepcopy(NEP_CONSTANTS_DICT)
        
        self.fixed_params = fixed_params 
        for name in self.name_mapping[0]:
            if name in list(self.fixed_params.keys()):
                self.fixed_params.pop(name)
                
        print("Fixed params loaded inside the MicroToMacroTransform:")
        for key, value in self.fixed_params.items():
            print(f"    {key}: {value}")
            
        # Construct a lambda function for solving the TOV equations, fix the given parameters
        self.construct_family_lambda = lambda x: construct_family(x, ndat = self.ndat_TOV, min_nsat = self.min_nsat_TOV)
        
    def transform_func_MM(self, params: dict[str, Float]) -> dict[str, Float]:
        
        params.update(self.fixed_params)
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}
        
        # Create the EOS, ignore mu and cs2 (final 2 outputs)
        ns, ps, hs, es, dloge_dlogps, _, cs2 = self.eos.construct_eos(NEP)
        
        # Limit cs2 so that it is causal
        idx = jnp.argmax(cs2 >= 1.0)
        final_n = ns.at[idx].get()
        first_n = ns.at[0].get()
        
        ns_interp = jnp.linspace(first_n, final_n, len(ns))
        ps_interp = jnp.interp(ns_interp, ns, ps)
        hs_interp = jnp.interp(ns_interp, ns, hs)
        es_interp = jnp.interp(ns_interp, ns, es)
        dloge_dlogps_interp = jnp.interp(ns_interp, ns, dloge_dlogps)
        cs2_interp = jnp.interp(ns_interp, ns, cs2)
        
        # Solve the TOV equations
        eos_tuple = (ns_interp, ps_interp, hs_interp, es_interp, dloge_dlogps_interp)
        logpc_EOS, masses_EOS, radii_EOS, Lambdas_EOS = self.construct_family_lambda(eos_tuple)
    
        return_dict = {"logpc_EOS": logpc_EOS, "masses_EOS": masses_EOS, "radii_EOS": radii_EOS, "Lambdas_EOS": Lambdas_EOS,
                       "n": ns_interp, "p": ps_interp, "h": hs_interp, "e": es_interp, "dloge_dlogp": dloge_dlogps_interp, "cs2": cs2_interp}

        return return_dict

    def transform_func_MM_CSE(self, params: dict[str, Float]) -> dict[str, Float]:
        
        params.update(self.fixed_params)
        
        # Separate the MM and CSE parameters
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}
        NEP["nbreak"] = params["nbreak"]
        
        ngrids_u = jnp.array([params[f"n_CSE_{i}_u"] for i in range(self.nb_CSE)])
        ngrids_u = jnp.sort(ngrids_u)
        cs2grids = jnp.array([params[f"cs2_CSE_{i}"] for i in range(self.nb_CSE)])
        
        # From the "quantiles", i.e. the values between 0 and 1, convert between nbreak and nmax
        width = (self.nmax - params["nbreak"])
        ngrids = params["nbreak"] + ngrids_u * width
        
        # Append the final cs2 value, which is fixed at nmax 
        ngrids = jnp.append(ngrids, jnp.array([self.nmax]))
        cs2grids = jnp.append(cs2grids, jnp.array([params[f"cs2_CSE_{self.nb_CSE}"]]))
        
        # Create the EOS, ignore mu and cs2 (final 2 outputs)
        ns, ps, hs, es, dloge_dlogps, _, cs2 = self.eos.construct_eos(NEP, ngrids, cs2grids)
        eos_tuple = (ns, ps, hs, es, dloge_dlogps)
        
        # Solve the TOV equations
        logpc_EOS, masses_EOS, radii_EOS, Lambdas_EOS = self.construct_family_lambda(eos_tuple)
    
        return_dict = {"logpc_EOS": logpc_EOS, "masses_EOS": masses_EOS, "radii_EOS": radii_EOS, "Lambdas_EOS": Lambdas_EOS,
                       "n": ns, "p": ps, "h": hs, "e": es, "dloge_dlogp": dloge_dlogps, "cs2": cs2}
        
        return return_dict
    
    
class ChirpMassMassRatioToSourceComponentMasses(NtoMTransform):
        
    def __init__(
        self,
    ):
        name_mapping = (["M_c", "q", "d_L"], ["m_1", "m_2"])
        super().__init__(name_mapping=name_mapping, keep_names = "all")
        
        self.transform_func = get_source_masses
        
class ChirpMassMassRatioToLambdas(NtoMTransform):
    
    def __init__(
        self,
        name_mapping,
    ):
        super().__init__(name_mapping=name_mapping, keep_names = "all")
        
        self.mass_transform = ChirpMassMassRatioToSourceComponentMasses()
        
    def transform_func(self, params: dict[str, Float]) -> dict[str, Float]:
        
        masses_EOS = params["masses_EOS"]
        Lambdas_EOS = params["Lambdas_EOS"]
        
        # Get masses
        m_params = self.mass_transform.forward(params)
        m_1, m_2 = m_params["m_1"], m_params["m_2"]
        
        # Interpolate to get Lambdas
        lambda_1_interp = jnp.interp(m_1, masses_EOS, Lambdas_EOS, right = -1.0)
        lambda_2_interp = jnp.interp(m_2, masses_EOS, Lambdas_EOS, right = -1.0)
        
        return {"lambda_1": lambda_1_interp, "lambda_2": lambda_2_interp}
    
###################
### LIKELIHOODS ###
###################

class GWlikelihood_with_masses(LikelihoodBase):

    def __init__(self,
                 eos: str,
                 id: str,
                 transform: MicroToMacroTransform = None,
                 very_negative_value: float = -99999.0,
                 N_samples_masses: int = 2_000,
                 hdi_prob: float = 0.90):
        
        self.eos = eos
        self.id = id
        self.name = f"{self.eos}_{self.id}"
        self.transform = transform
        self.very_negative_value = very_negative_value
        
        # Locate the file
        nf_file = os.path.join(NF_PATH, f"models/{self.eos}_{self.id}.eqx")
        nf_kwargs_file = os.path.join(NF_PATH, f"models/{self.eos}_{self.id}_kwargs.json")
        
        if not os.path.exists(nf_file):
            print(f"Tried looking for the NF architecture at path {nf_file}, but it doesn't exist!")
        
        # Load the kwargs used to train the NF to define the PyTree structure
        with open(nf_kwargs_file, "r") as f:
            nf_kwargs = json.load(f)
            
        like_flow = make_flow(jax.random.PRNGKey(0), nf_kwargs["nn_depth"], nf_kwargs["nn_block_dim"])
        
        # Load the normalizing flow
        loaded_model: Transformed = eqx.tree_deserialise_leaves(nf_file, like=like_flow)
        self.NS_posterior = loaded_model
        
        print(f"Loaded the NF for run {self.eos}_{self.id}")
        seed = np.random.randint(0, 100000)
        key = jax.random.key(seed)
        key, subkey = jax.random.split(key)
        
        # Generate some samples from the NS posterior to know the mass range
        nf_samples = self.NS_posterior.sample(subkey, (N_samples_masses,))
        
        # Use it to get the range of m1 and m2
        m1 = nf_samples[:, 0]
        m2 = nf_samples[:, 1]
        
        # # Old method
        # self.m1_min = float(jnp.min(m1))
        # self.m1_max = float(jnp.max(m1))
        
        # self.m2_min = float(jnp.min(m2))
        # self.m2_max = float(jnp.max(m2))
        
        # Instead, we use the 99% credible interval:
        self.m1_min, self.m1_max = arviz.hdi(np.array(m1), hdi_prob=hdi_prob)
        self.m2_min, self.m2_max = arviz.hdi(np.array(m2), hdi_prob=hdi_prob)
        
        print(f"The range of m1 for {self.eos}_{self.id} is: {self.m1_min} to {self.m1_max}")
        print(f"The range of m2 for {self.eos}_{self.id} is: {self.m2_min} to {self.m2_max}")
        

    def evaluate(self, params: dict[str, float], data: dict) -> float:
        
        m1, m2 = params[f"m1_{self.name}"], params[f"m2_{self.name}"]
        penalty_masses = jnp.where(m1 < m2, self.very_negative_value, 0.0)
        
        masses_EOS, Lambdas_EOS = params['masses_EOS'], params['Lambdas_EOS']
        mtov = jnp.max(masses_EOS)
        
        penalty_mass1_mtov = jnp.where(m1 > mtov, self.very_negative_value, 0.0)
        penalty_mass2_mtov = jnp.where(m2 > mtov, self.very_negative_value, 0.0)

        # Lambdas: interpolate to get the values
        lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right = 1.0)
        lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right = 1.0)

        # Make a 4D array of the m1, m2, and lambda values and evalaute NF log prob on it
        ml_grid = jnp.array([m1, m2, lambda_1, lambda_2])
        logpdf_NS = self.NS_posterior.log_prob(ml_grid)
        
        log_likelihood = logpdf_NS + penalty_masses + penalty_mass1_mtov + penalty_mass2_mtov
        
        return log_likelihood
    
class RadioTimingLikelihood(LikelihoodBase):
    
    def __init__(self,
                 psr_name: str,
                 mean: float, 
                 std: float,
                 nb_masses: int = 100,
                 transform: MicroToMacroTransform = None):
        
        self.psr_name = psr_name
        self.transform = transform
        self.nb_masses = nb_masses
        
        self.mean = mean
        self.std = std
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        # Log likelihood is a Gaussian with give mean and std, evalaute it on the masses:
        masses_EOS = params["masses_EOS"]
        mtov = jnp.max(masses_EOS)
        m = jnp.linspace(1.0, mtov, self.nb_masses)
        
        log_likelihood_array = -0.5 * (m - self.mean)**2 / self.std**2
        # Do integration with discrete sum
        log_likelihood = logsumexp(log_likelihood_array) - jnp.log(len(log_likelihood_array))
        log_likelihood -= mtov
        
        return log_likelihood
    
class ChiEFTLikelihood(LikelihoodBase):
    
    def __init__(self,
                 transform: MicroToMacroTransform = None,
                 nb_n: int = 100):
        
        self.transform = transform
        
        # Load the chi EFT data
        low_filename = "/home/twouters2/projects/projection_BNS_A_plus/src/projection_BNS/data/low.dat"
        f = np.loadtxt(low_filename)
        n_low = jnp.array(f[:, 0]) / 0.16 # convert to nsat
        p_low = jnp.array(f[:, 1])
        # NOTE: this is not a spline but it is the best I can do -- does this matter? Need to check later on
        EFT_low = lambda x: jnp.interp(x, n_low, p_low)
        
        high_filename = "/home/twouters2/projects/projection_BNS_A_plus/src/projection_BNS/data/high.dat"
        f = np.loadtxt(high_filename)
        n_high = jnp.array(f[:, 0]) / 0.16 # convert to nsat
        p_high = jnp.array(f[:, 1])
        
        EFT_high = lambda x: jnp.interp(x, n_high, p_high)
        
        self.n_low = n_low
        self.p_low = p_low
        self.EFT_low = EFT_low
        
        self.n_high = n_high
        self.p_high = p_high
        self.EFT_high = EFT_high
        
        self.nb_n = nb_n
        
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        # Get relevant parameters
        n, p = params["n"], params["p"]
        nbreak = params["nbreak"]
        
        # Convert to nsat for convenience
        nbreak = nbreak / 0.16
        n = n / jose_utils.fm_inv3_to_geometric / 0.16
        p = p / jose_utils.MeV_fm_inv3_to_geometric
        
        prefactor = 1 / (nbreak - 0.75 * 0.16)
        
        # Lower limit is at 0.12 fm-3
        this_n_array = jnp.linspace(0.75, nbreak, self.nb_n)
        dn = this_n_array.at[1].get() - this_n_array.at[0].get()
        low_p = self.EFT_low(this_n_array)
        high_p = self.EFT_high(this_n_array)
        
        # Evaluate the sampled p(n) at the given n
        sample_p = jnp.interp(this_n_array, n, p)
        
        # Compute f
        def f(sample_p, low_p, high_p):
            beta = 6/(high_p-low_p)
            return_value = (
                -beta * (sample_p - high_p) * jnp.heaviside(sample_p - high_p, 0) +
                -beta * (low_p - sample_p) * jnp.heaviside(low_p - sample_p, 0) +
                1 * jnp.heaviside(sample_p - low_p, 0) * jnp.heaviside(high_p - sample_p, 0)
            )
            return return_value
            
        f_array = f(sample_p, low_p, high_p) # Well actually already log f
        log_likelihood = prefactor * jnp.sum(f_array) * dn
        
        return log_likelihood
        
class CombinedLikelihood(LikelihoodBase):
    
    def __init__(self,
                 likelihoods_list: list[LikelihoodBase],
                 transform: MicroToMacroTransform = None):
        
        # TODO: remove transform input?
        
        super().__init__()
        self.likelihoods_list = likelihoods_list
        self.transform = transform
        self.counter = 0
        
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        all_log_likelihoods = jnp.array([likelihood.evaluate(params, data) for likelihood in self.likelihoods_list])
        return jnp.sum(all_log_likelihoods)
    
class ZeroLikelihood(LikelihoodBase):
    def __init__(self,
                 transform: MicroToMacroTransform = None):
        
        super().__init__()
        self.transform = transform
        self.counter = 0
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        return 0.0