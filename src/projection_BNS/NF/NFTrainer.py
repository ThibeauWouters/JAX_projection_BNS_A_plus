"""
Code to train the NF on the GW posterior to approximate the marginal

Get cosmology in bilby uses Planck15 by default https://git.ligo.org/lscsoft/bilby/-/blob/c6bcb81649b7ebf97ae6e1fd689e8712fe028eb0/bilby/gw/cosmology.py#L17
"""

import os 
import sys

import matplotlib.pyplot as plt
import corner
import numpy as np
import copy
import json

from astropy.cosmology import Planck18 
from astropy import units as u
from astropy.cosmology import z_at_value

### Stuff for nice plots
params = {"axes.grid": True,
        "text.usetex" : False,
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

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        levels=[0.68, 0.9, 0.997],
                        plot_density=False, 
                        plot_datapoints=False, 
                        fill_contours=False,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "black",
                        density=True,
                        save=False)

import jax
import jax.numpy as jnp
import equinox as eqx
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.distributions import Normal
# jax.config.update("jax_enable_x64", True)

print("GPU found?")
print(jax.devices())

GW_PATH = "../GW/"

def make_cornerplot(chains_1: np.array, 
                    chains_2: np.array,
                    range: list[float],
                    name: str,
                    truths: list[float] = None):
    """
    Plot a cornerplot of the true data samples and the NF samples
    Note: the shape use is a bit inconsistent below, watch out.
    """

    # The training data:
    corner_kwargs = copy.deepcopy(default_corner_kwargs)
    hist_1d_kwargs = {"density": True, "color": "blue"}
    corner_kwargs["color"] = "blue"
    corner_kwargs["hist_kwargs"] = hist_1d_kwargs
    fig = corner.corner(chains_1.T, range=range, truths=truths, labels = [r"$m_1$", r"$m_2$", r"$\Lambda_1$", r"$\Lambda_2$"], **corner_kwargs)

    # The data from the normalizing flow
    corner_kwargs["color"] = "red"
    hist_1d_kwargs = {"density": True, "color": "red"}
    corner_kwargs["hist_kwargs"] = hist_1d_kwargs
    corner.corner(chains_2, range=range, fig = fig, **corner_kwargs)

    # Make a textbox just because that makes the plot cooler
    fs = 32
    plt.text(0.75, 0.75, "Training data", fontsize = fs, color = "blue", transform = plt.gcf().transFigure)
    plt.text(0.75, 0.65, "Normalizing flow", fontsize = fs, color = "red", transform = plt.gcf().transFigure)

    plt.savefig(name, bbox_inches = "tight")
    plt.close()
    
def get_source_masses(M_c: float, q: float, d_L: float):
    """
    Given the detector-frame chirp mass, the mass ratio and the redshift, compute the component masses

    Args:
        M_c (float): Detector frame chirp mass
        q (float): Mass ratio
        d_L (float): Luminosity distance in megaparsecs

    Returns:
        tuple[float, float]: Source frame component masses ()
    """
    d_L_units = d_L * u.Mpc
    z = z_at_value(Planck18.luminosity_distance, d_L_units)
    
    M_c_source = M_c / (1 + z)
    m_1 = M_c_source * ((1 + q) ** (1/5))/((q) ** (3/5))
    m_2 = M_c_source * ((q) ** (2/5)) * ((1+q) ** (1/5))
    return m_1, m_2

class NFTrainer:
    """Class to train an NF to approximate the marginal of the component masses and the tidal deformabilities"""
    
    def __init__(self, 
                 # general args
                 eos_name: str,
                 injection_idx: int,
                 nb_samples_train: int = 1_000,
                 # flowjax kwargs
                 num_epochs: int = 600,
                 learning_rate: float = 5e-4,
                 max_patience: int = 50,
                 nn_depth: int = 5,
                 nn_block_dim: int = 8):
        
        # Set attributes
        self.eos_name = eos_name
        self.injection_idx = injection_idx
        self.nb_samples_train = nb_samples_train
        
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_patience = max_patience
        self.nn_depth = nn_depth
        self.nn_block_dim = nn_block_dim
        
        # Get the full path
        self.directory = os.path.join(GW_PATH, self.eos_name, "outdir", f"injection_{self.injection_idx}")
        print(f"We are looking at the directory {self.directory}")
        
        self.chains_path = os.path.join(self.directory, "chains_production.npz")
        self.injection_path = os.path.join(self.directory, "injection.json")
        
        # Load the data
        self.load_data()
        
        # Load the injected values
        self.load_injection_values()
        
    def load_data(self):
        """
        Load the data from the GW run and preprocess it (downsample and get the component masses)
        """
        gw_data = np.load(self.chains_path)
        M_c, q, lambda_1, lambda_2, d_L = gw_data["M_c"].flatten(), gw_data["q"].flatten(), gw_data["lambda_1"].flatten(), gw_data["lambda_2"].flatten(), gw_data["d_L"].flatten()
        nb_samples = len(M_c)
        self.downsampling_factor = int(np.ceil(nb_samples // self.nb_samples_train))
        
        print(f"Downsampling the data with a factor of {self.downsampling_factor}")
        M_c = M_c[::self.downsampling_factor]
        q = q[::self.downsampling_factor]
        lambda_1 = lambda_1[::self.downsampling_factor]
        lambda_2 = lambda_2[::self.downsampling_factor]
        d_L = d_L[::self.downsampling_factor]
        
        print(f"Started with {nb_samples} samples and now have {len(M_c)} samples")

        # Compute the component masses, using cosmology to convert d_L to z        
        m_1, m_2 = get_source_masses(M_c, q, d_L)
        
        self.m1 = m_1
        self.m2 = m_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        
    def load_injection_values(self):
        """
        Load the true values of the injection here so we can also add them to the cornerplots
        """
        filename = os.path.join(self.directory, "injection.json")
        with open(filename, "r") as f:
            injection_dict = json.load(f)
            
        print(f"Loaded the injection values: {injection_dict}")
        
        m1_true, m2_true = get_source_masses(injection_dict["M_c"], injection_dict["q"], injection_dict["d_L"])
        
        self.m1_true = m1_true
        self.m2_true = m2_true
        self.lambda_1_true = injection_dict["lambda_1"]
        self.lambda_2_true = injection_dict["lambda_2"]
        
    def train(self):
        """
        Train the NF on the GW data to convergence and check the final result
        """
        n_dim = 4
        data_np = np.array([self.m1, self.m2, self.lambda_1, self.lambda_2])
        
        N_samples_plot = 10_000
        flow_key, train_key, sample_key = jax.random.split(jax.random.key(0), 3)

        x = data_np.T
        
        # Get range from the data for plotting
        my_range = np.array([[np.min(x.T[i]), np.max(x.T[i])] for i in range(n_dim)])
        widen_array = np.array([[-0.1, 0.1], [-0.1, 0.1], [-100, 100], [-20, 20]])
        my_range += widen_array
        
        flow = block_neural_autoregressive_flow(
            key=flow_key,
            base_dist=Normal(jnp.zeros(x.shape[1])),
            nn_depth=self.nn_depth,
            nn_block_dim=self.nn_block_dim,
        )
        
        flow, losses = fit_to_data(
            key=train_key,
            dist=flow,
            x=x,
            learning_rate=self.learning_rate,
            max_epochs=self.num_epochs,
            max_patience=self.max_patience
            )
        
        # Plot learning curves
        plt.figure(figsize = (12, 8))
        plt.plot(losses["train"], label = "Train", color = "red")
        plt.plot(losses["val"], label = "Val", color = "blue")
        plt.yscale("log")
        plt.legend()
        plt.savefig(f"./figures/loss_{self.eos_name}_{self.injection_idx}.png", bbox_inches = "tight")
        plt.close()
        
        # And sample the distribution
        nf_samples = flow.sample(sample_key, (N_samples_plot, ))
        nf_samples_np = np.array(nf_samples)
        
        if hasattr(self, "m1_true"):
            truths = [self.m1_true, self.m2_true, self.lambda_1_true, self.lambda_2_true]
        else:
            truths = None
        
        corner_name = f"./figures/corner_{self.eos_name}_{self.injection_idx}.png"
        make_cornerplot(data_np, nf_samples_np, my_range, corner_name, truths=truths)
        


# def train(WHICH: str):
    
#     if WHICH not in PATHS_DICT.keys():
#         raise ValueError(f"WHICH must be one of {PATHS_DICT.keys()}s")

#     print(f"\n\n\nTraining the NF for the {WHICH} data run . . . \n\n\n")

#     ############
#     ### BODY ###
#     ############

#     data = load_complete_data(WHICH)

#     print(f"Loaded data with shape {np.shape(data)}")
#     n_dim, n_samples = np.shape(data)
#     print(f"ndim = {n_dim}, nsamples = {n_samples}")
#     data_np = np.array(data)

#     N_samples_plot = 10_000
#     flow_key, train_key, sample_key = jax.random.split(jax.random.key(0), 3)

#     x = data.T # shape must be (n_samples, n_dim)
#     x = np.array(x)
#     print("np.shape(x)")
#     print(np.shape(x))

#     # Get range from the data for plotting
#     if n_dim == 4 and WHICH != "NF_prior":
#         # This is for the GW run
#         my_range = np.array([[np.min(x.T[i]), np.max(x.T[i])] for i in range(n_dim)])
#         widen_array = np.array([[-0.2, 0.2], [-0.2, 0.2], [-100, 100], [-20, 20]])
#         my_range += widen_array
#         num_epochs = 600
#     elif WHICH == "NF_prior":
#         num_epochs = 1_000
#         my_range = np.array([[0.75, 3.5],
#                              [0.75, 3.5],
#                              [-10.0, 2000.0],
#                              [-10.0, 6000.0]])
#     else:
#         my_range = None
#         num_epochs = 100
#     print(f"The range is {my_range}")

#     flow = block_neural_autoregressive_flow(
#         key=flow_key,
#         base_dist=Normal(jnp.zeros(x.shape[1])),
#         nn_depth=5,
#         nn_block_dim=8,
#     )

#     flow, losses = fit_to_data(
#         key=train_key,
#         dist=flow,
#         x=x,
#         learning_rate=5e-4,
#         max_epochs=num_epochs,
#         max_patience=50
#         )

#     plt.plot(losses["train"], label = "Train", color = "red")
#     plt.plot(losses["val"], label = "Val", color = "blue")
#     plt.yscale("log")
#     plt.legend()
#     plt.savefig(f"./figures/NF_training_losses_{WHICH}.png")
#     plt.close()

#     # And sample the distribution
#     nf_samples = flow.sample(sample_key, (N_samples_plot, ))
#     nf_samples_np = np.array(nf_samples)

#     make_cornerplot(data_np, nf_samples_np, range=my_range, name=f"./figures/NF_corner_{WHICH}.png")

#     # Save the model
#     save_path = f"./NF/NF_model_{WHICH}.eqx"
#     eqx.tree_serialise_leaves(save_path, flow)

#     loaded_model = eqx.tree_deserialise_leaves(save_path, like=flow)

#     # And sample the distribution
#     nf_samples_loaded = loaded_model.sample(sample_key, (N_samples_plot, ))
#     nf_samples_loaded_np = np.array(nf_samples_loaded)

#     log_prob = loaded_model.log_prob(nf_samples_loaded)

#     make_cornerplot(data_np, nf_samples_loaded_np, range=my_range, name=f"./figures/NF_corner_{WHICH}_reloaded.png")

def main():
    # # Get the "which" argument from the command line
    # if len(sys.argv) < 2:
    #     raise ValueError("Usage: python train_normalizing_flow.py <which>")
    # WHICH = sys.argv[1]
    # train(WHICH)
    
    trainer = NFTrainer("HQC18", 3)
    
    
if __name__ == "__main__":
    main()