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
                        plot_datapoints=True, 
                        fill_contours=False,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "black",
                        density=True,
                        save=False)

import argparse
import jax
import jax.numpy as jnp
import equinox as eqx
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.distributions import Normal
jax.config.update("jax_enable_x64", True) # this needs to be activated if we wish to have x64 in the EOS inference

print("GPU found?")
print(jax.devices())

GW_PATH = "/home/twouters2/projects/projection_BNS_A_plus/src/projection_BNS/GW"
NF_PATH = "/home/twouters2/projects/projection_BNS_A_plus/src/projection_BNS/NF"
ALLOWED_EOS = ["HQC18", "MPA1", "SLY230A"]
c = 299_792.458 # km/s
H0 = 67.74 # km/s/Mpc

def parse_arguments():
    parser = argparse.ArgumentParser(description="Trains an NF to approximate the marginal of the component masses and the tidal deformabilities")
    parser.add_argument("--eos", 
                        type=str, 
                        help="Name of the EOS. Choose from [HQC18, MPA1, SLY230A].")
    parser.add_argument("--ifo-network", 
                        type=str, 
                        help="Name of the network of detectors. Choose from [Aplus, Asharp, ET].")
    parser.add_argument("--id", 
                        type=int, 
                        help="Identifier of the GW injection for that EOS.")
    # Now come the flowjax kwargs etc:
    parser.add_argument("--num_epochs",
                        type=int,
                        default=600,
                        help="Number of epochs for the training.")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-3,
                        help="Learning rate for the training.")
    parser.add_argument("--max_patience",
                        type=int,
                        default=50,
                        help="Maximum patience for the training.")
    parser.add_argument("--nn_depth",
                        type=int,
                        default=5,
                        help="Depth of the neural network blocks.")
    parser.add_argument("--nn_block_dim",
                        type=int,
                        default=8,
                        help="Dimension of the neural network blocks.")
    parser.add_argument("--nb_samples_train",
                        type=int,
                        default=20_000,
                        help="Number of samples to train the NF on.")
    
    return parser.parse_args()

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
    corner.corner(chains_2, range=range, truths=truths, fig=fig, **corner_kwargs)

    # Make a textbox just because that makes the plot cooler
    fs = 32
    plt.text(0.75, 0.75, "Training data", fontsize = fs, color = "blue", transform = plt.gcf().transFigure)
    plt.text(0.75, 0.65, "Normalizing flow", fontsize = fs, color = "red", transform = plt.gcf().transFigure)

    plt.savefig(name, bbox_inches = "tight")
    plt.close()
    
def get_source_masses(M_c: float, q: float, d_L: float):
    """
    Given the detector-frame chirp mass, the mass ratio and the redshift, compute the component masses
    FIXME: Using the linear Hubble relation for now. Make this work with astropy or at least with more accurate cosmology -- perhaps get some "surrogate"

    The value of the Hubble constant is chosen similar to https://inspirehep.net/literature/2669070

    Args:
        M_c (float): Detector frame chirp mass
        q (float): Mass ratio
        d_L (float): Luminosity distance in megaparsecs

    Returns:
        tuple[float, float]: Source frame component masses (primary, secondary)
    """
    # TODO: this is more accurate, but is not jax-compatible yet!
    # d_L_units = d_L * u.Mpc
    # z = z_at_value(Planck18.luminosity_distance, d_L_units)
    
    z = d_L * H0 / c
    print("redshift is roughly", jnp.median(z))
    M_c_source = M_c / (1 + z)
    m_1 = M_c_source * ((1 + q) ** (1/5))/((q) ** (3/5))
    m_2 = M_c_source * ((q) ** (2/5)) * ((1+q) ** (1/5))
    return m_1, m_2

def make_flow(flow_key,
              nn_depth: int = 5,
              nn_block_dim: int = 8):
    """
    Simple function to make a flow just to unify this across the code.
    Documentation for the current default flow architecture can be found here: https://danielward27.github.io/flowjax/api/flows.html#flowjax.flows.block_neural_autoregressive_flow
    """
    
    flow = block_neural_autoregressive_flow(
            key=flow_key,
            base_dist=Normal(jnp.zeros(4)),
            nn_depth=nn_depth,
            nn_block_dim=nn_block_dim,
        )
    return flow

class NFTrainer:
    """Class to train an NF to approximate the marginal of the component masses and the tidal deformabilities"""
    
    def __init__(self, 
                 # general args
                 eos_name: str,
                 ifo_network: str,
                 injection_idx: int,
                 nb_samples_train: int,
                 # flowjax kwargs
                 num_epochs: int,
                 learning_rate: float,
                 max_patience: int,
                 nn_depth: int,
                 nn_block_dim: int,
                 plot_learning_curves: bool = False):
        
        # Set attributes
        self.eos_name = eos_name
        self.ifo_network = ifo_network
        self.injection_idx = injection_idx
        self.nb_samples_train = nb_samples_train
        
        self.figure_save_location = f"./figures/{self.eos_name}/{self.ifo_network}/{self.injection_idx}_"
        self.model_save_location = f"./models/{self.eos_name}/{self.ifo_network}/{self.injection_idx}"
        
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_patience = max_patience
        self.nn_depth = nn_depth
        self.nn_block_dim = nn_block_dim
        
        self.plot_learning_curves = plot_learning_curves
        
        # Get the full path
        self.directory = os.path.join(GW_PATH, self.eos_name, self.ifo_network, f"injection_{self.injection_idx}")
        print(f"We are looking at the directory {self.directory} for GW inference data")
        
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
        
        print(f"The true values are {m1_true}, {m2_true}, {self.lambda_1_true}, {self.lambda_2_true}")
        
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
        # TODO: fix this please
        my_range = np.array([[np.min(x.T[i]), np.max(x.T[i])] for i in range(n_dim)])
        widen_array = np.array([[-0.1, 0.1], [-0.1, 0.1], [-100, 100], [-20, 20]])
        my_range += widen_array
        
        flow = make_flow(flow_key, nn_depth=self.nn_depth, nn_block_dim=self.nn_block_dim)
        flow, losses = fit_to_data(key=train_key,
                                   dist=flow,
                                   x=x,
                                   learning_rate=self.learning_rate,
                                   max_epochs=self.num_epochs,
                                   max_patience=self.max_patience
                                   )
        
        # Plot learning curves
        if self.plot_learning_curves:
            plt.figure(figsize = (12, 8))
            plt.plot(losses["train"], label = "Train", color = "red")
            plt.plot(losses["val"], label = "Val", color = "blue")
            plt.yscale("log")
            plt.legend()
            plt.savefig(self.figure_save_location + "loss.png", bbox_inches = "tight")
            plt.close()
        
        # And sample the distribution
        nf_samples = flow.sample(sample_key, (N_samples_plot, ))
        nf_samples_np = np.array(nf_samples)
        
        if hasattr(self, "m1_true"):
            truths = [self.m1_true, self.m2_true, self.lambda_1_true, self.lambda_2_true]
        else:
            truths = None
        
        corner_name = self.figure_save_location + "corner.png"
        make_cornerplot(data_np, nf_samples_np, my_range, corner_name, truths=truths)
        
        # Save the model weights
        save_path = self.model_save_location + ".eqx"
        print(f"Saving the model weights to {save_path}")
        eqx.tree_serialise_leaves(save_path, flow)
        
        # Also dump all the flowjax kwargs so we can reproduce the NF architecture easily
        kwargs_save_path = self.model_save_location + ".eqx"
        print(f"Saving the model kwargs to {kwargs_save_path}")
        nf_kwargs = {"num_epochs": self.num_epochs, "learning_rate": self.learning_rate, "max_patience": self.max_patience, "nn_depth": self.nn_depth, "nn_block_dim": self.nn_block_dim}
        with open(kwargs_save_path, "w") as f:
            json.dump(nf_kwargs, f)
            
        print(f"Training of the NF for {self.eos_name} and injection {self.injection_idx} was successful")

        
def main():
    # Get the args:
    args = parse_arguments()
    
    trainer = NFTrainer(eos_name = args.eos,
                        ifo_network = args.ifo_network,
                        injection_idx = args.id,
                        nb_samples_train = args.nb_samples_train,
                        num_epochs = args.num_epochs,
                        learning_rate = args.learning_rate,
                        max_patience = args.max_patience,
                        nn_depth = args.nn_depth,
                        nn_block_dim = args.nn_block_dim)
    trainer.train()
    
    
if __name__ == "__main__":
    main()