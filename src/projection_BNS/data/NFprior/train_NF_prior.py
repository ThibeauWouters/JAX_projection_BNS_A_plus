"""
Train an NF to replicate an EOS informed prior for masses and Lambdas.

# FIXME: this is just a duplicate from NFTrainer.py, but for now, I'd like to keep these things separate
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import corner
import jax
import json
import copy
import jax.numpy as jnp
import equinox as eqx
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.distributions import Normal
jax.config.update("jax_enable_x64", True) # this needs to be activated if we wish to have x64 in the EOS inference

print("GPU found?")
print(jax.devices())

params = {"axes.grid": True,
        "text.usetex" : False,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        # "font.serif" : ["Computer Modern Serif"],
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
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)
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


class NFPriorCreator:
    
    def __init__(self,
                 eos_samples_filename: str = None,
                 downsample_factor: int = 1,
                 N_samples: int = 100_000,
                 N_samples_plot: int = 10_000,
                 ):
        
        self.eos_samples_filename = eos_samples_filename
        self.downsample_factor = downsample_factor
        self.N_samples = N_samples
        self.N_samples_plot = N_samples_plot
        
    def create_data(self):
        print("Loading EOS posterior samples and creating the prior samples dataset for training")
        if self.eos_samples_filename is None:
            # By default, we use the one from an EOS inference on only the radio timing (which ensures MTOV > 2)
            self.eos_samples_filename = "/home/twouters2/projects/projection_BNS_A_plus/src/projection_BNS/EOS/outdir_radio/eos_samples.npz"
            
        if not os.path.exists(self.eos_samples_filename):
            raise ValueError(f"File {self.eos_samples_filename} does not exist.")
        
        print(f"Reading the EOS data from {self.eos_samples_filename}")
        eos_samples = np.load(self.eos_samples_filename)

        # Print the keys
        print("list(eos_samples.keys()")
        print(list(eos_samples.keys()))
        
        # Get the data
        masses_EOS, radii_EOS, Lambdas_EOS = eos_samples["masses_EOS"], eos_samples["radii_EOS"], eos_samples["Lambdas_EOS"]
        
        print("np.shape(masses_EOS)")
        print(np.shape(masses_EOS))

        # Downsample data
        masses_EOS = masses_EOS[::self.downsample_factor]
        radii_EOS = radii_EOS[::self.downsample_factor]
        Lambdas_EOS = Lambdas_EOS[::self.downsample_factor]

        # Iterate over EOS and keep those that are fine
        nb_samples = len(masses_EOS)
        good_idx = np.ones(nb_samples, dtype=bool)

        # TODO: This is a bit of a hack... but it's ok for now
        for i in range(nb_samples):
            # First, sometimes the radius can be very large for low mass stars, which is unphysical
            bad_radii = (masses_EOS[i] > 1.0) * (radii_EOS[i] > 20.0)
            if any(bad_radii):
                good_idx[i] = False
                continue
            # Second, sometimes a negative Lambda was computed, remove that
            bad_Lambdas = (Lambdas_EOS[i] < 0.0)
            if any(bad_Lambdas):
                good_idx[i] = False
                continue
            # Finally, we want the TOV mass to be above 2.0 M_odot
            bad_MTOV = np.max(masses_EOS) < 2.0
            if bad_MTOV:
                good_idx[i] = False
                continue
    
        print("Number of good samples: ", np.sum(good_idx) / nb_samples)

        masses_EOS = masses_EOS[good_idx]
        radii_EOS = radii_EOS[good_idx]
        Lambdas_EOS = Lambdas_EOS[good_idx]

        m1_list = np.empty(self.N_samples)
        m2_list = np.empty(self.N_samples)
        Lambda1_list = np.empty(self.N_samples)
        Lambda2_list = np.empty(self.N_samples)

        # Construct the prior from sampling from the EOS set
        for i in range(self.N_samples):
            idx = np.random.randint(0, len(masses_EOS))
            m, l = masses_EOS[idx], Lambdas_EOS[idx]
            
            # Sample two masses between 1 and MTOV for this EOS
            mtov = np.max(m)
            mass_samples = np.random.uniform(1.0, mtov, 2)
            m1 = np.max(mass_samples)
            m2 = np.min(mass_samples)
            
            Lambda_1 = np.interp(m1, m, l)
            Lambda_2 = np.interp(m2, m, l)
            
            m1_list[i] = m1
            m2_list[i] = m2
            Lambda1_list[i] = Lambda_1
            Lambda2_list[i] = Lambda_2
            
        # Make a cornerplot of the dataset
        plot_range = [[np.min(m1_list), np.max(m1_list)], 
                        [np.min(m2_list), np.max(m2_list)], 
                        [0, 3000],
                        [0, 5000]
                        ]

        print(f"M1 ranges from {np.min(m1_list)} to {np.max(m1_list)}")
        print(f"M2 ranges from {np.min(m2_list)} to {np.max(m2_list)}")

        print(f"Saving data")
        np.savez("./eos_prior_samples.npz", m1 = np.array(m1_list), m2 = np.array(m2_list), lambda_1 = np.array(Lambda1_list), lambda_2 = np.array(Lambda2_list))
        print(f"Saving data DONE")

        # Downsample for plotting
        if self.N_samples_plot < self.N_samples:
            jump = self.N_samples // self.N_samples_plot
            
            m1_list = m1_list[::jump]
            m2_list = m2_list[::jump]
            Lambda1_list = Lambda1_list[::jump]
            Lambda2_list = Lambda2_list[::jump]
            
        # Make a cornerplot of masses and Lambdas
        data = np.array([m1_list, m2_list, Lambda1_list, Lambda2_list]).T
        print("np.shape(data)")
        print(np.shape(data))
        corner.corner(data, labels=[r"$m_1$ [$M_\odot$]", r"$m_2$ [$M_\odot$]", r"$\Lambda_1$", r"$\Lambda_2$"], range=plot_range, **default_corner_kwargs)
        plt.savefig("./prior_dataset.pdf", bbox_inches = "tight")
        plt.close()
        
    def load_data(self):
        data = np.load("./eos_prior_samples.npz")
        self.m1 = data["m1"]
        self.m2 = data["m2"]
        self.lambda_1 = data["lambda_1"]
        self.lambda_2 = data["lambda_2"]
        
    def train(self,
              num_epochs: int = 600,
              learning_rate: float = 1e-3,
              max_patience: int = 50,
              nn_depth: int = 5,
              nn_block_dim: int = 8,
              ):
        """
        As said above, a lot of duplication with the NFTrainer.py file, but I want to keep things separate for now. Might merge them, but I do not require this now.
        """
        
        self.load_data()
        data_np = np.array([self.m1, self.m2, self.lambda_1, self.lambda_2])
        
        flow_key, train_key, sample_key = jax.random.split(jax.random.key(0), 3)
        x = data_np.T
        
        # Get range from the data for plotting
        my_range = np.array([[np.min(x.T[i]), np.max(x.T[i])] for i in range(4)])
        widen_array = np.array([[-0.1, 0.1], [-0.1, 0.1], [-100, 100], [-20, 20]])
        my_range += widen_array
        
        flow = block_neural_autoregressive_flow(
            key=flow_key,
            base_dist=Normal(jnp.zeros(4)),
            nn_depth=nn_depth,
            nn_block_dim=nn_block_dim,
        )
        nf_kwargs = {"nn_depth": nn_depth, "nn_block_dim": nn_block_dim}
        flow, _ = fit_to_data(key=train_key,
                                   dist=flow,
                                   x=x,
                                   learning_rate=learning_rate,
                                   max_epochs=num_epochs,
                                   max_patience=max_patience
                                   )
        
        # And sample the distribution
        nf_samples = flow.sample(sample_key, (self.N_samples_plot, ))
        nf_samples_np = np.array(nf_samples)
        
        corner_name = "corner.pdf"
        make_cornerplot(data_np, nf_samples_np, my_range, corner_name)
        
        # Save the model weights
        save_path = "NFPrior.eqx"
        print(f"Saving the model weights to {save_path}")
        eqx.tree_serialise_leaves(save_path, flow)
        
        # Also dump all the flowjax kwargs so we can reproduce the NF architecture easily
        kwargs_save_path = "NFPrior_kwargs.json"
        print(f"Saving the model kwargs to {kwargs_save_path}")
        with open(kwargs_save_path, "w") as f:
            json.dump(nf_kwargs, f)
            
        print(f"Training done!")
        
        
            
def main():
    trainer = NFPriorCreator()
    # trainer.create_data()
    trainer.train()
    
if __name__ == "__main__":
    main()