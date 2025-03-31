import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import corner

import projection_BNS.utils as utils

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
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)

def compare_lambda_cornerplots(first_dir: str,
                               second_dir: str,
                               downsample_factor: int = 50):
    
    # Check if the directories exist
    if not os.path.isdir(first_dir):
        raise ValueError("Directory {} does not exist".format(first_dir))
    
    if not os.path.isdir(second_dir):
        raise ValueError("Directory {} does not exist".format(second_dir))
    
    # Get the names
    first_eos_name = utils.get_eos_name_from_dirname(first_dir)
    second_eos_name = utils.get_eos_name_from_dirname(second_dir)
    
    # From which we get the colors
    first_color = utils.TARGET_COLORS_DICT[first_eos_name]
    second_color = utils.TARGET_COLORS_DICT[second_eos_name]
    
    # Get the samples
    first_chains = np.load(os.path.join(first_dir, "chains_production.npz"))
    first_lambda_1 = first_chains["lambda_1"].flatten()
    first_lambda_2 = first_chains["lambda_2"].flatten()
    
    print("first_lambda_1.shape")
    print(first_lambda_1.shape)
    
    first_lambda_1 = first_lambda_1[::downsample_factor]
    first_lambda_2 = first_lambda_2[::downsample_factor]
    
    print("first_lambda_1.shape")
    print(first_lambda_1.shape)
    
    second_chains = np.load(os.path.join(second_dir, "chains_production.npz"))
    second_lambda_1 = second_chains["lambda_1"].flatten()
    second_lambda_2 = second_chains["lambda_2"].flatten()
    
    second_lambda_1 = second_lambda_1[::downsample_factor]
    second_lambda_2 = second_lambda_2[::downsample_factor]
    
    # Get the injected value
    first_injection_filename = os.path.join(first_dir, "injection.json")
    with open(first_injection_filename, "r") as f:
        first_injection = json.load(f)
        
    first_injected_lambda_1 = first_injection["lambda_1"]
    first_injected_lambda_2 = first_injection["lambda_2"]
    
    second_injection_filename = os.path.join(second_dir, "injection.json")
    with open(second_injection_filename, "r") as f:
        second_injection = json.load(f)

    second_injected_lambda_1 = second_injection["lambda_1"]
    second_injected_lambda_2 = second_injection["lambda_2"]
    
    # Now we can make the 2D corner plot
    corner_kwargs = copy.deepcopy(default_corner_kwargs)
    
    # contour_kwargs = {"alpha": 0.0}
    # corner_kwargs["contourf_kwargs"] = contour_kwargs
    
    corner_kwargs["color"] = first_color
    corner_kwargs["truth_color"] = first_color
    corner_kwargs["hist_kwargs"] = {"color": first_color, "density": True}
    
    fig = corner.corner(np.column_stack((first_lambda_1, first_lambda_2)), truths=[first_injected_lambda_1, first_injected_lambda_2], **corner_kwargs)
    
    # Now plot the second set
    corner_kwargs["color"] = second_color
    corner_kwargs["truth_color"] = second_color
    corner_kwargs["hist_kwargs"] = {"color": second_color, "density": True}
    
    corner.corner(np.column_stack((second_lambda_1, second_lambda_2)), truths=[second_injected_lambda_1, second_injected_lambda_2], fig=fig, **corner_kwargs)
    
    # Save it
    plt.savefig("./figures/comparison_cornerplot.pdf", bbox_inches="tight")
    plt.close()
    
    
if __name__ == "__main__":
    
    first_dir = os.path.abspath("../GW/HQC18/Aplus/injection_31")
    second_dir = os.path.abspath("../GW/MPA1/Aplus/injection_31")
    
    compare_lambda_cornerplots(first_dir, second_dir)