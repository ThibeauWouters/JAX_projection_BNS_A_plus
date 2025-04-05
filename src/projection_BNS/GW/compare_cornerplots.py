import numpy as np
import matplotlib.pyplot as plt
import corner 
import json
import copy

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


run_first = "./jester_hard/Aplus/injection_1/chains_production.npz"
run_second = "./jester_hard/Aplus/injection_1_v2/chains_production.npz"
data_first = np.load(run_first)
data_second = np.load(run_second)

NB_samples_plot = 5_000

# ### Big cornerplot
# keys = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'iota', 'psi', 'ra', 'dec']
# labels = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'iota', 'psi', 'ra', 'dec']

### Smaller cornerplot
keys = ['M_c', 'q', 'lambda_1', 'lambda_2']
labels = [r'$M_c$', r'$q$', r'$\Lambda_1$', r'$\Lambda_2$']

injection_filename = run_second.replace("chains_production.npz", "injection.json")
with open(injection_filename, 'r') as f:
    injection_data = json.load(f)
    
truths = []
for key in keys:
    if key in injection_data:
        truths.append(injection_data[key])
        
truths = np.array(truths)

values_first = []
for key in keys:
    values_first.append(data_first[key].flatten())
    
values_first = np.array(values_first)
print(np.shape(values_first))
# Downsample
len_values_first = values_first.shape[1]
jump = len_values_first // NB_samples_plot
values_first = values_first[:, ::jump]
print(np.shape(values_first))

corner_kwargs = copy.deepcopy(default_corner_kwargs)
hist_kwargs = {"color": "blue", "density": True}
corner_kwargs["color"] = "blue"
corner_kwargs["hist_kwargs"] = hist_kwargs

fig = corner.corner(values_first.T, labels=labels, truths=truths, **corner_kwargs)

values_second = []
for key in keys:
    values_second.append(data_second[key].flatten())

values_second = np.array(values_second)
print(np.shape(values_second))
# Downsample
len_values_second = values_second.shape[1]
jump = len_values_second // NB_samples_plot
values_second = values_second[:, ::jump]
print(np.shape(values_second))


corner_kwargs = copy.deepcopy(default_corner_kwargs)
hist_kwargs["color"] = "green"
corner_kwargs["color"] = "green"
corner_kwargs["hist_kwargs"] = hist_kwargs

if len(truths) < 5:
    save_name = run_second.replace("chains_production.npz", "corner_comparison_small.pdf")
else:
    save_name = run_second.replace("chains_production.npz", "corner_comparison.pdf")
corner.corner(values_second.T, labels=labels, truths=truths, fig=fig, **corner_kwargs)
plt.savefig(save_name, bbox_inches='tight')
plt.close()