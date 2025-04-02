import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import arviz
import corner
import tqdm
import sys

import joseTOV.utils as jose_utils
from projection_BNS.utils import DATA_PATH

mpl_params = {"axes.grid": True,
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

plt.rcParams.update(mpl_params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        levels=[0.68, 0.95],
                        plot_density=False,
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False)

# TODO: change this again, but I am testing now
TARGET_KWARGS = {"zorder": 1e12,
                 "lw": 2,
                 "linestyle": "-"}

def plot_corner(outdir):
    print(f"Plotting the cornerplot for {outdir}")
    
    # Load the chains:
    samples = np.load(os.path.join(outdir, "eos_samples.npz"))
    all_keys = list(samples.keys())
    keys = []
    
    # Only take the EOS parameters -- this is a bit messy but ensures we get all of them.
    for key in all_keys:
        if "_sym" in key or "_sat" in key or "CSE" in key or key == "nbreak":
            if key == "E_sat":
                continue
            keys.append(key)
            
    print(f"Making the cornerplot, these are the chains: {keys}")
    samples = {k: v.flatten() for k, v in samples.items() if k in keys}
    samples = np.array(list(samples.values()))
    
    print(np.shape(samples))
    
    corner.corner(samples.T, labels = keys, **default_corner_kwargs)
    plt.savefig(os.path.join(outdir, "corner.pdf"), bbox_inches = "tight")
    plt.close()
    
def report_credible_interval(values: np.array, 
                             hdi_prob: float = 0.95) -> None:
    med = np.median(values)
    low, high = arviz.hdi(values, hdi_prob = hdi_prob)
    
    low = med - low
    high = high - med
    
    print(f"\n\n\n{med:.2f}-{low:.2f}+{high:.2f} (at {hdi_prob} HDI prob)\n\n\n")
    
    return med, low, high

def check_convergence(outdir: str):

    print(f"Checking the convergence for {outdir}")
    
    data = np.load(os.path.join(outdir, "results_production.npz"))
    for key in data.keys():
        if key == "log_prob":
            continue
        print(f"Key: {key}")
        values = data[key]
        values = np.array(values)
        rhat = arviz.rhat(values)
        ess = arviz.ess(values)
        
        print(f"    rhat = {rhat}")
        print(f"    ess = {ess}")

def make_plots(outdir: str,
               plot_histograms: bool = True,
               make_master_plot: bool = True,
               max_samples: int = 3_000):
    """This is the master function to create all postprocessing plots"""
    
    from projection_BNS.utils import TARGET_COLORS_DICT
    
    if "jester" in outdir:
        print("This is a run for a jester target EOS")
        TARGET_COLORS_DICT = {k: v for k, v in TARGET_COLORS_DICT.items() if "jester" in k}
        labels_mapping_dict = {"jester_soft": "Soft",
                               "jester_middle": "Medium",
                               "jester_hard": "Stiff"}
    
        targets_dict = {"jester_soft": {},
                        "jester_middle": {},
                        "jester_hard": {},
                        }
        all_eos_names = list(targets_dict.keys())
        
    else:
        print("This is a run for a default target EOS")
        TARGET_COLORS_DICT = {k: v for k, v in TARGET_COLORS_DICT.items() if "jester" not in k}
        labels_mapping_dict = {k: k for k in TARGET_COLORS_DICT.keys()} # unity mapping
        targets_dict = {"HQC18": {},
                        "SLY230A": {},
                        "MPA1": {},
                        }
        all_eos_names = list(targets_dict.keys())
    
    # Load the targets
    for eos_name in all_eos_names:
        target_filepath = os.path.join(DATA_PATH, "eos", f"{eos_name}.npz")
        target_filepath = np.load(target_filepath)
        m_target, r_target, l_target = target_filepath["masses_EOS"], target_filepath["radii_EOS"], target_filepath["Lambdas_EOS"]
        
        targets_dict[eos_name]["m_target"] = m_target
        targets_dict[eos_name]["r_target"] = r_target
        targets_dict[eos_name]["l_target"] = l_target
    
    # Load the posterior dataset
    filename = os.path.join(outdir, "eos_samples.npz")
    data = np.load(filename)
    EOS_keys = ['E_sym', 'L_sym', 'K_sym', 'K_sat', 'nbreak', 'n_CSE_0_u', 'cs2_CSE_0', 'n_CSE_1_u', 'cs2_CSE_1', 'n_CSE_2_u', 'cs2_CSE_2', 'n_CSE_3_u', 'cs2_CSE_3', 'n_CSE_4_u', 'cs2_CSE_4', 'n_CSE_5_u', 'cs2_CSE_5', 'n_CSE_6_u', 'cs2_CSE_6', 'n_CSE_7_u', 'cs2_CSE_7', 'cs2_CSE_8']
    
    m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
    logpc_EOS = data["logpc_EOS"]
    n, p, e, cs2 = data["n"], data["p"], data["e"], data["cs2"]
    
    n = n / jose_utils.fm_inv3_to_geometric / 0.16
    p = p / jose_utils.MeV_fm_inv3_to_geometric
    e = e / jose_utils.MeV_fm_inv3_to_geometric
    
    pc_EOS = np.exp(logpc_EOS) / jose_utils.MeV_fm_inv3_to_geometric

    nb_samples = np.shape(m)[0]
    print(f"Number of samples: {nb_samples}")

    # Plotting
    samples_kwargs = {"rasterized": True}

    m_min, m_max = 0.975, 3.0
    r_min, r_max = 9.0, 14.0
    l_min, l_max = 2.0, 1e4
    
    # Sample requested number of indices randomly:
    log_prob = data["log_prob"]
    
    # First, we make a simple histogram of the log prob values
    plt.figure()
    plt.hist(log_prob, bins=50, color="blue", histtype="step", lw=2, density=True)
    plt.axvline(np.max(log_prob), color="red", lw=2)
    plt.xlabel("Log prob")
    plt.ylabel("Density")
    plt.savefig(os.path.join(outdir, "log_prob_histogram.pdf"), bbox_inches = "tight")
    plt.close()
    
    # Then do exp
    # log_prob = np.exp(log_prob) # so actually no longer log prob but prob... whatever
    max_log_prob_idx = np.argmax(log_prob)
    indices = np.random.choice(nb_samples, max_samples, replace=False)
    indices = np.append(indices, max_log_prob_idx)
    
    # Normalize log_prob for colormap
    log_prob_norm = (log_prob - np.min(log_prob)) / (np.max(log_prob) - np.min(log_prob))
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    # cmap = sns.color_palette("rocket_r", as_cmap=True)
    # cmap = sns.color_palette("dark:salmon", as_cmap=True)
    cmap = sns.color_palette("light:#e31f26", as_cmap=True)
    log_prob_norm = (log_prob - np.min(log_prob)) / (np.max(log_prob) - np.min(log_prob))
    colors = cmap(log_prob_norm[indices])
    
    print("\n\n\n")
    print(f"Showing the max log prob EOS values:")
    for key in EOS_keys:
        print(f"{key}: {data[key][max_log_prob_idx]}")
    print("\n\n\n")

    plt.subplots(1, 2, figsize=(12, 8))
    print("Creating NS plot . . .")
    bad_counter = 0
    for i, col in zip(indices, colors):

        if any(np.isnan(m[i])) or any(np.isnan(r[i])) or any(np.isnan(l[i])):
            bad_counter += 1
            continue
    
        if any(l[i] < 0):
            bad_counter += 1
            continue
        
        if any((m[i] > 1.0) * (r[i] > 20.0)):
            bad_counter += 1
            continue
        
        # Mass-radius plot
        samples_kwargs["color"] = col
        samples_kwargs["zorder"] = log_prob_norm[i]
        plt.subplot(121)
        plt.plot(r[i], m[i], **samples_kwargs)
        plt.xlim(r_min, r_max)
        plt.ylim(m_min, m_max)
        
        # Pressure as a function of density TODO: rather, plot M-Lambda curves here
        plt.subplot(122)
        plt.plot(m[i], l[i], **samples_kwargs)
        
    # Plot the max log prob values in black alpha = 1.0
    plt.subplot(121)
    plt.plot(r[max_log_prob_idx], m[max_log_prob_idx], color="black", lw=2)
    plt.subplot(122)
    plt.plot(m[max_log_prob_idx], l[max_log_prob_idx], color="black", lw=2)
        
    print(f"Bad counter: {bad_counter}")
    # Beautify the plots a bit
    plt.subplot(121)
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M$ [$M_{\odot}$]")
    plt.xlim(r_min, r_max)
    plt.ylim(m_min, m_max)
    plt.grid(False)
    
    plt.subplot(122)
    plt.xlabel(r"$M$ [$M_{\odot}$]")
    plt.ylabel(r"$\Lambda$")
    plt.xlim(m_min, m_max)
    plt.ylim(l_min, l_max)
    plt.yscale("log")
    plt.grid(False)
    
    # Add colorbar
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(log_prob), vmax=np.max(log_prob)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=[ax1, ax2], orientation='vertical', fraction=0.05, pad=0.02)
    cbar.set_label("Log Probability")
    
    # Plot the targets
    for eos_name in all_eos_names:
        r_target, m_target, l_target = targets_dict[eos_name]["r_target"], targets_dict[eos_name]["m_target"], targets_dict[eos_name]["l_target"]
        color = TARGET_COLORS_DICT[eos_name]
        label = labels_mapping_dict[eos_name]
        
        plt.subplot(121)
        plt.plot(r_target, m_target, color=color, label=label, **TARGET_KWARGS)
        
        plt.subplot(122)
        plt.plot(m_target, l_target, color=color, label=label, **TARGET_KWARGS)
    
    plt.subplot(121)
    plt.legend()
    
    plt.savefig(os.path.join(outdir, "postprocessing_NS.pdf"), bbox_inches = "tight", dpi=300)
    plt.close()
    print("Creating NS plot . . . DONE")
    
    if plot_histograms:
        # TODO: get the injected value of MTOV, R1.4 and plot them on the histograms
        ### Build a histogram of the TOV masses and R1.4 and Lambda1.4 values
        print("Creating histograms . . .")
        
        mtov_list = []
        r14_list = []
        l14_list = []
        ntov_list = []
        p3nsat_list = []
        
        negative_counter = 0
        for i in tqdm.tqdm(range(nb_samples)):
            _m, _r, _l = m[i], r[i], l[i]
            _pc = pc_EOS[i]
            _n, _p, _e, _cs2 = n[i], p[i], e[i], cs2[i]
            
            if any(_l < 0):
                negative_counter += 1
                continue
            
            mtov = np.max(_m)
            r14 = np.interp(1.4, _m, _r)
            l14 = np.interp(1.4, _m, _l)
            
            p3nsat = np.interp(3, _n, _p)
            
            pc_TOV = np.interp(mtov, _m, _pc)
            n_TOV = np.interp(pc_TOV, _p, _n)
            
            # Append all
            mtov_list.append(mtov)
            if mtov > 1.4:
                r14_list.append(r14)
                l14_list.append(l14)
            ntov_list.append(n_TOV)
            p3nsat_list.append(p3nsat)
            
        print(f"Negative counter: {negative_counter}")
        
        bins = 50
        hist_kwargs = dict(histtype="step", lw=2, density = True, bins=bins)
        plt.subplots(2, 2, figsize=(18, 12))
        
        ### MTOV
        plt.subplot(221)
        plt.hist(mtov_list, color="blue", label = "Jester", **hist_kwargs)
        plt.xlabel(r"$M_{\rm TOV}$ [$M_{\odot}$]")
        plt.ylabel("Density")
        
        print(f"MTOV credible interval")
        med, low, high = report_credible_interval(np.array(mtov_list))
        plt.title(r"$M_{\rm TOV}$: " + f"{med:.2f} - {low:.2f} + {high:.2f}")
        
        mtov_list = np.array(mtov_list)
        median = np.median(mtov_list)
        low, high = arviz.hdi(mtov_list, hdi_prob = 0.95)
        low = median - low
        high = high - median

        ### R1.4
        plt.subplot(222)
        r14_list = np.array(r14_list)
        mask = r14_list < 20.0
        r14_list = r14_list[mask]
        
        print(f"R1.4 credible interval")
        med, low, high = report_credible_interval(np.array(r14_list))
            
        plt.hist(r14_list, color="blue", label = "Jester", **hist_kwargs)
        plt.xlabel(r"$R_{1.4}$ [km]")
        plt.ylabel("Density")
        plt.title(r"$R_{1.4}$ [km]: " + f"{med:.2f}-{low:.2f}+{high:.2f}")
        
        ### n_TOV
        plt.subplot(223)
        plt.hist(ntov_list, color="blue", label = "Jester", **hist_kwargs)
        plt.xlabel(r"$n_{\rm{TOV}}$ [$n_{\rm{sat}}$]")
        plt.ylabel("Density")
        
        ntov_list = np.array(ntov_list)
        med, low, high = report_credible_interval(np.array(ntov_list))
        plt.title(r"$n_{\rm{TOV}}$ [$n_{\rm{sat}}$]: " + f"{med:.4f} - {low:.4f} + {high:.4f}")
        
        ### p(3nsat)
        plt.subplot(224)
        plt.hist(p3nsat_list, color="blue", label = "Jester", **hist_kwargs)
        plt.xlabel(r"$p_{3n_{\rm{sat}}}$ [MeV fm$^{-3}$]")
        plt.ylabel("Density")
        plt.legend(fontsize = 24)
        
        print(f"p3nsat credible interval")
        med, low, high  = report_credible_interval(np.array(p3nsat_list))
        plt.title(r"$p_{3n_{\rm{sat}}}$ [MeV fm$^{-3}$]: " + f"{med:.4f} - {low:.4f} + {high:.4f}")

        plt.savefig(os.path.join(outdir, "postprocessing_histograms.pdf"), bbox_inches = "tight")
        plt.close()
        
    # If this is the run where we combine all constraints, then also make the master plot
    if make_master_plot:
        print(f"Making the master plot")
        NB_POINTS = 100 # number of points from which we construct the uncertainty bands
        filename = os.path.join(outdir, "eos_samples.npz")
        
        data = np.load(filename)
        log_prob = data["log_prob"]
        
        m_min = 1.0
        m_max = 2.5
        
        m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
        logpc_EOS = data["logpc_EOS"]
        pc_EOS = np.exp(logpc_EOS) / jose_utils.MeV_fm_inv3_to_geometric
        
        n, p, e, cs2 = data["n"], data["p"], data["e"], data["cs2"]
        n = n / jose_utils.fm_inv3_to_geometric / 0.16
        p = p / jose_utils.MeV_fm_inv3_to_geometric
        e = e / jose_utils.MeV_fm_inv3_to_geometric
        
        # Get the maximum log prob index
        max_log_prob_idx = np.argmax(log_prob)
        
        # Interpolate for the NS quantities
        m_grid = np.linspace(m_min, m_max, NB_POINTS)
        r_interp_array = np.array([np.interp(m_grid, m[i], r[i], left = -1, right = -1) for i in range(nb_samples)]).T
        l_interp_array = np.array([np.interp(m_grid, m[i], l[i], left = -1, right = -1) for i in range(nb_samples)]).T
        
        # Neutron stars:
        plt.subplots(nrows = 1, ncols = 2, figsize=(12, 6))
        arrays = [r_interp_array, l_interp_array]
        for plot_idx in range(2):
            plt.subplot(1, 2, plot_idx + 1)
            interp_array = arrays[plot_idx]
            median_values = []
            low_values = []
            high_values = []
            
            print("np.shape(interp_array)")
            print(np.shape(interp_array))
            
            for i in range(NB_POINTS):
                # Determine, at this grid point of interpolation, the median values and the credible interval
                values_here = interp_array[i]
                mask = values_here > 0
                values_here = values_here[mask]
                
                median = np.median(values_here)
                median_values.append(median)
                
                # TODO: extend this to have multiple credible intervals
                # Use arviz to compute the credible interval
                low, high = arviz.hdi(values_here, hdi_prob = 0.90)
                low_values.append(low)
                high_values.append(high)

            color = "red"
        
            # Now, make the final plot
            if plot_idx == 0:
                # Mass-radius
                m_max_likelihood, r_max_likelihood = m[max_log_prob_idx], r[max_log_prob_idx]
                mask = m_max_likelihood > m_min
                plt.plot(r_max_likelihood[mask], m_max_likelihood[mask], color=color, linestyle = "--", label = "Max likelihood")
                plt.plot(median_values, m_grid, color=color, linestyle = "-", label = "Median")
                plt.fill_betweenx(m_grid, low_values, high_values, color=color, alpha=0.25, )
            else:
                # Mass-tidal deformability
                m_max_likelihood, l_max_likelihood = m[max_log_prob_idx], l[max_log_prob_idx]
                mask = m_max_likelihood > m_min
                plt.plot(m_grid, median_values, color=color, linestyle = "-", label = "Median")
                plt.plot(m_max_likelihood[mask], l_max_likelihood[mask], color=color, linestyle = "--", label = "Max likelihood")
                plt.fill_between(m_grid, low_values, high_values, color=color, alpha=0.25)
        
        # Add the labels here manually
        plt.subplot(121)
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M$ [$M_\odot$]")
        plt.ylim(m_min, m_max)
        plt.xlim(right = 15.0)
        
        # Add the labels here manually
        plt.subplot(122)
        plt.xlabel(r"$M$ [$M_\odot$]")
        plt.ylabel(r"$\Lambda$")
        plt.xlim(m_min, m_max)
        plt.ylim(2, 4e3)
        plt.yscale("log")
        
        # Plot the three target EOS
        for eos_name in all_eos_names:
            r_target, m_target, l_target = targets_dict[eos_name]["r_target"], targets_dict[eos_name]["m_target"], targets_dict[eos_name]["l_target"]
            color = TARGET_COLORS_DICT[eos_name]
            label = labels_mapping_dict[eos_name]
            
            plt.subplot(121)
            plt.plot(r_target, m_target, color=color, label=label, **TARGET_KWARGS)
            
            plt.subplot(122)
            plt.plot(m_target, l_target, color=color, label=label, **TARGET_KWARGS)
        
        # Add legend
        plt.subplot(122)
        plt.legend()
        
        # Save it
        plt.savefig(os.path.join(outdir, "master_plot.pdf"), bbox_inches = "tight")
        plt.close()
        
def main():
    
    outdir = sys.argv[1]
    
    check_convergence(outdir)
    plot_corner(outdir)
    
    print(f"Making plots for {outdir}")
    make_plots(outdir,
               plot_histograms=False,
               make_master_plot=True)
    
if __name__ == "__main__":
    main()