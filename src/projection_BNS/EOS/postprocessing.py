# TODO: but reminder that we can fetch the correct EOS and plot it on top with dashes

import numpy as np
import matplotlib.pyplot as plt
import jax
import os
import json
import arviz
import corner
import tqdm
import sys
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float
from jax.scipy.stats import gaussian_kde
import pandas as pd
import seaborn as sns

from joseTOV import utils
import joseTOV.utils as jose_utils

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

EOS_CURVE_COLOR = "darkgreen"
DATA_PATH = "/home/twouters2/projects/projection_BNS_A_plus/src/projection_BNS/data/"

# TODO: change this again, but I am testing now
TARGET_KWARGS = {"zorder": 1e10,
                 "lw": 2,
                 "linestyle": "--"}

def plot_corner(outdir,
                samples,
                keys):
    
    samples = np.reshape(samples, (len(keys), -1))
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
    
    # Load the samples
    data = np.load(os.path.join(outdir, "results_production.npz"))
    
    print("list(data.keys())")
    print(list(data.keys()))

    for key in data.keys():
        if key == "log_prob":
            continue
        print(f"Key: {key}")
        values = data[key]
        values = np.array(values)
        rhat = arviz.rhat(values)
        
        print(f"    rhat = {rhat}")
        
        ess = arviz.ess(values)
        print(f"    ess = {ess}")

def make_plots(outdir: str,
               eos_name: str,
               plot_R_and_p: bool = True,
               plot_histograms: bool = True,
               make_master_plot: bool = True,
               max_samples: int = 3_000):
    
    filename = os.path.join(outdir, "eos_samples.npz")
    
    # Load the target
    target_filepath = os.path.join(DATA_PATH, f"{eos_name}.npz")
    target_filepath = np.load(target_filepath)
    m_target, r_target, l_target = target_filepath["masses_EOS"], target_filepath["radii_EOS"], target_filepath["Lambdas_EOS"]
    
    data = np.load(filename)
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
    samples_kwargs = {"color": "black",
                      "alpha": 1.0,
                      "rasterized": True}

    plt.subplots(1, 2, figsize=(12, 8))

    m_min, m_max = 0.5, 3.0
    r_min, r_max = 8.0, 16.0
    
    # Sample requested number of indices randomly:
    log_prob = data["log_prob"]
    log_prob = np.exp(log_prob) # so actually no longer log prob but prob... whatever
    
    max_log_prob_idx = np.argmax(log_prob)
    indices = np.random.choice(nb_samples, max_samples, replace=False) # p=log_prob/np.sum(log_prob)
    indices = np.append(indices, max_log_prob_idx)

    # Get a colorbar for log prob, but normalized
    norm = plt.Normalize(vmin=np.min(log_prob), vmax=np.max(log_prob))
    cmap = sns.color_palette("crest", as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    if plot_R_and_p:
        print("Creating NS plot . . .")
        bad_counter = 0
        for i in tqdm.tqdm(indices):

            # Get color
            normalized_value = norm(log_prob[i])
            color = cmap(normalized_value)
            samples_kwargs["color"] = color
            samples_kwargs["zorder"] = 1e2 + normalized_value
            
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
            plt.subplot(121)
            plt.plot(r[i], m[i], **samples_kwargs)
            plt.xlim(r_min, r_max)
            plt.ylim(m_min, m_max)
            
            # Pressure as a function of density TODO: rather, plot M-Lambda curves here
            plt.subplot(122)
            plt.plot(m[i], l[i], **samples_kwargs)
            # last_pc = pc_EOS[i, -1]
            # n_TOV = np.interp(last_pc, p[i], n[i])
            # mask = (n[i] > 0.5) * (n[i] < n_TOV)
            # plt.plot(n[i][mask], p[i][mask], **samples_kwargs)
            
        print(f"Bad counter: {bad_counter}")
        # Beautify the plots a bit
        plt.subplot(121)
        plt.plot(r_target, m_target, color="red", label=eos_name, **TARGET_KWARGS)
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M$ [$M_{\odot}$]")
        plt.xlim(r_min, r_max)
        plt.ylim(m_min, m_max)
        plt.legend()
        
        plt.subplot(122)
        plt.plot(m_target, l_target, color="red", label=eos_name, **TARGET_KWARGS)
        plt.xlabel(r"$M$ [$M_{\odot}$]")
        plt.ylabel(r"$\Lambda$")
        plt.xlim(m_min, m_max)
        plt.ylim(0.0, 5000.0)
        
        # Save
        sm.set_array([])
        # Add the colorbar
        fig = plt.gcf()
        # cbar = plt.colorbar(sm, ax=fig.axes)
        # Add a single colorbar at the top spanning both subplots
        cbar_ax = fig.add_axes([0.15, 0.94, 0.7, 0.03])  # [left, bottom, width, height]
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Normalized posterior probability", fontsize = 16)
        cbar.set_ticks([])
        cbar.ax.xaxis.labelpad = 5
        cbar.ax.tick_params(labelsize=0, length=0)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.get_offset_text().set_visible(False)
        cbar.set_label(r"Normalized posterior probability")

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
        
        mass_at_2nat_list = []
        
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
            
            # # TODO: remove? No longer interested in this
            # p_at_2nsat = np.interp(2.0, _n, _p)
            # mass_at_2nat = np.interp(p_at_2nsat, _pc, _m)
            # mass_at_2nat_list.append(mass_at_2nat)
            # report_credible_interval(np.array(mass_at_2nat_list))
            
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
        NB_POINTS = 100
        nmin_grid = 0.5 
        nmax_grid = 8.0
        
        filename = os.path.join(outdir, "eos_samples.npz")
        
        data = np.load(filename)
        log_prob = data["log_prob"]
        
        m_min = 1.0
        m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
        logpc_EOS = data["logpc_EOS"]
        pc_EOS = np.exp(logpc_EOS) / jose_utils.MeV_fm_inv3_to_geometric
        
        n, p, e, cs2 = data["n"], data["p"], data["e"], data["cs2"]
        n = n / jose_utils.fm_inv3_to_geometric / 0.16
        p = p / jose_utils.MeV_fm_inv3_to_geometric
        e = e / jose_utils.MeV_fm_inv3_to_geometric
        
        # Get the maximum log prob index
        max_log_prob_idx = np.argmax(log_prob)
        
        # First comparison plot of max log prob:
        cutoff_mass = 0.6
        plt.subplots(1, 2, figsize=(12, 6))
        plt.subplot(121)
        _r, _m, _l = r[max_log_prob_idx], m[max_log_prob_idx], l[max_log_prob_idx]
        mask_jester = _m > cutoff_mass
        
        plt.plot(_r[mask_jester], _m[mask_jester], color="blue", lw=2)
        # Plot the target as well:
        mask_target = m_target > cutoff_mass
        _m_target = m_target[mask_target]
        _r_target = r_target[mask_target]
        _l_target = l_target[mask_target]
        plt.plot(_r_target, _m_target, color="red", **TARGET_KWARGS)
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M$ [$M_{\odot}$]")
        plt.ylim(bottom = cutoff_mass)
        
        plt.subplot(122)
        plt.plot(_m[mask_jester], _l[mask_jester], color="blue", lw=2)
        plt.plot(_m_target, _l_target, color="red", **TARGET_KWARGS)
        plt.yscale("log")
        plt.xlabel(r"$M$ [$M_{\odot}$]")
        plt.ylabel(r"$\Lambda$")
        plt.xlim(left = cutoff_mass)
        plt.legend()
        
        plt.savefig(os.path.join(outdir, "master_max_log_prob_comparison.pdf"), bbox_inches = "tight")
        plt.close()
        
        # Now, for the combined posteriors plots for EOS and NS, taking inspiration from Fig 26 of Koehn+
        
        # TODO: do subplots, but as test case, let us check out cs2
        n_grid = np.linspace(nmin_grid, nmax_grid, NB_POINTS)
        m_grid = np.linspace(cutoff_mass, 3.0, NB_POINTS)
        
        # Interpolate all EOS cs2 on this n_grid
        cs2_interp_array = np.array([np.interp(n_grid, n[i], cs2[i]) for i in range(nb_samples)]).T
        r_interp_array = np.array([np.interp(m_grid, m[i], r[i], left = -1, right = -1) for i in range(nb_samples)]).T
        
        plt.subplots(nrows = 1, ncols = 2, figsize=(12, 6))
        arrays = [r_interp_array, cs2_interp_array]
        for plot_idx in range(2):
            plt.subplot(1, 2, plot_idx + 1)
            interp_array = arrays[plot_idx]
            median_values = []
            low_values = []
            high_values = []
            
            for i in range(NB_POINTS):
                # Determine median
                values_here = interp_array[i]
                mask = values_here > 0
                values_here = values_here[mask]
                median = np.median(values_here)
                median_values.append(median)
                
                # Use arviz to compute the 90% CI
                low, high = arviz.hdi(values_here, hdi_prob = 0.95)
                low_values.append(low)
                high_values.append(high)
        
            # Now, make the final plot
            if plot_idx == 0:
                m_max, r_max = m[max_log_prob_idx], r[max_log_prob_idx]
                mask = m_max > 0.75
                plt.plot(r_max[mask], m_max[mask], color="blue")
                plt.fill_betweenx(m_grid, low_values, high_values, color="blue", alpha=0.25)
            else:
                cs2_max = cs2_interp_array.T[max_log_prob_idx]
                plt.plot(n_grid, median_values, color="blue")
                plt.plot(n_grid, cs2_max, color="green")
                plt.fill_between(n_grid, low_values, high_values, color="blue", alpha=0.25)
        
        # Add the labels here manually
        plt.subplot(121)
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M$ [$M_\odot$]")
        plt.ylim(bottom = 0.75, top = 2.5)
        
        # Add the labels here manually
        plt.subplot(122)
        plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
        plt.ylabel(r"$c_s^2$")
        plt.axhline(0.33, linestyle = "--", color="black")
        plt.savefig(os.path.join(outdir, "master_plot.pdf"), bbox_inches = "tight")
        plt.close()
        
        ### Check how many cs2 curves are above or below 0.33
        counter_cs2_above_033 = 0
        for i in range(nb_samples):
            mask = n[i] < 4.0
            if np.any(cs2[i][mask] > 0.33):
                counter_cs2_above_033 += 1
        
        print(f"Percentage of EOS samples that are above 0.33: {(counter_cs2_above_033 / nb_samples) * 100:.2f}%")
   
def make_haukeplot(outdir: str,
                   nb_samples: int = 3_000):
    print(f"Going to make the Hauke combination plot for outdir = {outdir}")
    
    filename = os.path.join(outdir, "eos_samples.npz")
    
    data = np.load(filename)
    log_prob = data["log_prob"]
    
    m_min = 1.0
    m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
    
    # Now, make the MR plot as before, color according to nb samples
    log_prob = data["log_prob"]
    log_prob = np.exp(log_prob) # so actually no longer log prob but prob... whatever
    
    max_log_prob_idx = np.argmax(log_prob)
    indices = np.random.choice(len(m), nb_samples, replace=False) # p=log_prob/np.sum(log_prob)
    indices = np.append(indices, max_log_prob_idx)

    # Get a colorbar for log prob, but normalized
    norm = plt.Normalize(vmin=np.min(log_prob), vmax=np.max(log_prob))
    cmap = sns.color_palette("crest", as_cmap=True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    fig, axs = plt.subplots(
        2, 2,
        figsize=(10, 12),
        gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [5, 1]}
    )
    
    main_plot = axs[0, 0]
    mtov_plot = axs[0, 1]
    r14_plot = axs[1, 0]
    
    plt.subplot(221)
    for i in tqdm.tqdm(indices):
        # Get color
        normalized_value = norm(log_prob[i])
        color = cmap(normalized_value)
        samples_kwargs = {"color": color, "alpha": 1.0, "rasterized": True, "zorder": 1e10 + normalized_value}
        
        if any(np.isnan(m[i])) or any(np.isnan(r[i])) or any(np.isnan(l[i])):
            continue
        
        if any(l[i] < 0):
            continue
        
        if any((m[i] > 1.0) * (r[i] > 20.0)):
            continue
        
        mask = m[i] > 0.5
        plt.plot(r[i][mask], m[i][mask], **samples_kwargs)
        
    plt.xlim(9.5, 15)
    plt.ylim(0.5, 3)
    plt.ylabel(r"$M$ [$M_{\odot}$]")
    plt.grid(False)
    
    # Here comes the MTOV histogram plot
    plt.subplot(222)
    mtov_list = []
    r14_list = []
    hist_kwargs = {"color": "blue", 
                   "histtype": "step", 
                   "density": True,
                   "linewidth": 2,
                   "bins": 50}
    
    for i in range(len(m)):
        _m, _r, _l = m[i], r[i], l[i]
        bad_radii = np.any((_m > 1.0) * (_r > 20.0))
        bad_lambdas = np.any((_m > 1.0) * (_l < 0.0))
        if bad_radii or bad_lambdas:
            continue
        else:
            mtov_list.append(np.max(_m))
            r14_list.append(np.interp(1.4, _m, _r))
    plt.hist(mtov_list, orientation = "horizontal", label = r"$M_{\rm{TOV}}$", **hist_kwargs)
    plt.legend()
    plt.grid(False)
    
    plt.subplot(223)
    plt.hist(r14_list, label = r"$R_{1.4}$", **hist_kwargs)
    plt.legend()
    plt.xlabel(r"$R$ [km]")
    plt.grid(False)
    
    r14_plot.sharex(main_plot)
    mtov_plot.sharey(main_plot)
    
    plt.setp(main_plot.get_xticklabels(), visible=False)
    plt.setp(mtov_plot.get_xticklabels(), visible=False)
    plt.setp(mtov_plot.get_yticklabels(), visible=False)
    plt.setp(r14_plot.get_yticklabels(), visible=False)
    
    # Set tick lengths to zero
    mtov_plot.tick_params(axis='x', length=0)
    r14_plot.tick_params(axis='y', length=0)
    
    # Remove the final subplot, adjust spacing, and finally, save
    fig.delaxes(axs[1, 1])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(os.path.join(outdir, "haukeplot.pdf"), bbox_inches = "tight")
    plt.close()
    
def main():
    
    # FIXME: in the end, we should have the EOS name in the outdir and therefore only one argument
    outdir = sys.argv[1]
    eos_name = sys.argv[2]
    
    check_convergence(outdir)
    
    print(f"Making plots for {outdir}")
    make_plots(outdir,
               eos_name,
               plot_R_and_p=True,
               plot_histograms=True,
               make_master_plot=False) # FIXME: this is broken?
    # make_haukeplot(outdir) # not making this now, uninformative
    
if __name__ == "__main__":
    main()