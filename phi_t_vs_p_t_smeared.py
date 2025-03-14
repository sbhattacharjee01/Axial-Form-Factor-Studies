import uproot
import numpy as np
import matplotlib.pyplot as plt

# Open the ROOT file
file_path = "/media/sf_Saikat_sharedfolder/NEUT Files/reader_rhc_1e6_rs.root"
root_file = uproot.open(file_path)
tree = root_file["event_tree"]

# Define number of events to process
num_events = 200000  

# Retrieve relevant branches
events = tree["event"].array()[:num_events]
modes = tree["mode"].array()[:num_events]
ibounds = tree["ibound"].array()[:num_events]
pids = tree["pid"].array()[:num_events]
pxs = tree["px"].array()[:num_events]
pys = tree["py"].array()[:num_events]
pzs = tree["pz"].array()[:num_events]

# Lists to store values
delta_phi_T_H_before, delta_phi_T_C_before = [], []
delta_phi_T_H_after, delta_phi_T_C_after = [], []
delta_pT_H_before, delta_pT_C_before = [], []
delta_pT_H_after, delta_pT_C_after = [], []

# Function to apply smearing
def apply_smearing_total(px, py, pz, factor):
    p_mu = np.sqrt(px**2 + py**2 + pz**2)
    smear_scale = factor * p_mu
    return (
        px + np.random.normal(0, smear_scale),
        py + np.random.normal(0, smear_scale),
        pz + np.random.normal(0, smear_scale)
    )

# Process events
current_event = -1
muon_px, muon_py, muon_pz = None, None, None
neutron_px, neutron_py = None, None
current_mode, current_ibound = None, None

for i in range(num_events):
    event = events[i]
    mode = modes[i]
    ibound = ibounds[i]
    pid = pids[i]
    px, py, pz = pxs[i], pys[i], pzs[i]

    if event != current_event:
        if current_mode == -1 and muon_px is not None and neutron_px is not None:
            # Compute delta phi T and delta pT before smearing
            pT_mu = np.sqrt(muon_px**2 + muon_py**2)
            pT_n = np.sqrt(neutron_px**2 + neutron_py**2)

            if pT_mu > 0 and pT_n > 0:
                dot_product = -(muon_px * neutron_px + muon_py * neutron_py)
                cos_phi_T = dot_product / (pT_mu * pT_n)
                if -1 <= cos_phi_T <= 1:
                    phi_T = np.arccos(cos_phi_T) * 180 / np.pi
                    delta_pT = np.sqrt((muon_px + neutron_px)**2 + (muon_py + neutron_py)**2)

                    # Store in respective lists
                    if current_ibound == 0:
                        delta_phi_T_H_before.append(phi_T)
                        delta_pT_H_before.append(delta_pT)
                    elif current_ibound == 1:
                        delta_phi_T_C_before.append(phi_T)
                        delta_pT_C_before.append(delta_pT)

            # Apply smearing
            smearing_factor = 0.04
            muon_px_smeared, muon_py_smeared, _ = apply_smearing_total(muon_px, muon_py, muon_pz, smearing_factor)

            # Compute delta phi T and delta pT after smearing
            pT_mu_smeared = np.sqrt(muon_px_smeared**2 + muon_py_smeared**2)
            dot_product_smeared = -(muon_px_smeared * neutron_px + muon_py_smeared * neutron_py)
            cos_phi_T_smeared = dot_product_smeared / (pT_mu_smeared * pT_n)

            delta_pT_smeared = np.sqrt((muon_px_smeared + neutron_px)**2 + (muon_py_smeared + neutron_py)**2)

            if -1 <= cos_phi_T_smeared <= 1:
                phi_T_smeared = np.arccos(cos_phi_T_smeared) * 180 / np.pi

                # Store in respective lists
                if current_ibound == 0:
                    delta_phi_T_H_after.append(phi_T_smeared)
                    delta_pT_H_after.append(delta_pT_smeared)
                elif current_ibound == 1:
                    delta_phi_T_C_after.append(phi_T_smeared)
                    delta_pT_C_after.append(delta_pT_smeared)

        # Reset for new event
        current_event = event
        current_mode = mode
        current_ibound = ibound
        muon_px, muon_py, muon_pz = None, None, None
        neutron_px, neutron_py = None, None

    if mode == -1 and (ibound == 0 or ibound == 1):
        if pid == -13:
            p_mu = np.sqrt(px**2 + py**2 + pz**2)
            if p_mu > 100:
                muon_px, muon_py, muon_pz = px, py, pz
        elif pid == 2112:
            neutron_px, neutron_py = px, py

# Binning
bin_width_phiT = 2
bin_edges_phiT = np.arange(0, 180, bin_width_phiT)
bin_width_pT = 5
bin_edges_pT = np.arange(0, max(max(delta_pT_H_after, default=0), max(delta_pT_C_after, default=0)) + bin_width_pT, bin_width_pT)

# Function to plot histograms with purity
def plot_hist_with_purity(data_H, data_C, bins, title, xlabel):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Compute histograms
    hist_H, bin_edges = np.histogram(data_H, bins=bins)
    hist_C, _ = np.histogram(data_C, bins=bins)
    hist_total = hist_H + hist_C

    # Compute purity
    purity = np.divide(hist_H, hist_total, where=hist_total > 0)

    # **Stacked histogram ensures total matches sum**
    axs[0].hist(data_H, bins=bins, color='red', alpha=0.6, label="Hydrogen", histtype='stepfilled')
    axs[0].hist(data_C, bins=bins, color='green', alpha=0.6, label="Carbon", histtype='stepfilled')

    # **Black step line ensures total is correctly overlaid**
    axs[0].step(bin_edges[:-1], hist_total, where='post', linestyle='solid', color='black', linewidth=2, label="Total (H + C)")
    axs[0].set_ylabel("Entries", fontsize=16)
    axs[0].set_title(title)
    axs[0].legend(fontsize=16)

    # **Hydrogen purity plot**
    axs[1].step(bin_edges[:-1], purity, where='post', linestyle='-', color='blue', linewidth=2, label="H Purity")
    axs[1].set_xlabel(xlabel, fontsize=16)
    axs[1].set_ylabel("Purity", fontsize=16)
    axs[1].legend(loc="upper right", fontsize=16)
    axs[1].grid()

    plt.tight_layout()
    plt.show()

# **Generate 4 Separate Plots**
plot_hist_with_purity(delta_phi_T_H_before, delta_phi_T_C_before, bin_edges_phiT, "ΔφT Before Smearing", "ΔφT (degrees)")
plot_hist_with_purity(delta_phi_T_H_after, delta_phi_T_C_after, bin_edges_phiT, "ΔφT After Smearing", "ΔφT (degrees)")
plot_hist_with_purity(delta_pT_H_before, delta_pT_C_before, bin_edges_pT, "ΔpT Before Smearing", "ΔpT (MeV/c)")
plot_hist_with_purity(delta_pT_H_after, delta_pT_C_after, bin_edges_pT, "ΔpT After Smearing", "ΔpT (MeV/c)")
