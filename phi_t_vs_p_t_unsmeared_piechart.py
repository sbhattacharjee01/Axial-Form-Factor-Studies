import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Open the ROOT file
file_path = "/media/sf_Saikat_sharedfolder/NEUT Files/reader_rhc_1e6_rs.root"
root_file = uproot.open(file_path)
tree = root_file["event_tree"]

# **Process ALL events**
num_events = int((len(tree["event"].array()))/10.0)
print(f"Total events in file: {num_events}")

# Retrieve relevant branches
events = tree["event"].array()[:num_events]
modes = tree["mode"].array()[:num_events]
ibounds = tree["ibound"].array()[:num_events]
pids = tree["pid"].array()[:num_events]
pxs = tree["px"].array()[:num_events]
pys = tree["py"].array()[:num_events]

# Lists to store ΔφT and ΔpT values
PHI_T_H_list, PHI_T_C_list = [], []
P_T_H_list, P_T_C_list = [], []

# Process events
current_event = -1
muon_px, muon_py = None, None
neutron_px, neutron_py = None, None
current_mode, current_ibound = None, None

for i in range(num_events):
    event = events[i]
    mode = modes[i]
    ibound = ibounds[i]
    pid = pids[i]
    px, py = pxs[i], pys[i]

    # **Print progress every 100k events**
    if i % 100000 == 0 and i > 0:
        print(f"Processed {i}/{num_events} events...")

    if event != current_event:
        if current_mode == -1 and muon_px is not None and neutron_px is not None:
            # Compute transverse momenta
            pT_mu = np.sqrt(muon_px**2 + muon_py**2)
            pT_n = np.sqrt(neutron_px**2 + neutron_py**2)

            if pT_mu > 0 and pT_n > 0:  # Avoid division by zero
                dot_product = -(muon_px * neutron_px + muon_py * neutron_py)
                cos_phi_T = dot_product / (pT_mu * pT_n)

                if -1 <= cos_phi_T <= 1:
                    PHI_T = np.arccos(cos_phi_T) * 180 / np.pi  # Convert to degrees
                    delta_pT = np.sqrt((muon_px + neutron_px)**2 + (muon_py + neutron_py)**2)

                    # Apply ΔφT < 50° cut for pie charts
                    if PHI_T < 50:
                        if current_ibound == 0:  
                            PHI_T_H_list.append(PHI_T)
                            P_T_H_list.append(delta_pT)
                        elif current_ibound == 1:  
                            PHI_T_C_list.append(PHI_T)
                            P_T_C_list.append(delta_pT)

        # Reset for new event
        current_event = event
        current_mode = mode
        current_ibound = ibound
        muon_px, muon_py = None, None
        neutron_px, neutron_py = None, None

    if mode == -1 and (ibound == 0 or ibound == 1):
        if pid == -13:  # Muon
            muon_px, muon_py = px, py  
        elif pid == 2112:  # Neutron
            neutron_px, neutron_py = px, py  

# **Final event count verification**
print(f"Total events processed: {num_events}")

# **Binning setup (unchanged)**
bin_width_phiT = 2
bin_edges_phiT = np.arange(0, 25, bin_width_phiT)  # Limit ΔφT to 50°
bin_width_pT = 5
bin_edges_pT = np.arange(0, 55, bin_width_pT)

# **Histograms for ΔφT and ΔpT**
def plot_hist_with_purity(data_H, data_C, bins, title, xlabel):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    hist_H, bin_edges = np.histogram(data_H, bins=bins)
    hist_C, _ = np.histogram(data_C, bins=bins)
    hist_total = hist_H + hist_C
    purity = np.divide(hist_H, hist_total, out=np.zeros_like(hist_H, dtype=float), where=hist_total > 0)

    axs[0].hist(data_H, bins=bins, color='red', alpha=0.6, label="Hydrogen", histtype='stepfilled')
    axs[0].hist(data_C, bins=bins, color='green', alpha=0.6, label="Carbon", histtype='stepfilled')
    axs[0].step(bin_edges[:-1], hist_total, where='post', linestyle='solid', color='black', linewidth=2, label="Total (H + C)")
    axs[0].set_ylabel("Entries", fontsize=16)
    axs[0].set_title(title, fontsize=16)
    axs[0].legend(fontsize=16)

    axs[1].step(bin_edges[:-1], purity, where='post', linestyle='-', color='blue', linewidth=2, label="H Purity")
    axs[1].set_xlabel(xlabel, fontsize=16)
    axs[1].set_ylabel("Purity", fontsize=16)
    axs[1].legend(loc="upper right", fontsize=16)
    axs[1].grid()

    plt.tight_layout()
    plt.show()

# **Plot Histograms**
plot_hist_with_purity(PHI_T_H_list, PHI_T_C_list, bin_edges_phiT, "ΔφT Before Smearing (ΔφT < 50°)", "ΔφT (degrees)")
plot_hist_with_purity(P_T_H_list, P_T_C_list, bin_edges_pT, "ΔpT Before Smearing", "ΔpT (MeV/c)")

# **Pie Chart Code (UNCHANGED)**
hist_H, _, _ = np.histogram2d(P_T_H_list, PHI_T_H_list, bins=[bin_edges_pT, bin_edges_phiT])
hist_C, _, _ = np.histogram2d(P_T_C_list, PHI_T_C_list, bins=[bin_edges_pT, bin_edges_phiT])
hist_total = hist_H + hist_C
frac_H = np.divide(hist_H, hist_total, out=np.zeros_like(hist_H, dtype=float), where=hist_total > 0)
frac_C = np.divide(hist_C, hist_total, out=np.zeros_like(hist_C, dtype=float), where=hist_total > 0)

x_centers = (bin_edges_pT[:-1] + bin_edges_pT[1:]) / 2
y_centers = (bin_edges_phiT[:-1] + bin_edges_phiT[1:]) / 2

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(bin_edges_pT[0], bin_edges_pT[-1])
ax.set_ylim(bin_edges_phiT[0], bin_edges_phiT[-1])
ax.set_xticks(x_centers)
ax.set_yticks(y_centers)
ax.grid(True, linestyle="--", alpha=0.6)
ax.set_xlabel(r"$\delta p_T$ [MeV]", fontsize=16)
ax.set_ylabel(r"$\delta \phi_T$ [Degrees]", fontsize=16)
ax.set_title(r"Fraction of Hydrogen vs Carbon Events (ΔφT < 50°)", fontsize=16)

def draw_pie(ax, sizes, colors, xy, pie_radius):
    x, y = xy
    pie_ax = ax.inset_axes([x - pie_radius/2, y - pie_radius/2, pie_radius, pie_radius], transform=ax.transData)
    pie_ax.pie(sizes, colors=colors, radius=1)
    pie_ax.set_xticks([])
    pie_ax.set_yticks([])
    pie_ax.set_frame_on(False)

colors = ["red", "green"]
pie_radius = (x_centers[1] - x_centers[0]) * 0.8
for i, x in enumerate(x_centers):
    for j, y in enumerate(y_centers):
        if hist_total[i, j] > 0:
            sizes = [frac_H[i, j], frac_C[i, j]]
            draw_pie(ax, sizes, colors, (x, y), pie_radius)

ax.legend(handles=[mpatches.Patch(color='red', label='Hydrogen'), mpatches.Patch(color='green', label='Carbon')], loc='upper right', fontsize=16)
plt.show()

# **2D Histogram for Hydrogen Fraction (H / (H + C))**
fig, ax = plt.subplots(figsize=(8, 8))

# Compute ratio (H / (H + C)), avoid division by zero
ratio_H = np.divide(hist_H, hist_total, out=np.zeros_like(hist_H, dtype=float), where=hist_total > 0)

# Plot 2D histogram for Hydrogen fraction
X, Y = np.meshgrid(x_centers, y_centers)
c1 = ax.pcolormesh(X, Y, ratio_H.T, cmap="coolwarm", shading='auto', vmin=0, vmax=1)

# Labels, title, and colorbar
ax.set_xlabel(r"$\delta p_T$ [MeV]", fontsize=16)
ax.set_ylabel(r"$\delta \phi_T$ [Degrees]", fontsize=16)
ax.set_title(r"Hydrogen Fraction: H / (H + C)", fontsize=16)
fig.colorbar(c1, ax=ax, label="Fraction of Hydrogen")

# Save the figure
plt.savefig("hydrogen_fraction_2D.png", dpi=300)
plt.close()

# **2D Histogram for Carbon Fraction (C / (H + C))**
fig, ax = plt.subplots(figsize=(8, 8))

# Compute ratio (C / (H + C)), avoid division by zero
ratio_C = np.divide(hist_C, hist_total, out=np.zeros_like(hist_C, dtype=float), where=hist_total > 0)

# Plot 2D histogram for Carbon fraction
c2 = ax.pcolormesh(X, Y, ratio_C.T, cmap="coolwarm", shading='auto', vmin=0, vmax=1)

# Labels, title, and colorbar
ax.set_xlabel(r"$\delta p_T$ [MeV]", fontsize=16)
ax.set_ylabel(r"$\delta \phi_T$ [Degrees]", fontsize=16)
ax.set_title(r"Carbon Fraction: C / (H + C)", fontsize=16)
fig.colorbar(c2, ax=ax, label="Fraction of Carbon")

# Save the figure
plt.savefig("carbon_fraction_2D.png", dpi=300)
plt.close()

# Compute fraction (H / (H + C)), avoid division by zero
ratio_H = np.divide(hist_H, hist_total, out=np.zeros_like(hist_H, dtype=float), where=hist_total > 0)
ratio_C = np.divide(hist_C, hist_total, out=np.zeros_like(hist_C, dtype=float), where=hist_total > 0)

# Print values
print("\n### Hydrogen Fraction (H / (H + C)) ###")
for i, x in enumerate(x_centers):
    for j, y in enumerate(y_centers):
        if hist_total[i, j] > 0:  # Only print non-empty bins
            print(f"δpT: {x:.2f} MeV, δφT: {y:.2f} degrees → H fraction: {ratio_H[i, j]:.3f}")

print("\n### Carbon Fraction (C / (H + C)) ###")
for i, x in enumerate(x_centers):
    for j, y in enumerate(y_centers):
        if hist_total[i, j] > 0:  # Only print non-empty bins
            print(f"δpT: {x:.2f} MeV, δφT: {y:.2f} degrees → C fraction: {ratio_C[i, j]:.3f}")

