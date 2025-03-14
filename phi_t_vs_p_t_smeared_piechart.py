import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Open the ROOT file
file_path = "/media/sf_Saikat_sharedfolder/NEUT Files/reader_rhc_1e6_rs.root"
root_file = uproot.open(file_path)
tree = root_file["event_tree"]

# Define number of events to process
num_events = (len(tree["event"].array()))

# Retrieve relevant branches
events = tree["event"].array()[:num_events]
modes = tree["mode"].array()[:num_events]
ibounds = tree["ibound"].array()[:num_events]
pids = tree["pid"].array()[:num_events]
pxs = tree["px"].array()[:num_events]
pys = tree["py"].array()[:num_events]
pzs = tree["pz"].array()[:num_events]

# Lists to store ΔφT and ΔpT values
PHI_T_H_list, PHI_T_C_list = [], []
P_T_H_list, P_T_C_list = [], []

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
            # Apply 4% smearing
            smearing_factor = 0.04
            muon_px_smeared, muon_py_smeared, _ = apply_smearing_total(muon_px, muon_py, muon_pz, smearing_factor)

            # Compute smeared transverse momenta
            pT_mu_smeared = np.sqrt(muon_px_smeared**2 + muon_py_smeared**2)
            pT_n = np.sqrt(neutron_px**2 + neutron_py**2)

            if pT_mu_smeared > 0 and pT_n > 0:
                dot_product_smeared = -(muon_px_smeared * neutron_px + muon_py_smeared * neutron_py)
                cos_phi_T_smeared = dot_product_smeared / (pT_mu_smeared * pT_n)

                if -1 <= cos_phi_T_smeared <= 1:
                    PHI_T_smeared = np.arccos(cos_phi_T_smeared) * 180 / np.pi
                    delta_pT_smeared = np.sqrt((muon_px_smeared + neutron_px)**2 + (muon_py_smeared + neutron_py)**2)

                    # Apply ΔφT < 50° cut for pie charts
                    if PHI_T_smeared < 50:
                        if current_ibound == 0:  
                            PHI_T_H_list.append(PHI_T_smeared)
                            P_T_H_list.append(delta_pT_smeared)
                        elif current_ibound == 1:  
                            PHI_T_C_list.append(PHI_T_smeared)
                            P_T_C_list.append(delta_pT_smeared)

        # Reset for new event
        current_event = event
        current_mode = mode
        current_ibound = ibound
        muon_px, muon_py, muon_pz = None, None, None
        neutron_px, neutron_py = None, None

    if mode == -1 and (ibound == 0 or ibound == 1):
        if pid == -13:
            muon_px, muon_py, muon_pz = px, py, pz
        elif pid == 2112:
            neutron_px, neutron_py = px, py  

# Binning setup
bin_width_phiT = 2
bin_edges_phiT = np.arange(0, 35, bin_width_phiT)  # Limit ΔφT to 50°
bin_width_pT = 5
bin_edges_pT = np.arange(0, 55, bin_width_pT)

# **Pie Chart for ΔφT and ΔpT (After Smearing)**
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
ax.set_xlabel(r"$\delta p_T$ [MeV]", fontsize=20)
ax.set_ylabel(r"$\delta \phi_T$ [Degrees]", fontsize=20)
ax.set_title(r"Fraction of Hydrogen vs Carbon Events (After Smearing)", fontsize=20)

# **Function to draw correct small pie charts**
def draw_pie(ax, sizes, colors, xy, pie_radius):
    x, y = xy
    pie_ax = ax.inset_axes(
        [x - pie_radius/2, y - pie_radius/2, pie_radius, pie_radius],
        transform=ax.transData
    )
    pie_ax.pie(sizes, colors=colors, radius=1)
    pie_ax.set_xticks([])
    pie_ax.set_yticks([])
    pie_ax.set_frame_on(False)

# **Correct Pie Chart Scaling**
colors = ["red", "green"]
pie_radius = (x_centers[1] - x_centers[0]) * 0.55  # **Reduced size to prevent overlap**

# **Plot small pies at correct grid positions**
# Swap i and j to correctly place pie charts
for j, x in enumerate(x_centers):  # Loop over p_T bins (x-axis)
    for i, y in enumerate(reversed(y_centers)):  # Reverse order for φ_T bins (y-axis)
        if hist_total[j, i] > 0:  # Use flipped indices
            sizes = [frac_H[j, i], frac_C[j, i]]
            draw_pie(ax, sizes, colors, (x, y), pie_radius)


# **Add Legend for Pie Chart**
red_patch = mpatches.Patch(color='red', label='Hydrogen')
green_patch = mpatches.Patch(color='green', label='Carbon')
ax.legend(handles=[red_patch, green_patch], loc='upper right', fontsize=16)

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
