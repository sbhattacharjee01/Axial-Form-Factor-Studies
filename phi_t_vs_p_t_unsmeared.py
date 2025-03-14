import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Open the ROOT file
file_path = "/media/sf_Saikat_sharedfolder/NEUT Files/reader_rhc_1e6_rs.root"
root_file = uproot.open(file_path)
tree = root_file["event_tree"]

# Define number of events to process
num_events = 2000000 

# Retrieve relevant branches
events = tree["event"].array(library="np")[:num_events]
modes = tree["mode"].array(library="np")[:num_events]
ibounds = tree["ibound"].array(library="np")[:num_events]
pids = tree["pid"].array(library="np")[:num_events]
pxs = tree["px"].array(library="np")[:num_events]
pys = tree["py"].array(library="np")[:num_events]

# Lists to store δφT and δpT values
PHI_T_H_list, PHI_T_C_list = [], []
P_T_H_list, P_T_C_list = [], []

# Process events
current_event = -1
muon_px, muon_py = None, None
neutron_px, neutron_py = None, None
current_mode = None
current_ibound = None

for i in range(num_events):
    event = events[i]
    mode = modes[i]
    ibound = ibounds[i]
    pid = pids[i]
    px, py = pxs[i], pys[i]

    if event != current_event:
        if current_mode == -1 and muon_px is not None and neutron_px is not None:
            # Compute transverse momenta
            pT_mu = np.sqrt(muon_px**2 + muon_py**2)
            pT_n = np.sqrt(neutron_px**2 + neutron_py**2)

            if pT_mu > 0 and pT_n > 0:  # Avoid division by zero
                dot_product = -(muon_px * neutron_px + muon_py * neutron_py)
                cos_phi_T = dot_product / (pT_mu * pT_n)

                # Ensure valid arccos argument
                if -1 <= cos_phi_T <= 1:
                    PHI_T = np.arccos(cos_phi_T) * 180 / np.pi  # Convert to degrees
                    delta_pT = np.sqrt((muon_px + neutron_px)**2 + (muon_py + neutron_py)**2)

                    # Apply selection cuts: Only δpT < 50 MeV and δφT < 50°
                    if delta_pT < 50 and PHI_T < 50:
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

# **Explicit bin edges**
pT_bins = np.arange(0, 55, 5)  # [0, 5, 10, ..., 50]
phiT_bins = np.arange(0, 55, 5)  # [0, 5, 10, ..., 50]

# Create 2D histograms for selected events
hist_H, _, _ = np.histogram2d(P_T_H_list, PHI_T_H_list, bins=[pT_bins, phiT_bins])
hist_C, _, _ = np.histogram2d(P_T_C_list, PHI_T_C_list, bins=[pT_bins, phiT_bins])

# Compute total event counts per bin
hist_total = hist_H + hist_C

# Compute fractions but **ignore empty bins**
frac_H = np.divide(hist_H, hist_total, out=np.zeros_like(hist_H), where=hist_total > 0)
frac_C = np.divide(hist_C, hist_total, out=np.zeros_like(hist_C), where=hist_total > 0)

# Grid positions for pie charts
x_centers = (pT_bins[:-1] + pT_bins[1:]) / 2
y_centers = (phiT_bins[:-1] + phiT_bins[1:]) / 2

# Set up figure
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(pT_bins[0], pT_bins[-1])
ax.set_ylim(phiT_bins[0], phiT_bins[-1])
ax.set_xticks(x_centers)
ax.set_yticks(y_centers)
ax.grid(True, linestyle="--", alpha=0.6)
ax.set_xlabel(r"$\delta p_T$ [MeV]")
ax.set_ylabel(r"$\delta \phi_T$ [Degrees]")
ax.set_title(r"Fraction of Hydrogen vs Carbon Events")

# Function to draw compact pie charts
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

# Define pie chart colors
colors = ["red", "blue"]  # Red for H, Blue for C

# Plot pies at grid positions **ONLY where events exist**
pie_radius = (x_centers[1] - x_centers[0]) * 0.95  # Ensure touching but no overlap

for i, x in enumerate(x_centers):
    for j, y in enumerate(y_centers):
        if hist_total[i, j] > 0:  # **Only plot pies where events exist**
            sizes = [frac_H[i, j], frac_C[i, j]]
            draw_pie(ax, sizes, colors, (x, y), pie_radius)

# **Legend with colored blobs inside a box**
red_patch = mpatches.Patch(color='red', label='Hydrogen')
blue_patch = mpatches.Patch(color='blue', label='Carbon')

legend = ax.legend(handles=[red_patch, blue_patch], loc='upper right', frameon=True)
legend.get_frame().set_facecolor('white')  # White background
legend.get_frame().set_alpha(0.8)  # Slight transparency for visibility

plt.show()
