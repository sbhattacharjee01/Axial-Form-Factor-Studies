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
events = tree["event"].array(library="np")[:num_events]
modes = tree["mode"].array(library="np")[:num_events]
ibounds = tree["ibound"].array(library="np")[:num_events]
pids = tree["pid"].array(library="np")[:num_events]
pxs = tree["px"].array(library="np")[:num_events]
pys = tree["py"].array(library="np")[:num_events]

# Lists to store δφT values
PHI_T_H_list, PHI_T_C_list = [], []

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
                PHI_T = np.arccos(dot_product / (pT_mu * pT_n)) * 180 / np.pi  # Convert to degrees

                # Store values for Hydrogen or Carbon
                if current_ibound == 0:  
                    PHI_T_H_list.append(PHI_T)
                elif current_ibound == 1:  
                    PHI_T_C_list.append(PHI_T)

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

# Set binning for δφT
bin_edges_PHIT = np.linspace(0, 180, num=50)  

# --- Plot δφT Distribution ---
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(PHI_T_H_list, bins=bin_edges_PHIT, histtype="step", color="red", linewidth=1.5, label=r"$\delta \phi_T$ - Hydrogen")
ax.hist(PHI_T_C_list, bins=bin_edges_PHIT, histtype="step", color="blue", linewidth=1.5, label=r"$\delta \phi_T$ - Carbon")

ax.set_xlabel(r"$\delta \phi_T$ [Degrees]")
ax.set_ylabel("Entries")
ax.set_title(r"$\delta \phi_T$ Distribution for Hydrogen & Carbon (No Smearing)")
ax.legend()
ax.grid()

plt.tight_layout()
plt.show()
