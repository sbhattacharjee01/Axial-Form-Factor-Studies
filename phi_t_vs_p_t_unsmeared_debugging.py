import uproot
import numpy as np
import matplotlib.pyplot as plt

# Open the ROOT file
file_path = "/media/sf_Saikat_sharedfolder/NEUT Files/reader_rhc_1e6_rs.root"
root_file = uproot.open(file_path)
tree = root_file["event_tree"]

# Define number of events to process
num_events = 2000

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
IBOUND_list = []  # Store ibound for each event

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

                    # Store ibound value for debugging
                    IBOUND_list.append(current_ibound)

                    # **🔹 Debugging Print Statements**
                    print(f"Event {current_event}: ibound = {current_ibound}, pT_mu = {pT_mu:.2f}, pT_n = {pT_n:.2f}, cos(δφT) = {cos_phi_T:.4f}, δφT = {PHI_T:.2f}, δpT = {delta_pT:.2f}")

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

# Debugging: Print first 20 δφT values with ibound
print("\nFirst 20 δφT values (Unsmoothed) with Debugging:\n")
print(f"{'ibound':>7} | {'pT_mu (MeV)':>15} | {'pT_n (MeV)':>15} | {'cos(δφT)':>15} | {'Unsmoothed δφT':>15} | {'δpT (MeV)':>15}")
print("-" * 100)
for i in range(min(20, len(PHI_T_H_list))):
    print(f"{IBOUND_list[i]:>7} | {P_T_H_list[i]:>15.2f} | {P_T_C_list[i]:>15.2f} | {PHI_T_H_list[i]:>15.4f} | {np.arccos(PHI_T_H_list[i]) * 180 / np.pi:>15.2f} | {P_T_H_list[i]:>15.2f}")
