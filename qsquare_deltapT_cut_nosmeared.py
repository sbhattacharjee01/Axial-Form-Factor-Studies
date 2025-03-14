import uproot
import ROOT
import numpy as np
import matplotlib.pyplot as plt

# Open the ROOT file
file_path = "/media/sf_Saikat_sharedfolder/AxFF Files/axFF_models/rhc_SimpleDipole.root"
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
pzs = tree["pz"].array(library="np")[:num_events]
energies = tree["E"].array(library="np")[:num_events]

# Constants
muon_mass = 0.105658  # GeV/c^2

# Lists to store Q² values and ΔpT values (After Cut)
Q2_H_list, Q2_C_list = [], []
delta_pT_H_list, delta_pT_C_list = [], []

# Process events
current_event = -1
muon_px, muon_py, muon_pz, muon_E = None, None, None, None
neutron_px, neutron_py = None, None
nu_E = None  # Neutrino energy
current_mode = None
current_ibound = None

for i in range(num_events):
    event = events[i]
    mode = modes[i]
    ibound = ibounds[i]
    pid = pids[i]
    px, py, pz, E = pxs[i], pys[i], pzs[i], energies[i]

    if event != current_event:
        if current_mode == -1 and muon_px is not None and neutron_px is not None and nu_E is not None:
            # Compute ΔpT
            delta_pxT = muon_px + neutron_px
            delta_pyT = muon_py + neutron_py
            delta_pT = np.sqrt(delta_pxT**2 + delta_pyT**2)

            # Apply ΔpT < 1200 MeV cut
            if delta_pT < 1200:
                # Compute total muon momentum
                muon_p = np.sqrt(muon_px**2 + muon_py**2 + muon_pz**2)  

                # Create TLorentzVectors
                nu_vec = ROOT.TLorentzVector(0, 0, nu_E, nu_E)  # Assume neutrino moving in z-direction
                mu_vec = ROOT.TLorentzVector(muon_px, muon_py, muon_pz, muon_E)

                # Compute angle between neutrino and muon
                cos_theta = ROOT.TMath.Cos(nu_vec.Vect().Angle(mu_vec.Vect()))

                # Compute Q²
                Q2 = (2 * nu_E * (muon_E - muon_p * cos_theta) - muon_mass**2) * 1.0e-06  # Convert to GeV²

                # Store values for Hydrogen or Carbon
                if Q2 > 0:  # Avoid unphysical values
                    if current_ibound == 0:  # Hydrogen
                        Q2_H_list.append(Q2)
                        delta_pT_H_list.append(delta_pT)  # Store ΔpT for Hydrogen
                    elif current_ibound == 1:  # Carbon
                        Q2_C_list.append(Q2)
                        delta_pT_C_list.append(delta_pT)  # Store ΔpT for Carbon

        # Reset for new event
        current_event = event
        current_mode = mode
        current_ibound = ibound
        muon_px, muon_py, muon_pz, muon_E = None, None, None, None
        neutron_px, neutron_py = None, None
        nu_E = None  

    if mode == -1 and (ibound == 0 or ibound == 1):
        if pid == -14:  # Neutrino
            nu_E = E  

        if pid == -13:  # Muon
            p_mu = np.sqrt(px**2 + py**2 + pz**2)  
            if p_mu > 100:  
                muon_px, muon_py, muon_pz, muon_E = px, py, pz, E  

        elif pid == 2112:  # Neutron
            neutron_px, neutron_py = px, py  

# Set binning for Q² (Log Spacing)
bin_edges_Q2 = np.logspace(np.log10(1e-3), np.log10(10), num=30)  
bin_edges_pT = np.linspace(0, 1200, num=50)  # Linear bins for ΔpT

# --- Plot Q² and ΔpT ---
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Q² Plot
axs[0].hist(Q2_H_list, bins=bin_edges_Q2, histtype="step", color="red", linewidth=1.5, label="Q² - Hydrogen")
axs[0].hist(Q2_C_list, bins=bin_edges_Q2, histtype="step", color="blue", linewidth=1.5, label="Q² - Carbon")
axs[0].set_xscale('log')
axs[0].set_xlabel(r"$Q^2$ [GeV$^2$]")
axs[0].set_ylabel("Entries")
axs[0].set_title("Q² Distribution for Hydrogen & Carbon")
axs[0].legend()
axs[0].grid()

# ΔpT Plot
axs[1].hist(delta_pT_H_list, bins=bin_edges_pT, histtype="step", color="red", linewidth=1.5, label=r"$\delta p_T$ - Hydrogen")
axs[1].hist(delta_pT_C_list, bins=bin_edges_pT, histtype="step", color="blue", linewidth=1.5, label=r"$\delta p_T$ - Carbon")
axs[1].set_xlabel(r"$\delta p_T$ [MeV/c]")
axs[1].set_ylabel("Entries")
axs[1].set_title(r"$\delta p_T$ Distribution for Hydrogen & Carbon")
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()
