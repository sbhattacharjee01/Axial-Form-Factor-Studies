import uproot
import ROOT
import numpy as np
import matplotlib.pyplot as plt

# Open the ROOT file
file_path = "/media/sf_Saikat_sharedfolder/AxFF Files/axFF_models/rhc_Zexp.root"
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
muon_mass = 0.105658  # GeV/c^2 (for muons)
smearing_factor_H = 0.04  # Hydrogen smearing
smearing_factor_C = 0.04  # Carbon smearing

# Lists to store Q^2 values
Q2_H_list = []
Q2_C_list = []

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
            # Compute delta pT (before smearing)
            delta_pxT = muon_px + neutron_px
            delta_pyT = muon_py + neutron_py
            delta_pT = np.sqrt(delta_pxT**2 + delta_pyT**2)

            # Apply smearing (Using different factors if needed)
            smearing_factor = smearing_factor_H if current_ibound == 0 else smearing_factor_C  
            px_mu_smeared, py_mu_smeared, pz_mu_smeared = apply_smearing_total(muon_px, muon_py, muon_pz, smearing_factor)

            # Compute delta pT (after smearing)
            delta_pxT_smeared = px_mu_smeared + neutron_px
            delta_pyT_smeared = py_mu_smeared + neutron_py
            delta_pT_smeared = np.sqrt(delta_pxT_smeared**2 + delta_pyT_smeared**2)

            # Apply delta pT < 50 MeV cut
            if delta_pT_smeared < 50:
                # Compute Q^2
                muon_p_smeared = np.sqrt(px_mu_smeared**2 + py_mu_smeared**2 + pz_mu_smeared**2)  # Smeared total muon momentum
                
                # Create TLorentzVectors
                nu_vec = ROOT.TLorentzVector(0, 0, nu_E, nu_E)  # Assume neutrino moving in z-direction
                mu_vec = ROOT.TLorentzVector(px_mu_smeared, py_mu_smeared, pz_mu_smeared, muon_E)

                # Compute angle between neutrino and muon
                cos_theta = ROOT.TMath.Cos(nu_vec.Vect().Angle(mu_vec.Vect()))

                # Compute Q^2
                Q2 = (2 * nu_E * (muon_E - muon_p_smeared * cos_theta) - muon_mass**2)*1.0e-06

                # Store Q^2 for Hydrogen or Carbon
                if Q2 > 0:  # Skip unphysical values
                    if current_ibound == 0:  # Hydrogen
                        Q2_H_list.append(Q2)
                    elif current_ibound == 1:  # Carbon
                        Q2_C_list.append(Q2)

                    # Print first few values for verification
                    if len(Q2_H_list) < 10:
                        print(f"Event {current_event}:")
                        print(f"  ΔpT After Smearing: {delta_pT_smeared:.2f} MeV")
                        print(f"  Q^2: {Q2:.3f} GeV^2")
                        print(f"  Target: {'Hydrogen' if current_ibound == 0 else 'Carbon'}\n")

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

# --- Plot Q^2 Distribution ---
bin_edges = np.logspace(np.log10(1e-3), np.log10(10), num=30)  # Log-spaced bins

plt.figure(figsize=(10, 6))
plt.hist(Q2_H_list, bins=bin_edges, histtype="step", color="red", linewidth=1.5, label="Q^2 - Hydrogen (4% smeared)")
plt.hist(Q2_C_list, bins=bin_edges, histtype="step", color="blue", linewidth=1.5, label="Q^2 - Carbon (4% smeared)")
plt.xscale('log')
plt.xlabel(r"$Q^2$ [GeV$^2$]")
plt.ylabel("Entries")
plt.title("Q^2 Distribution for Hydrogen & Carbon (ΔpT < 50 MeV)")
plt.legend()
plt.grid()
plt.show()
