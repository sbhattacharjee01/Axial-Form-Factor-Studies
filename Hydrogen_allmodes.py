import uproot
import ROOT  # Import ROOT for TLorentzVector and Angle functionality
import numpy as np
import matplotlib.pyplot as plt

# Open the ROOT file
file_path = "/media/sf_Saikat_sharedfolder/NEUT Files/reader_rhc_1e6_rs.root"
root_file = uproot.open(file_path)
tree = root_file["event_tree"]

# Retrieve branches from the ROOT file
events = tree["event"].array(library="np")
modes = tree["mode"].array(library="np")
ibounds = tree["ibound"].array(library="np")
pids = tree["pid"].array(library="np")
pxs = tree["px"].array(library="np")
pys = tree["py"].array(library="np")
pzs = tree["pz"].array(library="np")
energies = tree["E"].array(library="np")  # Adding energy ("E") branch

# Convert momentum and energy from MeV to GeV
pxs *= 0.001
pys *= 0.001
pzs *= 0.001
energies *= 0.001

# Constants
muon_mass = 0.105658  # GeV/c^2 (for muons)

# Filter hydrogen-only events (ibound == 0 for free proton)
hydrogen_mask = (ibounds == 0)
print(f"Total hydrogen-only entries: {np.sum(hydrogen_mask)}")

# Initialize Q^2 dictionary for modes
q2_by_mode = {}

# Process only the first 20000 unique events for debugging
unique_events = np.unique(events)[:20000]

for i, event_idx in enumerate(unique_events):
    if i % 100 == 0:  # Print progress every 100 events
        print(f"Processing event {i + 1}/{len(unique_events)}...")

    # Get indices for this event
    event_mask = events == event_idx

    # Apply hydrogen mask for free proton events
    mask = hydrogen_mask & event_mask
    if not np.any(mask):
        continue

    # Incoming neutrino (assume pid == -14 for antineutrino or 14 for neutrino)
    neutrino_mask = (pids[mask] == -14) | (pids[mask] == 14)
    if not np.any(neutrino_mask):
        continue
    px_nu, py_nu, pz_nu, E_nu = pxs[mask][neutrino_mask][0], pys[mask][neutrino_mask][0], pzs[mask][neutrino_mask][0], energies[mask][neutrino_mask][0]

    # Outgoing lepton (assume pid == -13 for muon or 13 for antimuon)
    muon_mask = (pids[mask] == -13) | (pids[mask] == 13)
    if not np.any(muon_mask):
        continue
    px_mu, py_mu, pz_mu, E_mu = pxs[mask][muon_mask][0], pys[mask][muon_mask][0], pzs[mask][muon_mask][0], energies[mask][muon_mask][0]

    # Create TLorentzVectors for neutrino and muon
    nu = ROOT.TLorentzVector(px_nu, py_nu, pz_nu, E_nu)
    muon = ROOT.TLorentzVector(px_mu, py_mu, pz_mu, E_mu)

    # Calculate the angle using ROOT's Angle() function
    angle = nu.Vect().Angle(muon.Vect())  # Angle in radians
    cos_theta = ROOT.TMath.Cos(angle)

    # Calculate Q^2
    p_mu = muon.Vect().Mag()  # Magnitude of the muon's momentum
    Q2 = 2 * E_nu * (E_mu - p_mu * cos_theta) - muon_mass**2

    # Store Q^2 by mode
    mode = modes[event_mask][0]  # Get the mode for the event
    if mode not in q2_by_mode:
        q2_by_mode[mode] = []
    q2_by_mode[mode].append(Q2)

# Plot Q^2 distributions for each mode
plt.figure(figsize=(10, 7))
bins = np.logspace(np.log10(1e-3), np.log10(10), num=25)  # Log-spaced bins

for mode, q2_values in q2_by_mode.items():
    plt.hist(q2_values, bins=bins, histtype="step", label=f"Mode {mode}", linewidth=1.5)

plt.xscale('log')
plt.xlabel(r"$Q^2$ [GeV$^2$]")
plt.ylabel("Number of Events")
plt.title("Q^2 Distribution by Mode (Hydrogen Only)")
plt.legend()
plt.grid()
plt.show()
