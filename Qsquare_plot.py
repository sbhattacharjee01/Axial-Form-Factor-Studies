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

# Add CCQE filter (mode == -1)
ccqe_mask = (modes == -1)
combined_mask = hydrogen_mask & ccqe_mask
print(f"Total CCQE hydrogen-only entries: {np.sum(combined_mask)}")

# Initialize Q^2 list
q2_values = []

# Process only the first 20000 unique events for debugging
unique_events = np.unique(events)[:5000]

for i, event_idx in enumerate(unique_events):
    if i % 100 == 0:  # Print progress every 100 events
        print(f"Processing event {i + 1}/{len(unique_events)}...")

    # Get indices for this event
    event_mask = events == event_idx

    # Apply combined mask for hydrogen and CCQE
    mask = combined_mask & event_mask

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
    cos_theta = ROOT.TMath.Cos(angle)  # Cosine of the angle

    # Calculate Q^2
    p_mu = muon.Vect().Mag()  # Magnitude of the muon's momentum
    Q2 = 2 * E_nu * (E_mu - p_mu * cos_theta) - muon_mass**2
    q2_values.append(Q2)

# Convert to NumPy array
q2_values = np.array(q2_values)

# Define bins for Q^2
bins = np.logspace(np.log10(1e-3), np.log10(10), num=25)  # Logarithmic bins
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = np.diff(bins)

# Histogram the Q^2 values
hist, _ = np.histogram(q2_values, bins=bins)

# Normalize by bin width
n_per_delta_q2 = hist / bin_widths

# Plot histograms
plt.figure(figsize=(10, 6))

# Raw histogram
plt.step(bin_centers, hist, where="mid", color="blue", label="Raw Events")

# Normalized histogram (N / ΔQ²)
plt.step(bin_centers, n_per_delta_q2, where="mid", color="red", linestyle="--", label=r"$N / \Delta Q^2$")

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$Q^2$ [GeV$^2$]")
plt.ylabel("Counts")
plt.title("Comparison of Event Count and $N / \Delta Q^2$")
plt.legend()
plt.grid()
plt.show()
