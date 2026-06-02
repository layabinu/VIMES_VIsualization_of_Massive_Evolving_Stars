from pathlib import Path

import h5py as h5
import numpy as np
from tulips.blackbody import blackbody_color
from tulips.colormodels import irgb_color, irgb_from_xyz

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = Path(__file__).parent
FRAMES_NPZ = BASE_DIR / "frames_data.npz"
HDF5_PATH = BASE_DIR / "BSE_Detailed_Output_3.h5"
OUTPUT_NPZ = BASE_DIR / "frames_data.npz"


# ------------------------------
# Load HDF5 temperature data
# ------------------------------
def load_hdf5_temperatures(path):
    f = h5.File(str(path), "r")

    mask = f["Record_Type"][()] == 4

    time = f["Time"][()][mask]
    teff1 = f["Teff(1)"][()][mask]
    teff2 = f["Teff(2)"][()][mask]

    f.close()
    return time, teff1, teff2


# ------------------------------
# Interpolation helper
# ------------------------------
def interp_from_hdf5(frame_time, hdf5_time, values):
    """
    Uses exact value if possible, otherwise linear interpolation.
    """
    if frame_time <= hdf5_time[0]:
        return float(values[0])
    if frame_time >= hdf5_time[-1]:
        return float(values[-1])

    i = np.searchsorted(hdf5_time, frame_time)

    if hdf5_time[i] == frame_time:
        return float(values[i])

    t0, t1 = hdf5_time[i - 1], hdf5_time[i]
    v0, v1 = values[i - 1], values[i]

    alpha = (frame_time - t0) / (t1 - t0)
    return (1 - alpha) * v0 + alpha * v1


# ------------------------------
# Temperature → displayable irgb
# ------------------------------
def temperature_to_rgb(T_K):
    """
    Convert effective temperature [K] into a displayable irgb color (0–255).

    Pipeline:
        Teff → blackbody spectrum → XYZ → linear RGB → gamma corrected irgb
    """
    # guard against invalid temperatures
    if T_K <= 0 or not np.isfinite(T_K):
        return irgb_color(0, 0, 0)

    # blackbody → XYZ
    xyz = blackbody_color(T_K)

    # XYZ → displayable irgb (includes clipping + gamma)
    irgb = irgb_from_xyz(xyz)

    return irgb


# ------------------------------
# Main augmentation
# ------------------------------
def add_temperatures_and_rgb(
    hdf5_file=HDF5_PATH, frames_file=FRAMES_NPZ, output_frames_file=OUTPUT_NPZ
):
    print("Loading frames...")
    frames = np.load(frames_file, allow_pickle=True)["frames"].tolist()

    print("Loading HDF5 temperatures...")
    hdf5_time, teff1, teff2 = load_hdf5_temperatures(hdf5_file)

    print("Assigning Teff and RGB values...")
    for f in frames:
        t = f["Time"]

        T1 = interp_from_hdf5(t, hdf5_time, teff1)
        T2 = interp_from_hdf5(t, hdf5_time, teff2)

        f["Teff1"] = float(T1)
        f["Teff2"] = float(T2)

        f["RGB1"] = temperature_to_rgb(T1)
        f["RGB2"] = temperature_to_rgb(T2)

    print(f"Saving augmented frames → {output_frames_file}")
    np.savez_compressed(output_frames_file, frames=frames)

    print("Done.")


# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    add_temperatures_and_rgb()
