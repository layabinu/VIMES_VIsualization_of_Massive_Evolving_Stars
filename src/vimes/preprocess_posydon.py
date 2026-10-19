

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

ASE_DIR = Path(__file__).parent
HDF5_PATH = BASE_DIR / '1e+00_Zsun_population.h5'
OUTPUT_FRAMES_FILE = BASE_DIR / "frames_data.npz"

# binary to animate
BINARY_ID = 392
#48 for mass transfer

FRAMES_PER_PHASE = 100
INTERPOLATED_FRAMES_FOR_JUMP = 50
JUMP_THRESHOLD_RATIO = 1.2
JUMP_THRESHOLD_ABS = 50.0
MT_PADDING_FRAMES = 60

MT_RATE_THRESHOLD = -98.0


G = 6.67430e-11        # m^3 kg^-1 s^-2
M_SUN = 1.98847e30     # kg
R_SUN = 6.957e8        # m
DAY = 86400.0          # s


#helpers
def decode_bytes_df(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x
            )
    return df


def decode_bytes_series(s):
    s = s.copy()
    for k in s.index:
        if isinstance(s[k], (bytes, np.bytes_)):
            s[k] = s[k].decode("utf-8")
    return s


def period_to_separation_rsun(period_days, m1_msun, m2_msun):
    period_days = np.asarray(period_days, dtype=float)
    m1_msun = np.asarray(m1_msun, dtype=float)
    m2_msun = np.asarray(m2_msun, dtype=float)

    a_rsun = np.full(period_days.shape, np.nan, dtype=float)

    mask = (
        np.isfinite(period_days) &
        np.isfinite(m1_msun) &
        np.isfinite(m2_msun) &
        (period_days > 0)
    )

    if np.any(mask):
        P_sec = period_days[mask] * DAY
        M_tot = (m1_msun[mask] + m2_msun[mask]) * M_SUN
        a_m = (G * M_tot * P_sec**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)
        a_rsun[mask] = a_m / R_SUN

    return a_rsun


def detect_phases_indices(stellar_type_1, stellar_type_2):
    n = len(stellar_type_1)
    phases = []
    start = 0
    for i in range(1, n):
        if (
            stellar_type_1[i] != stellar_type_1[i - 1]
            or stellar_type_2[i] != stellar_type_2[i - 1]
        ):
            phases.append((start, i - 1))
            start = i
    phases.append((start, n - 1))
    return phases


def detect_mt_starts_from_rate(lg_mtransfer_rate, threshold=MT_RATE_THRESHOLD):
    starts = set()
    mt_on = np.isfinite(lg_mtransfer_rate) & (lg_mtransfer_rate > threshold)

    for i in range(1, len(mt_on)):
        if (not mt_on[i - 1]) and mt_on[i]:
            starts.add(i)

    return starts


def sample_indices(start, end, n):
    if start == end:
        return [float(start)] * n
    return np.linspace(start, end, n).tolist()


def interp(data, idx):
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)

    if len(arr) == 0:
        return np.nan

    filled = pd.Series(arr).interpolate(method="linear", limit_direction="both")
    filled = filled.ffill().bfill().to_numpy(dtype=float)

    if abs(idx - round(idx)) < 1e-12:
        return float(filled[int(round(idx))])

    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    frac = idx - lo
    if lo < 0 or hi >= len(filled):
        return float(filled[min(max(lo, 0), len(filled) - 1)])
    return (1.0 - frac) * float(filled[lo]) + frac * float(filled[hi])


def interp_string(data, idx):
    return data[int(round(idx))]


def convert_posydon_time_to_myr(time_values):
    time_values = np.asarray(time_values, dtype=float)
    converted = np.full(time_values.shape, np.nan, dtype=float)
    mask = np.isfinite(time_values)
    if np.any(mask):
        converted[mask] = time_values[mask] / 1_000_000.0
    return converted


def convert_posydon_log_radius_to_rsun(log_radius_values):
    log_radius_values = np.asarray(log_radius_values, dtype=float)
    converted = np.full(log_radius_values.shape, np.nan, dtype=float)
    mask = np.isfinite(log_radius_values)
    if np.any(mask):
        converted[mask] = 10.0 ** log_radius_values[mask]
    return converted


def normalize_for_animation_display(history_df, target_max_radius=20.0, target_max_orbit=300.0):
    normalized = {}

    for key in ["Radius(1)", "Radius(2)"]:
        values = np.asarray(history_df[key], dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            normalized[key] = np.zeros_like(values, dtype=float)
            continue

        values_finite = values[finite]
        max_value = np.nanmax(values_finite)
        if max_value <= 0:
            scaled = np.zeros_like(values, dtype=float)
        else:
            scaled = target_max_radius * (values / max_value)
        scaled = np.where(finite, scaled, np.nan)
        normalized[key] = scaled

    for key in ["SemiMajorAxis"]:
        values = np.asarray(history_df[key], dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            normalized[key] = np.zeros_like(values, dtype=float)
            continue

        values_finite = values[finite]
        max_value = np.nanmax(values_finite)
        if max_value <= 0:
            scaled = np.zeros_like(values, dtype=float)
        else:
            scaled = target_max_orbit * (values / max_value)
        scaled = np.where(finite, scaled, np.nan)
        normalized[key] = scaled

    return normalized


def map_posydon_stellar_state(state):
    if state is None:
        return "unknown"

    state_str = str(state).strip()
    if not state_str:
        return "unknown"

    mapping = {
        "H-rich_Core_H_burning": "MS",
        "H-rich_Core_He_burning": "HG",
        "H-rich_Core_He_depleted": "FGB",
        "H-rich_Core_C_burning": "CHeB",
        "H-rich_Thermally_Pulsing_AGB": "TPAGB",
        "H-rich_Early_AGB": "EAGB",
        "stripped_He_Core_He_burning": "HeMS",
        "stripped_He_Core_He_depleted": "HeHG",
        "stripped_He_Core_C_depleted": "HeGB",
        "stripped_He_Core_C_burning": "HeGB",
        "stripped_He_Core_C_depleted": "HeGB",
        "stripped_CO_Core_He_burning": "COWD",
        "stripped_CO_Core_He_depleted": "COWD",
        "stripped_CO_Core_C_depleted": "ONeWD",
        "stripped_CO_Core_C_burning": "ONeWD",
        "WD": "HeWD",
        "CO_WD": "COWD",
        "ONe_WD": "ONeWD",
        "NS": "NS",
        "BH": "BH",
    }

    return mapping.get(state_str, "unknown")


def detect_large_jump(
    v1, v2, threshold_ratio=JUMP_THRESHOLD_RATIO, threshold_abs=JUMP_THRESHOLD_ABS
):
    if not np.isfinite(v1) or not np.isfinite(v2):
        return False

    if v1 <= 0 or v2 <= 0:
        return abs(v1 - v2) >= threshold_abs

    if max(v1, v2) / min(v1, v2) >= threshold_ratio:
        return True

    return abs(v1 - v2) >= threshold_abs




def make_event_string(idx, history_df, metallicity=None):
    if idx == 0:
        if metallicity is None or not np.isfinite(metallicity):
            return "Zero-age main-sequence"
        return f"Zero-age main-sequence, Z = {metallicity:.4f}"

    s1 = history_df["Stellar_Type(1)"].iloc[idx]
    s2 = history_df["Stellar_Type(2)"].iloc[idx]
    return f"Phase: {s1} + {s2}"

def load_posydon_binary(hdf5_path, binary_id):
    hist = pd.read_hdf(hdf5_path, key="history")
    oneline = pd.read_hdf(hdf5_path, key="oneline")

    hist = decode_bytes_df(hist)
    oneline = decode_bytes_df(oneline)

    if binary_id not in hist.index:
        raise ValueError(f"Binary ID {binary_id} not found in history.")

    one = hist.loc[binary_id].copy()
    if isinstance(one, pd.Series):
        one = one.to_frame().T
    one = one.reset_index(drop=True)

    one["Time"] = convert_posydon_time_to_myr(one["time"].values)

    one["Radius(1)"] = convert_posydon_log_radius_to_rsun(one["S1_log_R"].values)
    one["Radius(2)"] = convert_posydon_log_radius_to_rsun(one["S2_log_R"].values)

    one["SemiMajorAxis"] = period_to_separation_rsun(
        one["orbital_period"].values,
        one["S1_mass"].values,
        one["S2_mass"].values,
    )

    history_df = pd.DataFrame({
        "Time": one["Time"].astype(float).values,
        "SemiMajorAxis": one["SemiMajorAxis"].astype(float).values,
        "Eccentricity": one["eccentricity"].astype(float).values,
        "Radius(1)": one["Radius(1)"].astype(float).values,
        "Radius(2)": one["Radius(2)"].astype(float).values,
        "Mass(1)": one["S1_mass"].astype(float).values,
        "Mass(2)": one["S2_mass"].astype(float).values,
        "lg_mtransfer_rate": one["lg_mtransfer_rate"].astype(float).values,
        "Stellar_Type(1)": one["S1_state"].astype(str).values,
        "Stellar_Type(2)": one["S2_state"].astype(str).values,
    })

    normalized = normalize_for_animation_display(
        history_df,
        target_max_radius=15.0,
        target_max_orbit=250.0,
    )
    for key in ["Radius(1)", "Radius(2)", "SemiMajorAxis"]:
        history_df[key] = normalized[key]

    meta = {}

    if binary_id in oneline.index:
        row = oneline.loc[binary_id].copy()
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        row = decode_bytes_series(row)

        if "metallicity" in row.index:
            meta["metallicity"] = float(row["metallicity"])

        mt_cols = [c for c in oneline.columns if "mt_history" in c]
        for c in mt_cols:
            meta[c] = row[c]

    return history_df, meta


def summarize_history_df(history_df):
    print("Quantity summary (for animation):")
    for key in [
        "Time",
        "SemiMajorAxis",
        "Eccentricity",
        "Radius(1)",
        "Radius(2)",
        "Mass(1)",
        "Mass(2)",
    ]:
        if key in history_df.columns:
            values = np.asarray(history_df[key], dtype=float)
            finite = np.isfinite(values)
            if np.any(finite):
                print(
                    f"  {key}: min={np.nanmin(values):.6g}, max={np.nanmax(values):.6g}, "
                    f"finite={int(finite.sum())}/{len(values)}"
                )
            else:
                print(f"  {key}: no finite values")


#main loop
def preprocess_to_frames(hdf5_path, out_path, binary_id):
    print("Loading POSYDON binary...")
    history_df, meta = load_posydon_binary(hdf5_path, binary_id)
    summarize_history_df(history_df)

    phases = detect_phases_indices(
        history_df["Stellar_Type(1)"].values,
        history_df["Stellar_Type(2)"].values,
    )

    mt_starts = detect_mt_starts_from_rate(
        history_df["lg_mtransfer_rate"].values,
        threshold=MT_RATE_THRESHOLD
    )

    metallicity = meta.get("metallicity", np.nan)

    frames = []
    last_real_time = None

    for pidx, (start, end) in enumerate(phases):
        positions = sample_indices(start, end, FRAMES_PER_PHASE)
        sampled = []

        for pos in positions:
            frame = {}

            for k in [
                "Time",
                "SemiMajorAxis",
                "Eccentricity",
                "Radius(1)",
                "Radius(2)",
                "Mass(1)",
                "Mass(2)",
            ]:
                frame[k] = interp(history_df[k].values, pos)

            # make things look like the compas stuff
            raw_stype1 = interp_string(history_df["Stellar_Type(1)"].values, pos)
            raw_stype2 = interp_string(history_df["Stellar_Type(2)"].values, pos)
            frame["stypeName1"] = map_posydon_stellar_state(raw_stype1)
            frame["stypeName2"] = map_posydon_stellar_state(raw_stype2)

            idx_round = int(round(pos))
            frame["eventString"] = make_event_string(idx_round, history_df, metallicity)

            sampled.append(frame)

        if sampled:
            last_real_time = sampled[-1]["Time"]

        enhanced = []
        for i in range(len(sampled) - 1):
            a, b = sampled[i], sampled[i + 1]
            enhanced.append(a)

            sep_a = a["SemiMajorAxis"] * (1.0 + a["Eccentricity"])
            sep_b = b["SemiMajorAxis"] * (1.0 + b["Eccentricity"])

            if (
                detect_large_jump(a["Radius(1)"], b["Radius(1)"])
                or detect_large_jump(a["Radius(2)"], b["Radius(2)"])
                or detect_large_jump(sep_a, sep_b)
            ):
                for j in range(1, INTERPOLATED_FRAMES_FOR_JUMP + 1):
                    alpha = j / (INTERPOLATED_FRAMES_FOR_JUMP + 1)
                    inter = {}

                    for k in [
                        "Time",
                        "SemiMajorAxis",
                        "Eccentricity",
                        "Radius(1)",
                        "Radius(2)",
                        "Mass(1)",
                        "Mass(2)",
                    ]:
                        inter[k] = (1.0 - alpha) * a[k] + alpha * b[k]

                    inter["stypeName1"] = a["stypeName1"]
                    inter["stypeName2"] = a["stypeName2"]
                    inter["eventString"] = a["eventString"]
                    enhanced.append(inter)

        enhanced.append(sampled[-1])

        mt_frames = []
        last_mt_frame = -999

        for i, frame in enumerate(enhanced):
            mt_frames.append(frame)

            data_idx = int(round(start + (end - start) * (i / max(1, len(enhanced) - 1))))

            if data_idx in mt_starts:
                if i - last_mt_frame < 25:
                    continue

                for _ in range(MT_PADDING_FRAMES):
                    g = frame.copy()
                    g["eventString"] = "Mass Transfer"
                    mt_frames.append(g)

                last_mt_frame = i

        if frames and mt_frames:
            prev = frames[-1]
            nxt = mt_frames[0]

            sep_prev = prev["SemiMajorAxis"] * (1.0 + prev["Eccentricity"])
            sep_next = nxt["SemiMajorAxis"] * (1.0 + nxt["Eccentricity"])

            if (
                detect_large_jump(prev["Radius(1)"], nxt["Radius(1)"])
                or detect_large_jump(prev["Radius(2)"], nxt["Radius(2)"])
                or detect_large_jump(sep_prev, sep_next)
            ):
                for j in range(1, INTERPOLATED_FRAMES_FOR_JUMP + 1):
                    alpha = j / (INTERPOLATED_FRAMES_FOR_JUMP + 1)
                    inter = {}

                    for k in [
                        "Time",
                        "SemiMajorAxis",
                        "Eccentricity",
                        "Radius(1)",
                        "Radius(2)",
                        "Mass(1)",
                        "Mass(2)",
                    ]:
                        inter[k] = (1.0 - alpha) * prev[k] + alpha * nxt[k]

                    inter["stypeName1"] = prev["stypeName1"]
                    inter["stypeName2"] = prev["stypeName2"]
                    inter["eventString"] = prev["eventString"]
                    frames.append(inter)

        frames.extend(mt_frames)
        print(f"Phase {pidx + 1}/{len(phases)} -> {len(mt_frames)} frames")

    if frames and last_real_time is not None:
        final_frame = frames[-1]
        final_frame["Time"] = float(last_real_time)

    np.savez_compressed(
        out_path,
        frames=np.array(frames, dtype=object),
        binary_id=binary_id,
        metallicity=meta.get("metallicity", np.nan),
        mt_history_HMS_HMS=meta.get("mt_history_HMS_HMS", np.nan),
        mt_history_CO_HMS_RLO=meta.get("mt_history_CO_HMS_RLO", np.nan),
        mt_history_CO_HeMS=meta.get("mt_history_CO_HeMS", np.nan),
        mt_history_CO_HeMS_RLO=meta.get("mt_history_CO_HeMS_RLO", np.nan),
    )

    print(f"Saved {len(frames)} frames -> {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", default=str(HDF5_PATH))
    parser.add_argument("--out", default=str(OUTPUT_FRAMES_FILE))
    parser.add_argument("--binary-id", type=int, default=BINARY_ID)
    args = parser.parse_args()

    preprocess_to_frames(args.hdf5, args.out, args.binary_id)


if __name__ == "__main__":
    main()