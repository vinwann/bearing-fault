"""
Bearing fault detection for the centrifuge Motor DE sensor.

Two outputs per steady-running waveform snapshot:
  * Bearing Health Index (BHI): physics-informed, bearing-specific score
    -> status in {"normal", "warning", "alarm"}
  * General anomaly score (robust Mahalanobis on all selected features)
    -> flags non-bearing changes (operating regime, load, imbalance)

Every scored snapshot also carries its explanation: the contribution of each
bearing feature to the BHI (they sum exactly to the raw BHI) and the top driver.

Snapshots that cannot be assessed get status "stopped" or "not_assessed".

Typical use (see analysis.ipynb):
    snap  = steady_snapshots(dev_rows)            # waveform snapshots + running / run info
    feats = build_features(snap[snap["steady"]])  # 21 features per steady snapshot
    art   = fit_artifacts(F)                      # F = features + "split" + "run_id"
    pred  = predict(heldout_rows, art)            # raw rows in, status + explanation out
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import kurtosis
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler

from utils import (
    bandpass_filter,
    get_fft,
    get_rms,
    highpass_filter,
    preprocess_acceleration,
    preprocess_velocity,
    remove_settling_time,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_COL = "Motor DE_Z Short Wave High Frequency"
LF_COL = "Motor DE_Z Short Wave Low Frequency"
FS_HF = 51200
FS_LF = 5120

RUNNING_THR = 3.0         # waveform RMS (m/s^2) above which the machine is running
RUN_GAP_H = 2.0           # gap between running snapshots that starts a new run
WARMUP_H = 3.0            # start-up transient excluded from assessment
WINDOW = 6                # rolling median window (snapshots, about 6 h)
MIN_PERIODS = 3           # snapshots needed before a smoothed score is issued
F1X_NOMINAL = 24.7        # running speed (Hz), used if the LF waveform is missing

BANDS = [(10, 500), (500, 1500), (1500, 4500), (4500, 9000),
         (9000, 13000), (13000, 17000), (17000, 20000)]
ENV_BAND = (1800, 4500)   # resonance band excited by bearing impacts

# Bearing-specific features and the direction in which they move with a fault
BHI_FEATS = {"band_1500_4500": +1, "env_floor": +1, "spec_centroid": -1, "env_1x_5x": +1}

# Plain-language names and meanings, used in alert explanations
FEATURE_LABELS = {
    "band_1500_4500": "Resonance band 1.5-4.5 kHz",
    "env_floor": "Envelope floor",
    "spec_centroid": "Spectral shift (centroid)",
    "env_1x_5x": "Shaft-rate modulation 1X-5X",
}
FEATURE_MEANING = {
    "band_1500_4500": "impacts are exciting the 2-4 kHz structural resonances (typical of bearing defects)",
    "env_floor": "vibration in the resonance band has become more impulsive (repeated impacts)",
    "spec_centroid": "energy has shifted towards lower frequencies (damage spreading)",
    "env_1x_5x": "impacts are modulated at shaft speed (consistent with an inner-race defect or looseness)",
}

# All selected features, used for the general anomaly score
ANOMALY_FEATS = ["band_1500_4500", "env_floor", "spec_centroid", "env_1x_5x",
                 "acc_rms", "vel_1x", "vel_305", "vel_rms"]


# ---------------------------------------------------------------------------
# Snapshot preparation (shared by training and inference)
# ---------------------------------------------------------------------------
def _is_waveform(v) -> bool:
    return isinstance(v, np.ndarray) and v.size > 0


def wf_rms(a: np.ndarray, fs: int = FS_HF) -> float:
    """RMS of the 10 Hz high-passed waveform (same units as the input, m/s^2)."""
    a = np.asarray(a, dtype=float)
    x = highpass_filter(a - a.mean(), 10, fs)
    x = remove_settling_time(x, 10, len(a), fs)
    return get_rms(x - x.mean())


def steady_snapshots(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Waveform snapshots with running state and run structure.

    Returns one row per snapshot that has a Z high-frequency waveform, with the
    original columns plus: wf_rms, running, run_id (running snapshots only),
    t_in_run and steady (running and past the start-up transient).
    """
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    snap = df[df[HF_COL].apply(_is_waveform)].sort_values("date").reset_index(drop=True)

    snap["wf_rms"] = snap[HF_COL].apply(wf_rms)
    snap["running"] = snap["wf_rms"] > RUNNING_THR
    # Run id: consecutive running snapshots, a new run after a gap of more than RUN_GAP_H
    run = snap[snap["running"]]
    run_id = (run["date"].diff() > pd.Timedelta(hours=RUN_GAP_H)).cumsum()
    snap["run_id"] = run_id.astype(float).reindex(snap.index)        # NaN when not running

    # Time since the start of the run, built as a whole column (no partial assignment,
    # which some pandas versions reject when mixing datetime and timedelta dtypes)
    run_start = snap.groupby("run_id")["date"].transform("min")      # NaT when not running
    snap["t_in_run"] = snap["date"] - run_start
    snap["steady"] = snap["running"] & (snap["t_in_run"] >= pd.Timedelta(hours=WARMUP_H)).fillna(False)
    return snap


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _band_rms(f: np.ndarray, amp: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f < hi)
    return float(np.sqrt(np.sum(amp[m] ** 2)))


def hf_features(a: np.ndarray, f1x: float = F1X_NOMINAL) -> dict:
    """Time, spectral and envelope features from the Z high-frequency waveform."""
    a = np.asarray(a, dtype=float)
    x = preprocess_acceleration(a, FS_HF)          # g
    f, amp = get_fft(x, FS_HF)
    rms = get_rms(x)
    feats = {
        "acc_rms": rms,
        "acc_kurt": float(kurtosis(x)),
        "acc_crest": float(np.max(np.abs(x)) / rms),
        "spec_centroid": float(np.sum(f * amp ** 2) / np.sum(amp ** 2)),
    }
    for lo, hi in BANDS:
        feats[f"band_{lo}_{hi}"] = _band_rms(f, amp, lo, hi)
    feats["ratio_low_high"] = feats["band_10_500"] / (
        feats["band_13000_17000"] + feats["band_17000_20000"])

    # Envelope analysis in the resonance band
    xb = bandpass_filter(a - a.mean(), ENV_BAND[0], ENV_BAND[1], FS_HF)[len(a) // 10:]
    env = np.abs(hilbert(xb))
    e = env - env.mean()
    es = np.abs(np.fft.rfft(e * np.hanning(len(e)))) / len(e)
    ef = np.fft.rfftfreq(len(e), 1 / FS_HF)
    feats["env_kurt"] = float(kurtosis(xb))
    feats["env_floor"] = float(np.median(es[(ef > 5) & (ef < 500)]))
    feats["env_1x_5x"] = float(sum(
        es[(ef > k * f1x - 2) & (ef < k * f1x + 2)].max() for k in range(1, 6)))
    return feats


def lf_features(a: np.ndarray) -> dict:
    """Velocity features from the Z low-frequency waveform (mm/s)."""
    a = np.asarray(a, dtype=float)
    v = preprocess_velocity(a, FS_LF)
    f, amp = get_fft(v, FS_LF)
    m = (f > 20) & (f < 30)
    f1x = float(f[m][np.argmax(amp[m])])

    def peak(c: float) -> float:
        return float(amp[(f > c - 1.5) & (f < c + 1.5)].max())

    return {
        "vel_rms": get_rms(v),
        "f_1x": f1x,
        "vel_1x": peak(f1x),
        "vel_2x": peak(2 * f1x),
        "vel_305": float(amp[(f > 300) & (f < 310)].max()),
        "vel_floor": float(np.median(amp[(f > 30) & (f < 500)])),
    }


def snapshot_features(row: pd.Series) -> dict:
    """All features for one snapshot; LF features are skipped if that waveform is missing."""
    feats = {}
    lf = row.get(LF_COL)
    f1x = F1X_NOMINAL
    if _is_waveform(lf):
        feats.update(lf_features(lf))
        f1x = feats["f_1x"]
    feats.update(hf_features(row[HF_COL], f1x=f1x))
    return feats


def build_features(snap: pd.DataFrame) -> pd.DataFrame:
    """Feature table (one row per snapshot, same index as `snap`)."""
    return pd.DataFrame([snapshot_features(r) for _, r in snap.iterrows()], index=snap.index)


# ---------------------------------------------------------------------------
# Scoring and explanation helpers
# ---------------------------------------------------------------------------
def _log(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df.abs() + 1e-9)


def bhi_zscores(feat_df: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Signed z-scores of the bearing features against the healthy baseline."""
    z = (_log(feat_df[list(BHI_FEATS)]) - art["bhi_mu"]) / art["bhi_sd"]
    for c, s in BHI_FEATS.items():
        z[c] *= s
    return z


def bhi_contributions(feat_df: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Contribution of each bearing feature to the raw BHI (columns sum exactly to it)."""
    return bhi_zscores(feat_df, art) / len(BHI_FEATS)


def mahalanobis(feat_df: pd.DataFrame, art: dict) -> pd.Series:
    out = pd.Series(np.nan, index=feat_df.index)
    ok = feat_df[ANOMALY_FEATS].notna().all(axis=1)
    if ok.any():
        z = art["scaler"].transform(_log(feat_df.loc[ok, ANOMALY_FEATS]))
        out[ok] = np.sqrt(art["mcd"].mahalanobis(z))
    return out


def mahalanobis_contributions(feat_df: pd.DataFrame, art: dict) -> pd.DataFrame:
    """
    Per-feature terms of the squared Mahalanobis distance (they sum exactly to it).
    A negative term means the feature moved consistently with the normal correlations.
    """
    ok = feat_df[ANOMALY_FEATS].notna().all(axis=1)
    x = art["scaler"].transform(_log(feat_df.loc[ok, ANOMALY_FEATS]))
    d = x - art["mcd"].location_
    return pd.DataFrame(d * (d @ art["mcd"].precision_), columns=ANOMALY_FEATS,
                        index=feat_df.index[ok])


def _smooth(s: pd.Series, run_id: pd.Series) -> pd.Series:
    return s.groupby(run_id).transform(
        lambda x: x.rolling(WINDOW, min_periods=MIN_PERIODS).median())


def explain_alert(row: pd.Series, min_contribution: float = 0.25) -> str:
    """Plain-language explanation for one row of predict() output."""
    contrib = {f: row.get(f"c_{f}") for f in BHI_FEATS}
    contrib = {f: v for f, v in contrib.items() if pd.notna(v)}
    lines = [f"{row['date']:%d %b %Y %H:%M}  status={row['status']}  BHI={row['bhi']:.2f}"]
    for f, v in sorted(contrib.items(), key=lambda kv: -kv[1]):
        if v >= min_contribution:
            lines.append(f"  - {FEATURE_LABELS[f]}: +{v:.2f} "
                         f"({v * len(BHI_FEATS):.1f} sd above baseline): {FEATURE_MEANING[f]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def fit_artifacts(F: pd.DataFrame, split_col: str = "split", train: str = "train",
                  healthy_val: str = "val_healthy", fault: str = "labelled_fault") -> dict:
    """
    Fit the BHI baseline, the anomaly model and both thresholds.

    F must contain the feature columns, a run identifier `run_id` and a split column.
    Only three splits are used:
      train       : healthy data used to fit the baselines
      healthy_val : healthy data used, with train, to set the Warning threshold
      fault       : labelled fault data used to set the Alarm threshold
    Any other split (including held-out data) is ignored.
    """
    s = F[split_col]
    tr, healthy, flt = s == train, s.isin([train, healthy_val]), s == fault
    for name, m in [(train, tr), (healthy_val, s == healthy_val), (fault, flt)]:
        if not m.any():
            raise ValueError(f"No rows in split '{name}'")

    art = {}

    # Bearing Health Index baseline (median: robust to outliers)
    xb = _log(F.loc[tr, list(BHI_FEATS)])
    art["bhi_mu"], art["bhi_sd"] = xb.median(), xb.std()

    # General anomaly model
    xa = _log(F.loc[tr, ANOMALY_FEATS].dropna())
    art["scaler"] = StandardScaler().fit(xa)
    art["mcd"] = MinCovDet(random_state=0).fit(art["scaler"].transform(xa))

    # Smoothed scores on the fitting data, to set thresholds
    bhi = _smooth(bhi_zscores(F, art).mean(axis=1), F["run_id"])
    mah = _smooth(mahalanobis(F, art), F["run_id"])

    art["bhi_warn"] = float(np.nanquantile(bhi[healthy], 0.995))   # above the healthy range
    art["bhi_alarm"] = float(np.nanquantile(bhi[flt], 0.10))       # as severe as the labelled fault
    art["anomaly_thr"] = float(np.nanquantile(mah[healthy], 0.995))

    if art["bhi_alarm"] <= art["bhi_warn"]:
        raise ValueError(f"Alarm ({art['bhi_alarm']:.2f}) must exceed warning ({art['bhi_warn']:.2f})")

    art["config"] = dict(RUNNING_THR=RUNNING_THR, RUN_GAP_H=RUN_GAP_H, WARMUP_H=WARMUP_H,
                         WINDOW=WINDOW, MIN_PERIODS=MIN_PERIODS, BHI_FEATS=BHI_FEATS,
                         ANOMALY_FEATS=ANOMALY_FEATS, ENV_BAND=ENV_BAND,
                         splits=dict(train=train, healthy_val=healthy_val, fault=fault))
    return art


def save_artifacts(art: dict, path: str) -> None:
    joblib.dump(art, path)


def load_artifacts(path: str) -> dict:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict(df_raw: pd.DataFrame, art: dict) -> pd.DataFrame:
    """
    Raw parquet rows in; one row per waveform snapshot out.

    Columns:
      date, wf_rms, running, run_id, bhi, anomaly_score, status, general_anomaly,
      top_driver, c_<feature> (contribution of each bearing feature to the raw BHI), condition
    status (what the model saw in this snapshot):
      "stopped"       machine not running
      "not_assessed"  start-up transient, or not enough snapshots yet for a smoothed score
      "normal" / "warning" / "alarm"   from the Bearing Health Index
    condition (asset health for dashboards):
      last assessed status carried forward through stopped / not-assessed periods;
      "unknown" before the first assessment
    """
    snap = steady_snapshots(df_raw)
    out = snap[["date", "wf_rms", "running", "run_id"]].copy()
    out["bhi"] = np.nan
    out["anomaly_score"] = np.nan
    out["status"] = np.where(out["running"], "not_assessed", "stopped")
    out["general_anomaly"] = False
    out["top_driver"] = None
    for f in BHI_FEATS:
        out[f"c_{f}"] = np.nan

    idx = snap.index[snap["steady"]]
    if len(idx) == 0:
        return _add_condition(out)

    feats = build_features(snap.loc[idx])
    run_id = snap.loc[idx, "run_id"]

    contrib = bhi_contributions(feats, art)
    bhi = _smooth(contrib.sum(axis=1), run_id)
    mah = _smooth(mahalanobis(feats, art), run_id)

    out.loc[idx, "bhi"] = bhi
    out.loc[idx, "anomaly_score"] = mah
    out.loc[idx, "general_anomaly"] = (mah > art["anomaly_thr"]).values
    out.loc[idx, "top_driver"] = contrib.idxmax(axis=1).map(FEATURE_LABELS)
    for f in BHI_FEATS:
        out.loc[idx, f"c_{f}"] = contrib[f]

    scored = out.index.isin(idx) & out["bhi"].notna()
    out.loc[scored, "status"] = np.select(
        [out.loc[scored, "bhi"] >= art["bhi_alarm"], out.loc[scored, "bhi"] >= art["bhi_warn"]],
        ["alarm", "warning"], default="normal")
    return _add_condition(out)


def _add_condition(out: pd.DataFrame) -> pd.DataFrame:
    """
    `condition` = last assessed bearing status, carried forward through
    "stopped" / "not_assessed" snapshots (a damaged bearing does not recover
    while the machine is off). "unknown" before the first assessment.
    """
    assessed = out["status"].isin(["normal", "warning", "alarm"])
    out["condition"] = out["status"].where(assessed).ffill().fillna("unknown")
    return out


def status_at(pred: pd.DataFrame, times, tolerance: str = "2h") -> pd.DataFrame:
    """
    Map snapshot predictions onto arbitrary timestamps (e.g. evaluation rows).

    status    : from the most recent snapshot within `tolerance`, otherwise "no_data"
    condition : most recent known condition with no time limit (health persists
                while no snapshots arrive), otherwise "unknown"
    """
    t = pd.DataFrame({"date": pd.to_datetime(pd.Series(times), utc=True)}).sort_values("date")
    p = pred.sort_values("date")
    m = pd.merge_asof(t, p.drop(columns="condition"), on="date",
                      direction="backward", tolerance=pd.Timedelta(tolerance))
    m["status"] = m["status"].fillna("no_data")
    c = pd.merge_asof(t, p[["date", "condition"]], on="date", direction="backward")
    m["condition"] = c["condition"].fillna("unknown").values
    return m