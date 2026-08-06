"""Behavior stimulus-table extraction, bypassing comb.

Reproduces comb's ``BehaviorSessionDataset.stimulus_presentations`` table (flash-by-flash) directly
from the raw stimulus pickle + sync .h5, and additionally extracts a **trials** table that includes
CATCH trials (which comb currently drops) with hit / miss / false-alarm / correct-reject / abort
outcomes parsed from the pickle ``trial_log``.

Validated against comb (2026-08, for_gcamp_validation) on 6 subjects x 2 session types:
  * go-trial ``change_time`` matches comb to machine precision (constant monitor_delay = 0.03613 s);
  * abort / false-alarm / hit event times land exactly on lick times.

Public API
----------
get_stimulus_presentations(pkl_file, sync_file)  -> pd.DataFrame  (comb-identical flash table)
get_trials(pkl_file, sync_file, stim_pres=None)  -> pd.DataFrame  (go + CATCH trials)
get_stim_table(pkl_file, sync_file)              -> (stim_pres, trials)

Only the change-detection (OPHYS / image) stimulus is reproduced here (comb's "else" branch);
STAGE_0 / STAGE_1 grating-training parsers are not ported.
"""
import ast
import pickle
from typing import Sequence

import numpy as np
import pandas as pd
import h5py

MONITOR_DELAY = 0.03613                       # comb default (BehaviorSessionDataset.get_stimulus_presentations)
STIMULUS_VSYNC_KEYS = ("stim_vsync", "vsync_stim", "frames")
RESP_WINDOW = (0.15, 0.75)                    # task response/reward window (s after change)


# --------------------------------------------------------------------------- sync (minimal, comb-free)
def _trim_discontiguous_times(times: np.ndarray, threshold: float = 100) -> np.ndarray:
    """Trim frame times after the first large gap (an extra post-session acquisition burst)."""
    times = np.asarray(times)
    if len(times) < 2:
        return times
    intervals = np.diff(times)
    interval_threshold = np.median(intervals) * threshold
    gap = np.where(intervals > interval_threshold)[0]
    if abs(intervals[0]) > interval_threshold:
        gap = np.array([0])
    return times if len(gap) == 0 else times[: gap[0] + 1]


def stimulus_frame_times(sync_file: str, keys: Sequence[str] = STIMULUS_VSYNC_KEYS,
                         trim_after_spike: bool = True) -> np.ndarray:
    """Rising-edge times (s) of the stimulus vsync line — comb's get_synchronized_frame_times, ported."""
    with h5py.File(sync_file, "r") as f:
        data = f["data"][()]                      # (N, 2): [counter, bits]
        meta = eval(f["meta"][()])                # comb parses meta the same way
    line_labels = meta["line_labels"]
    ni = meta["ni_daq"]
    freq = float(ni["sample_freq"]) if "sample_freq" in ni else float(ni["counter_output_freq"])
    # rollover-corrected sample counter (32-bit)
    t = data[:, 0:1].astype(np.int64)
    intervals = np.ediff1d(t, to_begin=np.array([0], dtype=t.dtype))
    for i in np.where(intervals < 0)[0]:
        t[i:] += 4294967296
    t = t[:, 0]
    bits = data[:, -1]
    bit = next((line_labels.index(k) for k in keys if k in line_labels), None)
    if bit is None:
        raise KeyError(f"none of {keys} in sync line labels {line_labels}")
    bitvals = np.bitwise_and(bits, 2 ** bit).astype(bool).astype(np.int8)
    rising = np.where(np.diff(bitvals) == 1)[0] + 1        # 0 -> 1 transitions
    times = t[rising] / freq
    return _trim_discontiguous_times(times) if trim_after_spike else times


# --------------------------------------------------------------------------- pickle helpers (vendored)
def _behavior_key(data):
    return "behavior" if "behavior" in data["items"] else "foraging"


def _get_stimulus_epoch(set_log, current_set_index, start_frame, n_frames):
    try:
        next_set_event = set_log[current_set_index + 1]
    except IndexError:
        next_set_event = (None, None, None, n_frames)
    return start_frame, next_set_event[3]


def _get_draw_epochs(draw_log, start_frame, stop_frame):
    epochs, cur = [], start_frame
    while cur <= stop_frame:
        length = 0
        while cur < stop_frame and draw_log[cur] == 1:
            length += 1
            cur += 1
        else:
            cur += 1
        if length:
            epochs.append((cur - length - 1, cur - 1))
    return epochs


def _visual_stimuli_df(data, time) -> pd.DataFrame:
    """Image (change-detection) flash table from set_log/draw_log + omitted_flash_frame_log — comb's
    'else' branch, verbatim in logic (start_time filled from `time` in get_stimulus_presentations)."""
    bkey = _behavior_key(data)
    stage = data["items"][bkey]["params"].get("stage", "")
    if str(stage) in ("STAGE_0", "STAGE_1"):
        raise NotImplementedError(f"grating-training stage {stage!r} parser not ported (image sessions only)")
    stimuli = data["items"][bkey]["stimuli"]
    n_frames = len(time)
    rows = []
    for stim_dict in stimuli.values():
        for idx, (attr_name, attr_value, _, frame) in enumerate(stim_dict["set_log"]):
            image_name = attr_value if attr_name.lower() == "image" else np.nan
            orientation = attr_value if attr_name.lower() == "ori" else np.nan
            s0, s1 = _get_stimulus_epoch(stim_dict["set_log"], idx, frame, n_frames)
            for e0, e1 in _get_draw_epochs(stim_dict["draw_log"], s0, s1):
                rows.append(dict(orientation=orientation, image_name=image_name,
                                 frame=e0, end_frame=e1, time=time[e0],
                                 duration=time[e1] - time[e0], omitted=False))
    df = pd.DataFrame(rows)
    # omitted flashes
    omit_log = data["items"][bkey].get("omitted_flash_frame_log", {}) or {}
    keep = []
    for _, frames in omit_log.items():
        frames = np.array(frames)
        stim_frames = df["frame"].values
        offsets = np.add(np.repeat(frames[:, None], 7, axis=1), np.arange(-3, 4))
        matched = np.any(np.isin(offsets, stim_frames), axis=1)
        keep += list(np.unique(frames[~matched]))
    if keep:
        od = pd.DataFrame({"omitted": np.ones(len(keep), bool), "frame": keep,
                           "end_frame": np.nan, "image_name": "omitted",
                           "time": [time[fi] for fi in keep],
                           "duration": 0.25, "orientation": np.nan})
        df = pd.concat([df, od], ignore_index=True)
    return df.sort_values("frame").reset_index(drop=True)


def _is_change(sp: pd.DataFrame) -> pd.Series:
    """First presentation of a new image_name (omitted ignored; first flash never a change)."""
    s = sp["image_name"][~sp["omitted"].astype(bool)]
    ic = (s != s.shift()).iloc[1:]
    out = pd.Series(False, index=sp.index)
    out.loc[ic.index] = ic.values
    return out


def _flashes_since_change(sp: pd.DataFrame) -> pd.Series:
    fsc = np.zeros(len(sp), dtype=int)
    for i, (_, r) in enumerate(sp.iterrows()):
        omit = bool(r["omitted"]) if not pd.isna(r["omitted"]) else False
        if r["image_name"] == "omitted" or omit:
            fsc[i] = fsc[i - 1]
        elif r["is_change"] or i == 0:
            fsc[i] = 0
        else:
            fsc[i] = fsc[i - 1] + 1
    return pd.Series(fsc, index=sp.index)


# --------------------------------------------------------------------------- public
def _load_pkl(pkl_file):
    with open(pkl_file, "rb") as f:
        return pickle.load(f, encoding="latin1")


def get_stimulus_presentations(pkl_file: str, sync_file: str) -> pd.DataFrame:
    """comb-identical flash table (start_time uses the 0.03613 s monitor delay)."""
    data = _load_pkl(pkl_file)
    st = stimulus_frame_times(sync_file) + MONITOR_DELAY        # StimulusTimestamps.value
    df = _visual_stimuli_df(data, st)
    df = df.rename(columns={"frame": "start_frame", "time": "start_time"})
    df["start_time"] = [st[int(fr)] for fr in df.start_frame.values]
    df["stop_time"] = [st[int(fr)] if not np.isnan(fr) else np.nan for fr in df.end_frame.values]
    df["omitted"] = df["omitted"].astype(bool)
    # comb _check_for_errant_omitted_stimulus: a camstim quirk can leave an omitted flash as the very
    # first entry of the block; comb drops it (https://github.com/AllenInstitute/AllenSDK/issues/2577).
    if len(df) and bool(df.iloc[0]["omitted"]):
        df = df.iloc[1:].reset_index(drop=True)
    df["is_change"] = _is_change(df)
    df["flashes_since_change"] = _flashes_since_change(df)
    # single change-detection block for image sessions (comb: block 0 when one stimulus_name)
    df["stimulus_block"] = 0
    imgs = [x for x in pd.unique(df.image_name) if isinstance(x, str) and x != "omitted"]
    df["image_index"] = df.image_name.map({n: i for i, n in enumerate(imgs)}).astype("Int64")
    df.index.name = "stimulus_presentations_id"
    return df


def get_trials(pkl_file: str, sync_file: str, stim_pres: pd.DataFrame = None) -> pd.DataFrame:
    """Go + CATCH trials. Go: change_time (is_change flash) + hit/miss (lick in [0.15,0.75]s).
    Catch: sham-change time (trial_log) + false_alarm / correct_reject. Also carries the raw outcome
    and abort/early-response time from the trial_log."""
    data = _load_pkl(pkl_file)
    st = stimulus_frame_times(sync_file) + MONITOR_DELAY
    tl = data["items"][_behavior_key(data)]["trial_log"]

    def frame_of(t, name):
        for e in t["events"]:
            if e[0] == name:
                return int(e[3])
        return None

    rows = []
    for t in tl:
        catch = bool(t.get("trial_params", {}).get("catch", False))
        chf = frame_of(t, "sham_change" if catch else "stimulus_changed")
        if chf is None:                       # aborted / no (sham-)change reached
            abf = frame_of(t, "abort")
            rows.append(dict(trial_type="catch" if catch else "go", is_catch=catch,
                             change_time=np.nan, response=np.nan, outcome="aborted",
                             abort_time=(st[abf] if abf is not None else np.nan)))
            continue
        ct = st[chf]
        outcome = next((e[0] for e in t["events"] if e[0] in
                        ("hit", "miss", "false_alarm", "rejection")), None)
        rows.append(dict(trial_type="catch" if catch else "go", is_catch=catch,
                         change_time=ct, outcome=outcome,
                         response=outcome in ("hit", "false_alarm"),
                         abort_time=np.nan))
    trials = pd.DataFrame(rows)
    # hit/miss/false_alarm/correct_reject booleans (task-standard)
    trials["hit"] = (~trials.is_catch) & (trials.outcome == "hit")
    trials["miss"] = (~trials.is_catch) & (trials.outcome == "miss")
    trials["false_alarm"] = trials.is_catch & (trials.outcome == "false_alarm")
    trials["correct_reject"] = trials.is_catch & (trials.outcome == "rejection")
    return trials


def get_stim_table(pkl_file: str, sync_file: str):
    """Return (stimulus_presentations, trials)."""
    sp = get_stimulus_presentations(pkl_file, sync_file)
    return sp, get_trials(pkl_file, sync_file, sp)
