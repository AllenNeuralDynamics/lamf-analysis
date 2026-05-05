from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Any, Iterable



def resolve_result_bundle_pair(
    result_path: Path | str,
    *,
    h5_path: Path | str | None = None,
    allow_dir_pair: bool = False,
    allow_generic_suffix_swap: bool = False,
) -> tuple[Path, Path]:
    """Resolve metadata/H5 pair from a bundle path.

    Parameters are configurable so callers can preserve existing behaviors.
    """
    p = Path(result_path)
    if h5_path is not None:
        return p, Path(h5_path)

    if allow_dir_pair and p.is_dir():
        meta = next(p.glob("*metadata.json"), None)
        h5 = next(p.glob("*results.h5"), None)
        if meta.exists() and h5.exists():
            return meta, h5

    name = p.name
    suffix = p.suffix.lower()

    if name == "metadata.json":
        return p, p.with_name("results.h5")
    if name == "result.h5":
        return p.with_name("metadata.json"), p

    if name.endswith("_metadata.json"):
        stem = name[: -len("_metadata.json")]
        return p, p.with_name(f"{stem}_results.h5")
    if name.endswith("_results.h5"):
        stem = name[: -len("_results.h5")]
        return p.with_name(f"{stem}_metadata.json"), p

    if allow_generic_suffix_swap:
        if suffix == ".json":
            return p, p.with_suffix(".h5")
        if suffix == ".h5":
            return p.with_suffix(".json"), p

    raise ValueError(f"Cannot resolve result bundle paths from: {p}")


def to_portable_relpath(
    path_value: Path | str,
    *,
    base_candidates: Iterable[Path | str] | None = None,
    add_leading_parent: bool = True,
) -> str:
    """Return a normalized portable relative path string.

    Intended for persisted output artifact references.
    """
    p = Path(path_value)
    if not p.is_absolute():
        rel = p.as_posix()
    else:
        if base_candidates is None:
            base_candidates = (Path("/root/capsule"), Path.cwd())

        rel = ""
        for base in base_candidates:
            try:
                rel = p.relative_to(Path(base)).as_posix()
                break
            except ValueError:
                continue
        if not rel:
            rel = p.as_posix().lstrip("/")

    rel = rel.replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    while rel.startswith("../"):
        rel = rel[3:]
    rel = rel or p.name
    if add_leading_parent:
        return f"../{rel}"
    return rel


def load_result_from_bundle(
    result_path: Path | str,
    h5_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load a full result dict from metadata + h5 result files."""
    import h5py

    if h5_path is None:
        metadata_path, h5_file_path = resolve_result_bundle_pair(
            result_path,
            allow_dir_pair=True,
            allow_generic_suffix_swap=True,
        )
    else:
        metadata_path = Path(result_path)
        h5_file_path = Path(h5_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not h5_file_path.exists():
        raise FileNotFoundError(h5_file_path)
    
    with open(metadata_path, "r") as f:
        manifest = json.load(f)

    result = {}
    with h5py.File(h5_file_path, "r") as f:
        if "registered_zstack" in f:
            result["registered_zstack"] = np.array(f["registered_zstack"])
        if "matched_plane_indices" in f:
            result["matched_plane_indices"] = np.array(f["matched_plane_indices"])
        if "padded_plane_indices" in f:
            result["padded_plane_indices"] = np.array(f["padded_plane_indices"])
        if "crop_y_inds" in f:
            result["crop_y_inds"] = np.array(f["crop_y_inds"])
        if "crop_x_inds" in f:
            result["crop_x_inds"] = np.array(f["crop_x_inds"])
        if "fov_mean" in f:
            result["fov_mean"] = np.array(f["fov_mean"])
        if "fov_mean_cropped" in f:
            result["fov_mean_cropped"] = np.array(f["fov_mean_cropped"])
        if "roi_time_profile" in f:
            g = f["roi_time_profile"]
            columns = json.loads(g.attrs.get("columns_json", "[]"))
            if not columns:
                columns = list(g.keys())
            roi_time_data: dict[str, Any] = {}
            for col in columns:
                if col not in g:
                    continue
                arr = np.array(g[col])
                if arr.dtype.kind in {"S", "O"}:
                    arr = np.array([
                        x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
                        for x in arr.tolist()
                    ])
                roi_time_data[str(col)] = arr
            result["roi_time_df"] = pd.DataFrame(roi_time_data)
            result["roi_time_profile_params"] = json.loads(g.attrs.get("params_json", "{}"))

        padded_z_start = int(f.attrs.get("padded_z_start", -1))
        padded_z_end = int(f.attrs.get("padded_z_end", -1))
        desired_z_start = int(f.attrs.get("desired_z_start_with_padding", padded_z_start))
        desired_z_end = int(f.attrs.get("desired_z_end_with_padding", padded_z_end))

        h5_pad = f.attrs.get("pad", None)
        if h5_pad is None and ("matched_plane_indices" in result) and desired_z_start >= 0 and desired_z_end >= 0:
            matched = np.asarray(result["matched_plane_indices"]).astype(int)
            if matched.size > 0:
                z_min = int(np.min(matched))
                z_max = int(np.max(matched))
                lower = max(0, z_min - desired_z_start)
                upper = max(0, desired_z_end - (z_max + 1))
                h5_pad = int(min(lower, upper))

        result["z_stack_padding"] = {
            "z_drift_min": int(f.attrs.get("z_drift_min", -1)),
            "z_drift_max": int(f.attrs.get("z_drift_max", -1)),
            "desired_z_start_with_padding": int(desired_z_start),
            "desired_z_end_with_padding": int(desired_z_end),
            "pad": int(h5_pad) if h5_pad is not None else -1,
        }

        if result["z_stack_padding"]["z_drift_min"] < 0 and ("matched_plane_indices" in result):
            matched = np.asarray(result["matched_plane_indices"]).astype(int)
            if matched.size > 0:
                result["z_stack_padding"]["z_drift_min"] = int(np.min(matched))
                result["z_stack_padding"]["z_drift_max"] = int(np.max(matched))

    if "z_stack_padding_info" in manifest:
        padding_info = manifest["z_stack_padding_info"]
        result["z_stack_padding"] = {
            "z_drift_min": padding_info.get("z_drift_range", [-1, -1])[0],
            "z_drift_max": padding_info.get("z_drift_range", [-1, -1])[1],
            "desired_z_start_with_padding": padding_info.get("desired_padded_range", [-1, -1])[0],
            "desired_z_end_with_padding": padding_info.get("desired_padded_range", [-1, -1])[1],
            "pad": padding_info.get("pad", -1),
        }
    elif ("padded_z_start" in manifest) or ("padded_z_end" in manifest):
        current = result.get("z_stack_padding", {})
        result["z_stack_padding"] = {
            "z_drift_min": int(current.get("z_drift_min", -1)),
            "z_drift_max": int(current.get("z_drift_max", -1)),
            "desired_z_start_with_padding": int(manifest.get("padded_z_start", current.get("desired_z_start_with_padding", -1))),
            "desired_z_end_with_padding": int(manifest.get("padded_z_end", current.get("desired_z_end_with_padding", -1))),
            "pad": int(current.get("pad", -1)),
        }

    result["summary"] = {
        "selected_method": manifest.get("selected_method"),
        "metrics_by_method": manifest.get("metrics_by_method", {}),
        "transforms_by_method": manifest.get("transforms_by_method", {}),
        "shared_eval_valid_frac": manifest.get("shared_eval_valid_frac", np.nan),
        "gate": manifest.get("gate", {}),
    }
    result["all_methods"] = {}
    result["affine_candidates"] = []
    result["nonrigid_candidates"] = []
    
    # Add metadata from manifest
    result["row_i"] = int(manifest.get("row_i", -1))
    result["winner"] = manifest.get("winner", None)
    result["session_key"] = manifest.get("session_key", None)
    result["plane_path"] = manifest.get("plane_path", None)
    result["plane_id"] = manifest.get("plane_id", None)
    result["processed_name"] = manifest.get("processed_name", None)
    result["source_result_file"] = to_portable_relpath(metadata_path)
    if "roi_time_profile" in manifest and "roi_time_profile_params" not in result:
        result["roi_time_profile_params"] = dict(manifest.get("roi_time_profile", {}))
    
    return result