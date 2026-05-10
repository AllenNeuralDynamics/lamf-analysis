from collections import defaultdict
from datetime import datetime

import pandas as pd
from aind_data_access_api.document_db import MetadataDbClient

API_GATEWAY_HOST = "api.allenneuraldynamics.org"
DATABASE = "metadata_index"
COLLECTION = "data_assets"


def get_imaging_plane_metadata(subject_id: str, raw_only: bool = True) -> pd.DataFrame:
    """Get imaging plane metadata for all multiplane-ophys sessions for a given mouse.

    Queries the AIND DocDB for session.data_streams[].ophys_fovs[] fields.

    Parameters
    ----------
    subject_id : str
        Mouse subject ID (e.g. "782149").
    raw_only : bool
        If True (default), only return raw acquisition sessions
        (excludes derived assets like registration/segmentation).

    Returns
    -------
    pd.DataFrame
        One row per imaging plane (FOV) per session, with columns:
        mouse_id, session_name, session_date, fov_index, coupled_fov_index,
        imaging_depth_um, targeted_structure, plane_name, fov_width,
        fov_height, fov_scale_factor_um_per_px, frame_rate_hz, power_pct,
        power_ratio_pct, scanfield_z_um, scanimage_roi_index
    """
    client = MetadataDbClient(
        host=API_GATEWAY_HOST,
        database=DATABASE,
        collection=COLLECTION,
    )

    # Raw sessions match "multiplane-ophys_<id>_<date>_<time>" with no further suffix
    if raw_only:
        name_pattern = (
            f"^multiplane-ophys_{subject_id}_"
            r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
        )
    else:
        name_pattern = f"^multiplane-ophys_{subject_id}"

    records = client.retrieve_docdb_records(
        filter_query={"name": {"$regex": name_pattern}},
        projection={
            "name": 1,
            "session": 1,
            "_id": 0,
        },
        limit=100,
    )

    rows = []
    for record in records:
        session_name = record.get("name")
        session = record.get("session")
        if not session:
            continue

        session_start = session.get("session_start_time", "")
        # Extract YYYY-MM-DD from the ISO timestamp
        session_date = session_start[:10] if session_start else ""

        for stream in session.get("data_streams", []):
            for fov in stream.get("ophys_fovs", []):
                targeted = fov.get("targeted_structure", "")
                fov_index = fov.get("index")
                plane_name = f"{targeted}_{fov_index}" if targeted and fov_index is not None else None
                rows.append({
                    "subject_id": subject_id,
                    "session_name": session_name,
                    "session_key": f"{subject_id}_{session_date}",
                    "session_date": session_date,
                    "fov_index": fov_index,
                    "coupled_fov_index": fov.get("coupled_fov_index"),
                    "imaging_depth_um": fov.get("imaging_depth"),
                    "targeted_structure": targeted,
                    "plane_id": plane_name,
                    "fov_width": fov.get("fov_width"),
                    "fov_height": fov.get("fov_height"),
                    "fov_scale_factor_um_per_px": fov.get("fov_scale_factor"),
                    "frame_rate_hz": fov.get("frame_rate"),
                    "power_pct": fov.get("power"),
                    "power_ratio_pct": fov.get("power_ratio"),
                    "scanfield_z_um": fov.get("scanfield_z"),
                    "scanimage_roi_index": fov.get("scanimage_roi_index"),
                    "mouse_id": subject_id,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["session_name", "fov_index"]).reset_index(drop=True)
    return df


def get_session_metadata(subject_id: str, raw_only: bool = True) -> pd.DataFrame:
    """Get session-level metadata for all multiplane-ophys sessions for a given mouse.

    Parameters
    ----------
    subject_id : str
        Mouse subject ID (e.g. "782149").
    raw_only : bool
        If True (default), only return raw acquisition sessions
        (excludes derived assets like registration/segmentation).

    Returns
    -------
    pd.DataFrame
        One row per session with columns including:
        mouse_id, session_name, session_date, session_key, session_type,
        stimulus, session_type_exposures, stimulus_exposures,
        project_name, rig_id, num_planes, num_coupled_groups,
        session_start_time, session_end_time, session_duration_min,
        stimulus_names, mouse_platform_name, platform, modalities,
        data_level, schema_version, s3_location, raw_asset_id,
        derived_asset_name, dff_long_window, dlc_eye_asset_name
    """
    client = MetadataDbClient(
        host=API_GATEWAY_HOST,
        database=DATABASE,
        collection=COLLECTION,
    )

    if raw_only:
        name_pattern = (
            f"^multiplane-ophys_{subject_id}_"
            r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
        )
    else:
        name_pattern = f"^multiplane-ophys_{subject_id}"

    records = client.retrieve_docdb_records(
        filter_query={"name": {"$regex": name_pattern}},
        projection={
            "name": 1,
            "session": 1,
            "data_description": 1,
            "external_links": 1,
            "location": 1,
            "schema_version": 1,
            "_id": 0,
        },
        limit=100,
    )

    rows = []
    for record in records:
        session_name = record.get("name")
        session = record.get("session")
        if not session:
            continue

        dd = record.get("data_description", {}) or {}
        session_start = session.get("session_start_time", "")
        session_end = session.get("session_end_time", "")
        session_date = session_start[:10] if session_start else ""

        # Compute duration
        duration_min = None
        if session_start and session_end:
            try:
                start = datetime.fromisoformat(session_start)
                end = datetime.fromisoformat(session_end)
                duration_min = round((end - start).total_seconds() / 60, 1)
            except (ValueError, TypeError):
                pass

        # Count planes and coupled groups
        num_planes = 0
        coupled_indices = set()
        for stream in session.get("data_streams", []):
            fovs = stream.get("ophys_fovs", [])
            num_planes += len(fovs)
            for fov in fovs:
                ci = fov.get("coupled_fov_index")
                if ci is not None:
                    coupled_indices.add(ci)

        # Collect stimulus names
        stim_names = []
        for epoch in session.get("stimulus_epochs", []):
            name = epoch.get("stimulus_name")
            if name and name not in stim_names:
                stim_names.append(name)

        # External links / Code Ocean asset IDs
        ext_links = record.get("external_links", {}) or {}
        co_ids = ext_links.get("Code Ocean", [])
        raw_asset_id = co_ids[0] if co_ids else None

        # Modality list
        modalities = dd.get("modality", [])
        modality_str = ", ".join(m.get("abbreviation", "") for m in modalities) if modalities else None

        # Platform
        platform = dd.get("platform", {}) or {}

        rows.append({
            "subject_id": subject_id,
            "session_name": session_name,
            "session_key": f"{subject_id}_{session_date}",
            "session_date": session_date,
            "session_type": session.get("session_type"),
            "project_name": dd.get("project_name"),
            "rig_id": session.get("rig_id"),
            "num_planes": num_planes,
            "num_coupled_groups": len(coupled_indices),
            "session_start_time": session_start,
            "session_end_time": session_end,
            "session_duration_min": duration_min,
            "stimulus_names": ", ".join(stim_names) if stim_names else None,
            "mouse_platform_name": session.get("mouse_platform_name"),
            "platform": platform.get("abbreviation"),
            "modalities": modality_str,
            "data_level": dd.get("data_level"),
            "schema_version": record.get("schema_version"),
            "s3_location": record.get("location"),
            "raw_asset_id": raw_asset_id,
            "mouse_id": subject_id,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("session_date").reset_index(drop=True)

        # --- Computed exposure counters ---
        df["session_type_exposures"] = (
            df.groupby("session_type").cumcount() + 1
        )
        df["stimulus"] = df["session_type"].apply(_map_session_type_to_stimulus)
        df["stimulus_exposures"] = (
            df.groupby("stimulus").cumcount() + 1
        )

        # --- Look up the latest "processed" derived asset for each raw session ---
        processed_records = client.retrieve_docdb_records(
            filter_query={"name": {"$regex": f"^multiplane-ophys_{subject_id}_.*_processed_"}},
            projection={"name": 1, "external_links": 1, "_id": 0},
            limit=0,
            paginate=True,
        )
        processed_map = defaultdict(list)
        for rec in processed_records:
            pname = rec.get("name", "")
            co = (rec.get("external_links") or {}).get("Code Ocean", [])
            asset_id = co[0] if co else None
            for raw_name in df["session_name"]:
                if pname.startswith(raw_name + "_processed_"):
                    processed_map[raw_name].append((pname, asset_id))
        # Pick the latest (alphabetically last, since timestamps sort correctly)
        latest_processed = {
            k: sorted(v)[-1] for k, v in processed_map.items()
        }
        df["derived_asset_name"] = df["session_name"].map(
            {k: v[0] for k, v in latest_processed.items()}
        )
        df["derived_asset_id"] = df["session_name"].map(
            {k: v[1] for k, v in latest_processed.items()}
        )

        # --- dF/F long_window from processed assets ---
        processed_asset_ids = df["derived_asset_id"].dropna().tolist()
        if processed_asset_ids:
            dff_results = client.aggregate_docdb_records([
                {"$match": {
                    "external_links.Code Ocean": {"$in": processed_asset_ids},
                }},
                {"$project": {
                    "_id": 0,
                    "code_ocean_id": {
                        "$arrayElemAt": ["$external_links.Code Ocean", 0]
                    },
                    "df_f_params": {
                        "$filter": {
                            "input": "$processing.processing_pipeline.data_processes",
                            "as": "process",
                            "cond": {"$eq": ["$$process.name", "dF/F estimation"]},
                        }
                    },
                }},
                {"$project": {
                    "code_ocean_id": 1,
                    "long_window": {
                        "$arrayElemAt": ["$df_f_params.parameters.long_window", 0]
                    },
                }},
            ])
            dff_map = {
                r["code_ocean_id"]: r.get("long_window")
                for r in dff_results
                if r.get("long_window") is not None
            }
            df["dff_long_window"] = df["derived_asset_id"].map(dff_map)
        else:
            df["dff_long_window"] = None

        # --- DLC eye-tracking derived assets ---
        dlc_records = client.retrieve_docdb_records(
            filter_query={"name": {"$regex": f"^multiplane-ophys_{subject_id}_.*dlc-eye"}},
            projection={"name": 1, "external_links": 1, "_id": 0},
            limit=0,
            paginate=True,
        )
        dlc_map = defaultdict(list)
        for rec in dlc_records:
            dname = rec.get("name", "")
            co = (rec.get("external_links") or {}).get("Code Ocean", [])
            dlc_id = co[0] if co else None
            for raw_name in df["session_name"]:
                if dname.startswith(raw_name + "_dlc-eye"):
                    dlc_map[raw_name].append((dname, dlc_id))
        latest_dlc = {k: sorted(v)[-1] for k, v in dlc_map.items()}
        df["dlc_eye_asset_name"] = df["session_name"].map(
            {k: v[0] for k, v in latest_dlc.items()}
        )
        df["dlc_eye_asset_id"] = df["session_name"].map(
            {k: v[1] for k, v in latest_dlc.items()}
        )

    return df


def _map_session_type_to_stimulus(session_type: str) -> str:
    """Map a session_type string to a stimulus category."""
    if not session_type:
        return "unknown"
    if any(s in session_type for s in ("gratings", "STAGE_0", "STAGE_1")):
        return "gratings"
    if "images_A" in session_type:
        return "images_A"
    if "images_B" in session_type:
        return "images_B"
    return "unknown"


def get_subject_metadata(subject_id: str) -> pd.DataFrame:
    """Get subject-level metadata for a given mouse from any multiplane-ophys session.

    Pulls subject info (genotype, sex, DOB, etc.) and surgical procedure details
    from the first available session record for the mouse.

    Parameters
    ----------
    subject_id : str
        Mouse subject ID (e.g. "782149").

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns:
        mouse_id, sex, genotype, date_of_birth, species, source,
        breeding_group, maternal_id, maternal_genotype, paternal_id,
        paternal_genotype, cage_id, room_id, surgery_date,
        craniotomy_type, headframe_type, well_type,
        bregma_to_lambda_mm, experimenter
    """
    client = MetadataDbClient(
        host=API_GATEWAY_HOST,
        database=DATABASE,
        collection=COLLECTION,
    )

    records = client.retrieve_docdb_records(
        filter_query={"name": {"$regex": f"^multiplane-ophys_{subject_id}_"}},
        projection={
            "subject": 1,
            "procedures": 1,
            "_id": 0,
        },
        limit=1,
    )

    if not records:
        return pd.DataFrame()

    record = records[0]
    subj = record.get("subject", {}) or {}
    proc = record.get("procedures", {}) or {}

    # Parse breeding info
    breeding = subj.get("breeding_info", {}) or {}
    housing = subj.get("housing", {}) or {}

    # Parse surgical procedure details
    surgery_date = None
    craniotomy_type = None
    headframe_type = None
    well_type = None
    bregma_to_lambda_mm = None
    experimenter = None

    for sp in proc.get("subject_procedures", []):
        if sp.get("procedure_type") == "Surgery":
            surgery_date = sp.get("start_date")
            experimenter = sp.get("experimenter_full_name")
            for p in sp.get("procedures", []):
                if p.get("procedure_type") == "Craniotomy":
                    craniotomy_type = p.get("craniotomy_type")
                    bregma_to_lambda_mm = p.get("bregma_to_lambda_distance")
                elif p.get("procedure_type") == "Headframe":
                    headframe_type = p.get("headframe_type")
                    well_type = p.get("well_type")

    row = {
        "subject_id": subject_id,
        "sex": subj.get("sex"),
        "genotype": subj.get("genotype"),
        "date_of_birth": subj.get("date_of_birth"),
        "species": (subj.get("species") or {}).get("name"),
        "source": (subj.get("source") or {}).get("name"),
        "breeding_group": breeding.get("breeding_group"),
        "maternal_id": breeding.get("maternal_id"),
        "maternal_genotype": breeding.get("maternal_genotype"),
        "paternal_id": breeding.get("paternal_id"),
        "paternal_genotype": breeding.get("paternal_genotype"),
        "cage_id": housing.get("cage_id"),
        "room_id": housing.get("room_id"),
        "surgery_date": surgery_date,
        "craniotomy_type": craniotomy_type,
        "headframe_type": headframe_type,
        "well_type": well_type,
        "bregma_to_lambda_mm": bregma_to_lambda_mm,
        "experimenter": experimenter,
        "mouse_id": subject_id,
    }

    return pd.DataFrame([row])


def get_all_asset_names(subject_id: str) -> pd.DataFrame:
    """Get names of all data assets (raw and derived) for a given mouse.

    Parameters
    ----------
    subject_id : str
        Mouse subject ID (e.g. "782149").

    Returns
    -------
    pd.DataFrame
        One row per asset with columns:
        mouse_id, asset_name, data_level, s3_location
    """
    client = MetadataDbClient(
        host=API_GATEWAY_HOST,
        database=DATABASE,
        collection=COLLECTION,
    )

    records = client.retrieve_docdb_records(
        filter_query={"name": {"$regex": f"^multiplane-ophys_{subject_id}_"}},
        projection={
            "name": 1,
            "data_description.data_level": 1,
            "location": 1,
            "_id": 0,
        },
        limit=0,
        paginate=True,
    )

    rows = []
    for record in records:
        dd = record.get("data_description", {}) or {}
        rows.append({
            "mouse_id": subject_id,
            "subject_id": subject_id,
            "asset_name": record.get("name"),
            "data_level": dd.get("data_level"),
            "s3_location": record.get("location"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("asset_name").reset_index(drop=True)
    return df


def get_derived_assets(raw_asset_name: str) -> pd.DataFrame:
    """Get all derived assets associated with a given raw data asset.

    Derived asset names follow the pattern: <raw_asset_name>_<pipeline>_<datetime>.

    Parameters
    ----------
    raw_asset_name : str
        Name of the raw data asset (e.g.
        "multiplane-ophys_782149_2025-04-10_09-26-55").

    Returns
    -------
    pd.DataFrame
        One row per derived asset with columns:
        raw_asset_name, derived_asset_name, pipeline_name, data_level,
        s3_location, code_ocean_ids
    """
    client = MetadataDbClient(
        host=API_GATEWAY_HOST,
        database=DATABASE,
        collection=COLLECTION,
    )

    # Derived assets start with the raw name followed by an underscore
    records = client.retrieve_docdb_records(
        filter_query={"name": {"$regex": f"^{raw_asset_name}_"}},
        projection={
            "name": 1,
            "data_description.data_level": 1,
            "external_links": 1,
            "location": 1,
            "_id": 0,
        },
        limit=0,
        paginate=True,
    )

    rows = []
    for record in records:
        derived_name = record.get("name", "")
        # Extract pipeline name: everything between raw name and the trailing timestamp
        suffix = derived_name[len(raw_asset_name) + 1:]  # strip "<raw>_"
        # Pipeline name is everything before the last _YYYY-MM-DD_HH-MM-SS
        parts = suffix.rsplit("_", 2)
        pipeline_name = parts[0] if len(parts) >= 3 else suffix

        dd = record.get("data_description", {}) or {}
        ext_links = record.get("external_links", {}) or {}
        co_ids = ext_links.get("Code Ocean", [])

        rows.append({
            "raw_asset_name": raw_asset_name,
            "derived_asset_name": derived_name,
            "pipeline_name": pipeline_name,
            "data_level": dd.get("data_level"),
            "s3_location": record.get("location"),
            "code_ocean_ids": ", ".join(co_ids) if co_ids else None,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("derived_asset_name").reset_index(drop=True)
    return df


# ===========================================================================
# OPTIMIZED FUNCTIONS FOR MULTIPLE MICE
# ===========================================================================
# The functions below are optimized to query multiple mice at once using a
# single database call with regex alternation, making them much faster than
# calling the single-mouse functions in a loop.
# ===========================================================================


def get_all_imaging_plane_metadata(
    subject_ids,
    raw_only: bool = True
) -> pd.DataFrame:
    """Get combined imaging plane metadata for multiple mice (OPTIMIZED).

    Makes a single database query for all mice instead of N separate queries.

    Parameters
    ----------
    subject_ids : list or dict
        List of mouse subject IDs (e.g. ["782149", "782150"]) or dictionary
        where values or keys are subject IDs.
    raw_only : bool
        If True (default), only return raw acquisition sessions.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with imaging plane metadata for all mice.
        Empty DataFrame if no data found.
    """
    # Handle dict input: try keys first, then values
    if isinstance(subject_ids, dict):
        if subject_ids:
            # Use keys if they look like subject IDs, else use values
            first_key = next(iter(subject_ids.keys()))
            if isinstance(first_key, str) and (first_key.isdigit() or '-' in first_key):
                subject_ids = list(subject_ids.keys())
            else:
                subject_ids = list(subject_ids.values())
        else:
            subject_ids = []

    # Convert to list if needed
    if not isinstance(subject_ids, list):
        subject_ids = list(subject_ids)

    if not subject_ids:
        return pd.DataFrame()

    # Make a single database query for all mice using regex alternation
    client = MetadataDbClient(
        host=API_GATEWAY_HOST,
        database=DATABASE,
        collection=COLLECTION,
    )

    # Build regex pattern: ^multiplane-ophys_(mouse1|mouse2|mouse3)_...
    subject_pattern = "|".join(subject_ids)
    if raw_only:
        name_pattern = (
            f"^multiplane-ophys_({subject_pattern})_"
            r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
        )
    else:
        name_pattern = f"^multiplane-ophys_({subject_pattern})"

    records = client.retrieve_docdb_records(
        filter_query={"name": {"$regex": name_pattern}},
        projection={
            "name": 1,
            "session": 1,
            "_id": 0,
        },
        limit=0,
        paginate=True,
    )

    # Parse records (same logic as get_imaging_plane_metadata)
    rows = []
    for record in records:
        session_name = record.get("name")
        session = record.get("session")
        if not session:
            continue

        # Extract subject_id from session name
        # Format: multiplane-ophys_<subject_id>_<date>_<time>...
        parts = session_name.split("_")
        subject_id = parts[1] if len(parts) > 1 else None

        session_start = session.get("session_start_time", "")
        session_date = session_start[:10] if session_start else ""

        for stream in session.get("data_streams", []):
            for fov in stream.get("ophys_fovs", []):
                targeted = fov.get("targeted_structure", "")
                fov_index = fov.get("index")
                plane_name = f"{targeted}_{fov_index}" if targeted and fov_index is not None else None
                rows.append({
                    "subject_id": subject_id,
                    "session_name": session_name,
                    "session_key": f"{subject_id}_{session_date}",
                    "session_date": session_date,
                    "plane_id": plane_name,
                    "imaging_depth_um": fov.get("imaging_depth"),
                    "targeted_structure": targeted,
                    "fov_index": fov_index,
                    "coupled_fov_index": fov.get("coupled_fov_index"),
                    "fov_width": fov.get("fov_width"),
                    "fov_height": fov.get("fov_height"),
                    "fov_scale_factor_um_per_px": fov.get("fov_scale_factor"),
                    "frame_rate_hz": fov.get("frame_rate"),
                    "power_pct": fov.get("power"),
                    "power_ratio_pct": fov.get("power_ratio"),
                    "scanfield_z_um": fov.get("scanfield_z"),
                    "scanimage_roi_index": fov.get("scanimage_roi_index"),
                    "mouse_id": subject_id,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["mouse_id", "session_date", "fov_index"]
        ).reset_index(drop=True)

    return df


def get_all_session_metadata(
    subject_ids,
    raw_only: bool = True
) -> pd.DataFrame:
    """Get combined session metadata for multiple mice (OPTIMIZED).

    Makes a single database query for all mice instead of N separate queries.

    Parameters
    ----------
    subject_ids : list or dict
        List of mouse subject IDs (e.g. ["782149", "782150"]) or dictionary
        where values or keys are subject IDs.
    raw_only : bool
        If True (default), only return raw acquisition sessions.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with session metadata for all mice.
        Empty DataFrame if no data found.
    """
    # Handle dict input: try keys first, then values
    if isinstance(subject_ids, dict):
        if subject_ids:
            # Use keys if they look like subject IDs, else use values
            first_key = next(iter(subject_ids.keys()))
            if isinstance(first_key, str) and (first_key.isdigit() or '-' in first_key):
                subject_ids = list(subject_ids.keys())
            else:
                subject_ids = list(subject_ids.values())
        else:
            subject_ids = []

    # Convert to list if needed
    if not isinstance(subject_ids, list):
        subject_ids = list(subject_ids)

    if not subject_ids:
        return pd.DataFrame()

    # Make a single database query for all mice
    client = MetadataDbClient(
        host=API_GATEWAY_HOST,
        database=DATABASE,
        collection=COLLECTION,
    )

    # Build regex pattern for all mice
    subject_pattern = "|".join(subject_ids)
    if raw_only:
        name_pattern = (
            f"^multiplane-ophys_({subject_pattern})_"
            r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
        )
    else:
        name_pattern = f"^multiplane-ophys_({subject_pattern})"

    # Get all session records in one query
    records = client.retrieve_docdb_records(
        filter_query={"name": {"$regex": name_pattern}},
        projection={
            "name": 1,
            "session": 1,
            "data_description": 1,
            "external_links": 1,
            "location": 1,
            "schema_version": 1,
            "_id": 0,
        },
        limit=0,
        paginate=True,
    )

    # Parse records (same logic as get_session_metadata)
    rows = []
    for record in records:
        session_name = record.get("name")
        session = record.get("session")
        if not session:
            continue

        # Extract subject_id from session name
        parts = session_name.split("_")
        subject_id = parts[1] if len(parts) > 1 else None

        dd = record.get("data_description", {}) or {}
        session_start = session.get("session_start_time", "")
        session_end = session.get("session_end_time", "")
        session_date = session_start[:10] if session_start else ""

        # Compute duration
        duration_min = None
        if session_start and session_end:
            try:
                start = datetime.fromisoformat(session_start)
                end = datetime.fromisoformat(session_end)
                duration_min = round((end - start).total_seconds() / 60, 1)
            except (ValueError, TypeError):
                pass

        # Count planes and coupled groups
        num_planes = 0
        coupled_indices = set()
        for stream in session.get("data_streams", []):
            fovs = stream.get("ophys_fovs", [])
            num_planes += len(fovs)
            for fov in fovs:
                ci = fov.get("coupled_fov_index")
                if ci is not None:
                    coupled_indices.add(ci)

        # Collect stimulus names
        stim_names = []
        for epoch in session.get("stimulus_epochs", []):
            name = epoch.get("stimulus_name")
            if name and name not in stim_names:
                stim_names.append(name)

        # External links / Code Ocean asset IDs
        ext_links = record.get("external_links", {}) or {}
        co_ids = ext_links.get("Code Ocean", [])
        raw_asset_id = co_ids[0] if co_ids else None

        # Modality list
        modalities = dd.get("modality", [])
        modality_str = ", ".join(m.get("abbreviation", "") for m in modalities) if modalities else None

        # Platform
        platform = dd.get("platform", {}) or {}

        rows.append({
            "subject_id": subject_id,
            "session_name": session_name,
            "session_key": f"{subject_id}_{session_date}",
            "session_date": session_date,
            "session_type": session.get("session_type"),
            "project_name": dd.get("project_name"),
            "rig_id": session.get("rig_id"),
            "num_planes": num_planes,
            "num_coupled_groups": len(coupled_indices),
            "session_start_time": session_start,
            "session_end_time": session_end,
            "session_duration_min": duration_min,
            "stimulus_names": ", ".join(stim_names) if stim_names else None,
            "mouse_platform_name": session.get("mouse_platform_name"),
            "platform": platform.get("abbreviation"),
            "modalities": modality_str,
            "data_level": dd.get("data_level"),
            "schema_version": record.get("schema_version"),
            "s3_location": record.get("location"),
            "raw_asset_id": raw_asset_id,
            "mouse_id": subject_id,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["mouse_id", "session_date"]).reset_index(drop=True)

        # --- Computed exposure counters (per mouse) ---
        df["session_type_exposures"] = (
            df.groupby(["mouse_id", "session_type"]).cumcount() + 1
        )
        df["stimulus"] = df["session_type"].apply(_map_session_type_to_stimulus)
        df["stimulus_exposures"] = (
            df.groupby(["mouse_id", "stimulus"]).cumcount() + 1
        )

        # --- Look up derived assets for all sessions in one query ---
        if raw_only:
            processed_pattern = f"^multiplane-ophys_({subject_pattern})_.*_processed_"
            processed_records = client.retrieve_docdb_records(
                filter_query={"name": {"$regex": processed_pattern}},
                projection={"name": 1, "external_links": 1, "_id": 0},
                limit=0,
                paginate=True,
            )

            processed_map = defaultdict(list)
            for rec in processed_records:
                pname = rec.get("name", "")
                co = (rec.get("external_links") or {}).get("Code Ocean", [])
                asset_id = co[0] if co else None
                for raw_name in df["session_name"]:
                    if pname.startswith(raw_name + "_processed_"):
                        processed_map[raw_name].append((pname, asset_id))
            latest_processed = {
                k: sorted(v)[-1] for k, v in processed_map.items()
            }
            df["derived_asset_name"] = df["session_name"].map(
                {k: v[0] for k, v in latest_processed.items()}
            )
            df["derived_asset_id"] = df["session_name"].map(
                {k: v[1] for k, v in latest_processed.items()}
            )

            # --- dF/F long_window from processed assets ---
            processed_asset_ids = df["derived_asset_id"].dropna().tolist()
            if processed_asset_ids:
                dff_results = client.aggregate_docdb_records([
                    {"$match": {
                        "external_links.Code Ocean": {"$in": processed_asset_ids},
                    }},
                    {"$project": {
                        "_id": 0,
                        "code_ocean_id": {
                            "$arrayElemAt": ["$external_links.Code Ocean", 0]
                        },
                        "df_f_params": {
                            "$filter": {
                                "input": "$processing.processing_pipeline.data_processes",
                                "as": "process",
                                "cond": {"$eq": ["$$process.name", "dF/F estimation"]},
                            }
                        },
                    }},
                    {"$project": {
                        "code_ocean_id": 1,
                        "long_window": {
                            "$arrayElemAt": ["$df_f_params.parameters.long_window", 0]
                        },
                    }},
                ])
                dff_map = {
                    r["code_ocean_id"]: r.get("long_window")
                    for r in dff_results
                    if r.get("long_window") is not None
                }
                df["dff_long_window"] = df["derived_asset_id"].map(dff_map)
            else:
                df["dff_long_window"] = None

            # --- DLC eye-tracking derived assets ---
            dlc_pattern = f"^multiplane-ophys_({subject_pattern})_.*dlc-eye"
            dlc_records = client.retrieve_docdb_records(
                filter_query={"name": {"$regex": dlc_pattern}},
                projection={"name": 1, "external_links": 1, "_id": 0},
                limit=0,
                paginate=True,
            )
            dlc_map = defaultdict(list)
            for rec in dlc_records:
                dname = rec.get("name", "")
                co = (rec.get("external_links") or {}).get("Code Ocean", [])
                dlc_id = co[0] if co else None
                for raw_name in df["session_name"]:
                    if dname.startswith(raw_name + "_dlc-eye"):
                        dlc_map[raw_name].append((dname, dlc_id))
            latest_dlc = {k: sorted(v)[-1] for k, v in dlc_map.items()}
            df["dlc_eye_asset_name"] = df["session_name"].map(
                {k: v[0] for k, v in latest_dlc.items()}
            )
            df["dlc_eye_asset_id"] = df["session_name"].map(
                {k: v[1] for k, v in latest_dlc.items()}
            )

    return df


def get_all_subject_metadata(subject_ids) -> pd.DataFrame:
    """Get combined subject metadata for multiple mice (OPTIMIZED).

    Makes a single database query for all mice instead of N separate queries.

    Parameters
    ----------
    subject_ids : list or dict
        List of mouse subject IDs (e.g. ["782149", "782150"]) or dictionary
        where values or keys are subject IDs.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with subject metadata for all mice.
        Empty DataFrame if no data found.
    """
    # Handle dict input: try keys first, then values
    if isinstance(subject_ids, dict):
        if subject_ids:
            # Use keys if they look like subject IDs, else use values
            first_key = next(iter(subject_ids.keys()))
            if isinstance(first_key, str) and (first_key.isdigit() or '-' in first_key):
                subject_ids = list(subject_ids.keys())
            else:
                subject_ids = list(subject_ids.values())
        else:
            subject_ids = []

    # Convert to list if needed
    if not isinstance(subject_ids, list):
        subject_ids = list(subject_ids)

    if not subject_ids:
        return pd.DataFrame()

    # Make a single database query for all mice
    client = MetadataDbClient(
        host=API_GATEWAY_HOST,
        database=DATABASE,
        collection=COLLECTION,
    )

    # Build regex pattern for all mice
    subject_pattern = "|".join(subject_ids)
    name_pattern = f"^multiplane-ophys_({subject_pattern})_"

    # Get one record per subject (limit to first session found per mouse)
    records = client.retrieve_docdb_records(
        filter_query={"name": {"$regex": name_pattern}},
        projection={
            "name": 1,
            "subject": 1,
            "procedures": 1,
            "_id": 0,
        },
        limit=0,
        paginate=True,
    )

    # Group by subject_id and take first record for each
    subject_records = {}
    for record in records:
        session_name = record.get("name", "")
        parts = session_name.split("_")
        subject_id = parts[1] if len(parts) > 1 else None

        if subject_id and subject_id not in subject_records:
            subject_records[subject_id] = record

    # Parse records (same logic as get_subject_metadata)
    rows = []
    for subject_id, record in subject_records.items():
        subj = record.get("subject", {}) or {}
        proc = record.get("procedures", {}) or {}

        # Parse breeding info
        breeding = subj.get("breeding_info", {}) or {}
        housing = subj.get("housing", {}) or {}

        # Parse surgical procedure details
        surgery_date = None
        craniotomy_type = None
        headframe_type = None
        well_type = None
        bregma_to_lambda_mm = None
        experimenter = None

        for sp in proc.get("subject_procedures", []):
            if sp.get("procedure_type") == "Surgery":
                surgery_date = sp.get("start_date")
                experimenter = sp.get("experimenter_full_name")
                for p in sp.get("procedures", []):
                    if p.get("procedure_type") == "Craniotomy":
                        craniotomy_type = p.get("craniotomy_type")
                        bregma_to_lambda_mm = p.get("bregma_to_lambda_distance")
                    elif p.get("procedure_type") == "Headframe":
                        headframe_type = p.get("headframe_type")
                        well_type = p.get("well_type")

        rows.append({
            "subject_id": subject_id,
            "sex": subj.get("sex"),
            "genotype": subj.get("genotype"),
            "date_of_birth": subj.get("date_of_birth"),
            "species": (subj.get("species") or {}).get("name"),
            "source": (subj.get("source") or {}).get("name"),
            "breeding_group": breeding.get("breeding_group"),
            "maternal_id": breeding.get("maternal_id"),
            "maternal_genotype": breeding.get("maternal_genotype"),
            "paternal_id": breeding.get("paternal_id"),
            "paternal_genotype": breeding.get("paternal_genotype"),
            "cage_id": housing.get("cage_id"),
            "room_id": housing.get("room_id"),
            "surgery_date": surgery_date,
            "craniotomy_type": craniotomy_type,
            "headframe_type": headframe_type,
            "well_type": well_type,
            "bregma_to_lambda_mm": bregma_to_lambda_mm,
            "experimenter": experimenter,
            "mouse_id": subject_id,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("mouse_id").reset_index(drop=True)

    return df


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)

    print("=== All assets for mouse 782149 ===")
    adf = get_all_asset_names("782149")
    print(f"{len(adf)} total assets")
    print(adf.to_string(index=False))

    print("\n=== Derived assets for one session ===")
    ddf = get_derived_assets("multiplane-ophys_782149_2025-04-10_09-26-55")
    print(f"{len(ddf)} derived assets")
    print(ddf.to_string(index=False))

    print("\n=== Create and save metadata tables for all mice ===")
    mouse_dict = {
    '767022': 'Rosemary',
    '755252': 'Lavender',
    '767018': 'Oregano',
    '788406': 'Sage',
    '790322': 'Laurel',
    '782149': 'Lemongrass',
    '800792': 'Chamomile',
    '800995': 'Dandilion',
    '804363': 'Nettle',
    }

    mice = get_all_subject_metadata(mouse_dict)
    sessions = get_all_session_metadata(mouse_dict)
    imaging_planes = get_all_imaging_plane_metadata(mouse_dict)

    imaging_planes.to_csv('ophys_imaging_planes_metadata.csv', index=False)
    sessions.to_csv('ophys_sessions_metadata.csv', index=False)
    mice.to_csv('mouse_metadata.csv', index=False)
