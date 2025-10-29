from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict, List, Union
from pathlib import Path
import json, os, uuid
from datetime import datetime

# Categories map
_ALLOWED_CATEGORIES: Dict[tuple, set] = {
    ("acquisition", "plane"): {
        "image_quality",
        "z_drift",
        "bleaching",
    },
    ("acquisition", "session"): {
        "lick_detection",
        "reward_delivery",
        "running_wheel",
        "eye_cam_quality",
    },
    ("processing", "plane"): {
        "motion_correction",
        "decrosstalk",
        "neuropil_correction",
        "dff_calculation",
        "events_deconvolution",
    },
    ("processing", "session"): {"pupil_tracking"},
}

@dataclass(frozen=True)
class QCEntry:
    id: str
    timestamp: str
    session_key: str
    qc_type: str
    level: str
    category: str
    plane_id: Optional[str] = None
    status: str = "fail"
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    asset_name: Optional[str] = None  # processed data asset identifier
    asset_id: Optional[str] = None    # unique id for processed data asset
    entry_type: str = "qc"  # always "qc" for these records

@dataclass(frozen=True)
class AmendmentEntry:
    id: str
    timestamp: str
    entry_type: str  # "amendment"
    target_id: str   # QCEntry.id being amended
    action: str      # "revert" | "update_status"
    new_status: Optional[str] = None
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class QCStore:
    _PATH = Path(__file__).resolve().parent / "qc_data" / "qc_data.json"

    @classmethod
    def _ensure(cls):
        if not cls._PATH.parent.exists():
            cls._PATH.parent.mkdir(parents=True, exist_ok=True)
        if not cls._PATH.exists():
            cls._atomic_save({"version": 1, "entries": []})

    @staticmethod
    def _atomic_save(data: Dict[str, Any]) -> None:
        tmp = QCStore._PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(QCStore._PATH)

    @classmethod
    def _load_raw(cls) -> Dict[str, Any]:
        cls._ensure()
        try:
            return json.loads(cls._PATH.read_text())
        except json.JSONDecodeError:
            return {"version": 1, "entries": []}

    @classmethod
    def all(cls) -> List[QCEntry]:
        blob = cls._load_raw()
        out: List[QCEntry] = []
        for raw in blob.get("entries", []):
            try:
                out.append(QCEntry(**raw))
            except TypeError:
                continue
        return out

    @classmethod
    def log(
        cls,
        session_key: str,
        qc_type: str,
        level: str,
        category: str,
        plane_id: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status: str = "fail",
        asset_name: Optional[str] = None,
        asset_id: Optional[str] = None,
    ) -> QCEntry:
        qc_type = qc_type.lower()
        level = level.lower()
        category = category.lower()
        status = status.lower()

        allowed = _ALLOWED_CATEGORIES.get((qc_type, level))
        if allowed is None:
            raise ValueError(f"Invalid (qc_type, level): ({qc_type}, {level})")
        if allowed and category not in allowed:
            raise ValueError(f"Category '{category}' not allowed. Allowed: {sorted(allowed)}")
        if level == "plane" and not plane_id:
            raise ValueError("plane_id required for plane-level QC")
        if level == "session" and plane_id is not None:
            # Ignore provided plane_id for session-level QC
            plane_id = None
        if qc_type == "processing" and asset_name is None:
            raise ValueError("asset_name required for processing QC entries")

        entry = QCEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            session_key=session_key,
            qc_type=qc_type,
            level=level,
            category=category,
            plane_id=plane_id,
            status=status,
            message=message,
            details=details,
            asset_name=asset_name,
            asset_id=asset_id,
        )
        blob = cls._load_raw()
        blob.setdefault("entries", []).append(asdict(entry))
        cls._atomic_save(blob)
        return entry

    @classmethod
    def revisit(
        cls,
        original_id: str,
        action: str = "revert",
        new_status: Optional[str] = "not_a_problem",
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AmendmentEntry:
        """
        Append an amendment entry referencing a prior QCEntry.
        action: "revert" (mark as not a problem) or "update_status" (override with new_status).
        new_status: applied if action != revert OR explicit override (default "not_a_problem" for revert).
        """
        action = action.lower()
        if action not in {"revert", "update_status"}:
            raise ValueError("action must be 'revert' or 'update_status'")
        # Validate original exists
        if not any(e.id == original_id for e in cls.qc_entries()):
            raise ValueError(f"Original QCEntry id '{original_id}' not found")
        amend = AmendmentEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            entry_type="amendment",
            target_id=original_id,
            action=action,
            new_status=new_status,
            message=message,
            details=details,
        )
        blob = cls._load_raw()
        blob.setdefault("entries", []).append(asdict(amend))
        cls._atomic_save(blob)
        return amend

    @classmethod
    def qc_entries(cls) -> List[QCEntry]:
        blob = cls._load_raw()
        out: List[QCEntry] = []
        for raw in blob.get("entries", []):
            if raw.get("entry_type", "qc") != "qc":
                continue
            try:
                out.append(QCEntry(**raw))
            except TypeError:
                continue
        return out

    @classmethod
    def amendments(cls) -> List[AmendmentEntry]:
        blob = cls._load_raw()
        out: List[AmendmentEntry] = []
        for raw in blob.get("entries", []):
            if raw.get("entry_type") != "amendment":
                continue
            try:
                out.append(AmendmentEntry(**raw))
            except TypeError:
                continue
        return out

    @classmethod
    def find(
        cls,
        session_key: Optional[str] = None,
        plane_id: Optional[str] = None,
        qc_type: Optional[str] = None,
        level: Optional[str] = None,
        category: Optional[str] = None,
        status: Optional[str] = None,
        apply_amendments: bool = False,
        exclude_reverted: bool = False,
    ) -> List[QCEntry]:
        entries = cls._apply_amendments(apply_amendments, exclude_reverted)
        results: List[QCEntry] = []
        for e in entries:
            if session_key and e.session_key != session_key: continue
            if qc_type and e.qc_type != qc_type.lower(): continue
            if level and e.level != level.lower(): continue
            if category and e.category != category.lower(): continue
            if status and e.status != status.lower(): continue
            # Only apply plane_id filter to plane-level entries
            if plane_id and e.level == "plane" and e.plane_id != plane_id: continue
            results.append(e)
        return results

    @classmethod
    def _apply_amendments(cls, apply_amendments: bool, exclude_reverted: bool) -> List[QCEntry]:
        if not apply_amendments:
            return cls.qc_entries()
        # Map target_id -> last amendment (chronological by append order)
        amendments = {}
        for a in cls.amendments():
            amendments[a.target_id] = a
        adjusted: List[QCEntry] = []
        for e in cls.qc_entries():
            a = amendments.get(e.id)
            if a:
                if a.action == "revert":
                    final_status = a.new_status or "not_a_problem"
                    if exclude_reverted and final_status in {"not_a_problem", "reverted"}:
                        continue
                    # Represent reverted entry with overridden status
                    e = QCEntry(
                        id=e.id,
                        timestamp=e.timestamp,
                        session_key=e.session_key,
                        qc_type=e.qc_type,
                        level=e.level,
                        category=e.category,
                        plane_id=e.plane_id,
                        status=final_status,
                        message=a.message or e.message,
                        details=a.details or e.details,
                        asset_name=e.asset_name,
                        asset_id=e.asset_id,
                        entry_type="qc",
                    )
                elif a.action == "update_status" and a.new_status:
                    e = QCEntry(
                        id=e.id,
                        timestamp=e.timestamp,
                        session_key=e.session_key,
                        qc_type=e.qc_type,
                        level=e.level,
                        category=e.category,
                        plane_id=e.plane_id,
                        status=a.new_status,
                        message=a.message or e.message,
                        details=a.details or e.details,
                        asset_name=e.asset_name,
                        asset_id=e.asset_id,
                        entry_type="qc",
                    )
            adjusted.append(e)
        return adjusted

    @classmethod
    def dataframe(
        cls,
        apply_amendments: bool = False,
        exclude_reverted: bool = False,
    ):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("Install pandas for dataframe output") from e
        entries = cls._apply_amendments(apply_amendments, exclude_reverted)
        rows = []
        for e in entries:
            row = asdict(e)
            # remove entry_type (always qc after processing)
            row.pop("entry_type", None)
            if row["details"] is not None:
                row["details"] = json.dumps(row["details"], sort_keys=True)
            rows.append(row)
        cols = ["id","timestamp","session_key","qc_type","level","category",
                "plane_id","status","message","details","asset_name","asset_id"]
        return pd.DataFrame(rows, columns=cols)

# Example:
# original = QCStore.log(session_key="sess1", qc_type="processing", level="plane",
#                        category="motion_correction", plane_id="p1",
#                        asset_name="motion_stack.tiff", message="Drift high")
# QCStore.revisit(original.id, action="revert", message="False positive after review")
# df = QCStore.dataframe(apply_amendments=True, exclude_reverted=True)
# print(df)