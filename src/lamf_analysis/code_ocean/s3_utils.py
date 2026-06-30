import s3fs
import json
import re
import fnmatch
from typing import Iterable, Pattern


def _normalize_to_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def list_files_from_s3_location(
    s3_path: str,
    matched_string: str | list[str] = None,
    regexp: str | Pattern | list[str | Pattern] = None,
):
    fs = s3fs.S3FileSystem(anon=False)
    if not s3_path.startswith("s3://"):
        s3_path = "s3://" + s3_path
    files = []
    for p in fs.glob(f"{s3_path}/**"):
        if fs.isfile(p):
            files.append(p if p.startswith("s3://") else "s3://" + p)

    if matched_string is not None:
        matched_strings = [s.lower() for s in _normalize_to_list(matched_string)]

        def _matches_matched_string(file_path: str) -> bool:
            lower_file_path = file_path.lower()
            for pattern in matched_strings:
                if any(wildcard in pattern for wildcard in "*?[]"):
                    if fnmatch.fnmatch(lower_file_path, f"*{pattern}*"):
                        return True
                elif pattern in lower_file_path:
                    return True
            return False

        files = [
            f for f in files if _matches_matched_string(f)
        ]

    if regexp is not None:
        regex_patterns = []
        for pattern in _normalize_to_list(regexp):
            if isinstance(pattern, re.Pattern):
                regex_patterns.append(pattern)
            else:
                regex_patterns.append(re.compile(pattern, re.IGNORECASE))

        files = [f for f in files if any(pattern.search(f) for pattern in regex_patterns)]

    return files


def read_json_from_s3(s3_path: str):
    fs = s3fs.S3FileSystem(anon=False)
    if not s3_path.startswith("s3://"):
        s3_path = "s3://" + s3_path
    with fs.open(s3_path, "r") as f:
        return json.load(f)