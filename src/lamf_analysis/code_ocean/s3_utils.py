import s3fs
import json

def list_files_from_s3_location(s3_path: str, matched_string: str = None):
    fs = s3fs.S3FileSystem(anon=False)
    if not s3_path.startswith("s3://"):
        s3_path = "s3://" + s3_path
    files = [f's3://{p}' for p in fs.glob(f"{s3_path}/**") if fs.isfile(p)]
    if matched_string is not None:
        matched_string = matched_string.lower()
        files = [f for f in files if matched_string in f.lower()]
    return files


def read_json_from_s3(s3_path: str):
    fs = s3fs.S3FileSystem(anon=False)
    if not s3_path.startswith("s3://"):
        s3_path = "s3://" + s3_path
    with fs.open(s3_path, "r") as f:
        return json.load(f)