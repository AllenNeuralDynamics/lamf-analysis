import s3fs
import json

def list_files_from_s3_location(s3_path: str):
    fs = s3fs.S3FileSystem(anon=False)
    if not s3_path.startswith("s3://"):
        s3_path = "s3://" + s3_path
    return [p for p in fs.glob(f"{s3_path}/**") if fs.isfile(p)]


def read_json_from_s3(s3_path: str):
    fs = s3fs.S3FileSystem(anon=False)
    if not s3_path.startswith("s3://"):
        s3_path = "s3://" + s3_path
    with fs.open(s3_path, "r") as f:
        return json.load(f)