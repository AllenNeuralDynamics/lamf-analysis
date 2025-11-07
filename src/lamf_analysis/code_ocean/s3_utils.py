import s3fs

def list_files_from_s3_location(s3_path: str):
    fs = s3fs.S3FileSystem(anon=False)
    if not s3_path.startswith("s3://"):
        s3_path = "s3://" + s3_path
    return [p for p in fs.glob(f"{s3_path}/**") if fs.isfile(p)]