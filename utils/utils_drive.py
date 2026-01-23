from pathlib import Path

def download_gdrive_folder(folder_url: str, output_dir: str, quiet: bool = False):
    """
    Download all files from a Google Drive shared folder link into `output_dir`.
    """
    import gdown  # local import so utils can be imported without gdown installed

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    downloaded = gdown.download_folder(
        url=folder_url,
        output=str(out),
        quiet=quiet,
        use_cookies=False,
    )
    return downloaded
