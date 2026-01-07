from pathlib import Path

def download_gdrive_folder(folder_url: str, output_dir: str, quiet: bool = False):
    """
    Download all files from a Google Drive shared folder link into `output_dir`.

    Works best when the folder is shared publicly (or accessible without extra auth prompts).
    Returns a list of downloaded file paths.

    Parameters
    ----------
    folder_url : str
        Google Drive folder URL like:
        https://drive.google.com/drive/folders/<FOLDER_ID>?usp=sharing
    output_dir : str
        Local directory where files will be saved
    quiet : bool
        Reduce output logs

    Notes
    -----
    - For private folders requiring login/OAuth, use the Google Drive API instead.
    - Google Docs/Sheets/Slides may not download as expected without export settings.
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
