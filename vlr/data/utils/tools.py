import os
import shutil


def clean_up(channel_name: str, dirs: list, overwrite: bool = False):
    """
    Clean up.
    :param channel_name:    Channel name.
    :param dirs:            Path to directories.
    :param overwrite:       Overwrite existing files.
    """
    dirs = [os.path.join(dir, channel_name) for dir in dirs]
    for dir in dirs:
        if overwrite and os.path.exists(dir):
            shutil.rmtree(dir)
        if not os.path.exists(dir):
            os.makedirs(dir)


def zip_dir(zip_dir: str):
    """
    Zip directory.
    :param dir:         Path to directory.
    :param zip_path:    Path to zip file.
    """
    if os.path.exists(zip_dir + ".zip"):
        os.remove(zip_dir + ".zip")
    shutil.make_archive(zip_dir, "zip", os.path.dirname(zip_dir), os.path.basename(zip_dir))
