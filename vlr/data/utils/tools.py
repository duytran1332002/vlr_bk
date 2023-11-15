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
            print(f"Removing old files in {dir}...")
            shutil.rmtree(dir)
        if not os.path.exists(dir):
            os.makedirs(dir)


def zip_dir(zip_dir: str, overwrite: bool = False):
    """
    Zip directory.
    :param dir:         Path to directory.
    :param zip_path:    Path to zip file.
    """
    if overwrite and os.path.exists(zip_dir + ".zip"):
        print("Removing old zip file...")
        os.remove(zip_dir + ".zip")
    print("Making directory archive...")
    shutil.make_archive(zip_dir, "zip", os.path.dirname(zip_dir), os.path.basename(zip_dir))
