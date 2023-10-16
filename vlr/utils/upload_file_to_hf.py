
if __name__ == "__main__":
    
    from huggingface_hub import HfApi, login
    api = HfApi()

    # Upload all the content from the local folder to your remote Space.
    # By default, files are uploaded at the root of the repo

    api.upload_file(
        path_or_fileobj="/home/duytran/Downloads/vlr/active_speaker/ha_hips_output.zip",
        path_in_repo="active_speaker/ha_hips_output.zip",
        repo_id="fptu/vlr",
        repo_type="dataset",
    )
