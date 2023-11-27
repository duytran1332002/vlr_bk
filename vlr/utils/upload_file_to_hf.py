
if __name__ == "__main__":
    
    from huggingface_hub import HfApi, login
    api = HfApi()

    # Upload all the content from the local folder to your remote Space.
    # By default, files are uploaded at the root of the repo

    # api.upload_file(
    #     path_or_fileobj="/home/duytran/Desktop/vlr/outputs/1st_200h_audio_scratch/model_avg_10.pth",
    #     path_in_repo="phase1/audio_200h_10epochs_scratch.pth",
    #     repo_id="fptu/vlr-e2e",
    #     repo_type="model",
    # )

    api.upload_file(
        path_or_fileobj="/home/duytran/Downloads/active_speaker/lepthelittlekid_output.zip",
        path_in_repo="active_speaker/lepthelittlekid_output.zip",
        repo_id="fptu/vlr",
        repo_type="dataset",
    )

    api.upload_file(
        path_or_fileobj="/home/duytran/Downloads/active_speaker/nqs.kinhte_output.zip",
        path_in_repo="active_speaker/nqs.kinhte_output.zip",
        repo_id="fptu/vlr",
        repo_type="dataset",
    )

    api.upload_file(
        path_or_fileobj="/home/duytran/Downloads/active_speaker/wesleyn313_output.zip",
        path_in_repo="active_speaker/wesleyn313_output.zip",
        repo_id="fptu/vlr",
        repo_type="dataset",
    )

    api.upload_file(
        path_or_fileobj="/home/duytran/Downloads/active_speaker/thayqueo_output.zip",
        path_in_repo="active_speaker/thayqueo_output.zip",
        repo_id="fptu/vlr",
        repo_type="dataset",
    )
