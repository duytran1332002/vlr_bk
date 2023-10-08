# Download Video From Tiktok
Run the following code to install the necessary library
```python
pip install tiktokapipy
python -m playwright install
```
Paste the id of channel from tiktok to the code
```python
if __name__ == "__main__":
    user_id = '@idchannel' # change here
    directory = '/home/duytran/Downloads/new_raw_video'

    output_path = os.path.join(directory, user_id)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    asyncio.run(down_video_tiktok_from_user(user_id=user_id, output_path=output_path))
```
And then, run the file download_tiktok_videos.py
```python
python donwload_tiktok_videos.py
```
# Extract Active Speaker

# Slice video
What this task do:
* Set fps of video to 25
* Cut video into smaller segment of 3s and overlap 1s
* Split audio and muted video from the original video

How to use:
1. Follow instructions in the `slice.py` file to modify the code
2. Run the command below to start slicing
```
python vlr/data/task/slice.py
```

# Denoise audio
What this task do:
* Denoise audio files

How to use:
1. Follow instructions in the `denoise.py` file to modify the code
2. Run the command below to start denoising
```
python vlr/data/task/denoise.py
```

# Transcribe audio
What this task do:
* Transcribe audio
* Remove records that can not be transcribe

How to use:
1. Follow instructions in the `transcribe.py` file to modify the code
2. Run the command below to start transcribing
```
python vlr/data/task/transcribe.py
```

# Crop mouth region
What this task do:
* Crop mouth region in muted videos
* Remove records that mouth regions can not be extracted from

How to use:
1. Follow instructions in the `crop.py` file to modify the code
2. Run the command below to start cropping
```
python vlr/data/task/crop.py
```

# Do statistics
What this task do:
* Count total duration
* Count number of vocabularies
* Count number of words

How to use:
1. Follow instructions in the `statistics.py` file to modify the code
2. Run the command below to start doing statistics
```
python vlr/data/task/statistics.py
```

# Push to HuggingFace
What this task do:
* Zip audio, denoised audio, muted video, transcript, mouth-cropped video directories of channels
* Push these zipped files to HuggingFace

How to use:
1. Follow instructions in the `push.py` file to modify the code
2. Run the command below and provide `HuggingFace access token` to start pushing to HuggingFace
```
python vlr/data/task/push.py --token <access-token>
```