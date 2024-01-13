# Data Structure on HuggingFace
```
fptu
|--- vietnamese-video
     |--- metadata
          |--- channel.parquet
     |--- vietnamese-video.py
|--- vietnamese-speaker-video
     |--- metadata
          |--- channel.parquet
     |--- video
          |--- channel
               |--- id.mp4
     |--- vietnamese-speaker-video.py
|--- vietnamese-speaker-clip
     |--- metadata
          |--- channel.parquet
     |--- audio
          |--- channel
               |--- id.wav
     |--- visual
          |--- channel
               |--- id.mp4
     |--- vietnamese-speaker-clip.py
|--- vietnamese-speaker-lip-clip
     |--- metadata
          |--- channel.parquet
     |--- visual
          |--- channel
               |--- id.mp4
     |--- vietnamese-speaker-lip-clip.py
|--- denoised-vietnamese-audio
     |--- metadata
          |--- channel.parquet
     |--- audio
          |--- channel
               |--- id.wav
     |--- denoised-vietnamese-audio.py
|--- purified-vietnamese-audio
     |--- metadata
          |--- channel.parquet
     |--- transcript
          |--- channel
               |--- id.txt
     |--- purified-vietnamese-audio.py
|--- vlr
     |--- vlr.py
```

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

# Download Video From Youtube
```python
pip install git+https://github.com/pishiko/pytube.git@42a7d8322dd7749a9e950baf6860d115bbeaedfc
```
# Extract Active Speaker

# Process data
There are totally 4 steps in this process, and everything has been made to be automatic. In order to start processing, simply modified the following commang according to each task.
```bash
python vlr/data/tasks/process.py \
     --task task \
     --output-dir path/to/your/output/directory \
     --channel-names /path/to/channel/list.txt \
     --overwrite \
     --upload-to-hub \
     --clean-input \
     --clean-output \
```
* `--task`: Choose 1 in 4 options which are `slice`, `crop`, `denoise`, `transcribe`.
* `--output-dir`: Path to directory that will contain the output data.
* `--channel-names`: Path to the text file that contains channel names you want to process. These names have to be separate by `\n` in the text file.
* `--overwrite`: Include this option if you want to overwrite output data available in the output directory.
* `--upload-to-hub`: Include this option if you want to upload the output data to HuggingFace after processing.
* `--clean-input`: Include this option if you want to remove all downloaded files used to process data.
* `--clean-output`: Include this option if you want to remove all output files after processing.

The following setions will present details of what each task does. It must be noticed that these tasks must follow the given order so as to guarantee no conflicts will happen.

## 1. Slice video
What this task do:
* Set fps of video to 25
* Cut video into smaller segment of 3s and overlap 1s
* Split audio and muted video from the original video

## 2. Crop mouth region
What this task do:
* Crop mouth region in muted videos based on facial landmarks
* Remove records that mouth regions can not be extracted from

## 2. Denoise audio
What this task do:
* Denoise audio files

## 3. Transcribe audio
What this task do:
* Classify if an audio is Vietnamese. Then, remove the audio if the confidence is less than 99%
* Transcribe audio
* Remove records that can not be transcribe or contains gibberish

## Do statistics
What this task do:
* Count total duration
* Count number of vocabularies
* Count number of words
