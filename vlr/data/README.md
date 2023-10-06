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
