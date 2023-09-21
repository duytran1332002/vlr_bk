import asyncio
import io
import glob
import os
import urllib.request
from os import path
import tqdm

import aiohttp
from tiktokapipy.async_api import AsyncTikTokAPI
from tiktokapipy.models.video import Video



async def save_slideshow(video: Video):
    # this filter makes sure the images are padded to all the same size
    vf = "\"scale=iw*min(1080/iw\,1920/ih):ih*min(1080/iw\,1920/ih)," \
         "pad=1080:1920:(1080-iw)/2:(1920-ih)/2," \
         "format=yuv420p\""

    for i, image_data in enumerate(video.image_post.images):
        url = image_data.image_url.url_list[-1]
        # this step could probably be done with asyncio, but I didn't want to figure out how
        urllib.request.urlretrieve(url, path.join(directory, f"temp_{video.id}_{i:02}.jpg"))

    urllib.request.urlretrieve(video.music.play_url, path.join(directory, f"temp_{video.id}.mp3"))

    # use ffmpeg to join the images and audio
    command = [
        "ffmpeg",
        "-r 2/5",
        f"-i {directory}/temp_{video.id}_%02d.jpg",
        f"-i {directory}/temp_{video.id}.mp3",
        "-r 30",
        f"-vf {vf}",
        "-acodec copy",
        f"-t {len(video.image_post.images) * 2.5}",
        f"{directory}/temp_{video.id}.mp4",
        "-y"
    ]
    ffmpeg_proc = await asyncio.create_subprocess_shell(
        " ".join(command),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await ffmpeg_proc.communicate()
    generated_files = glob.glob(path.join(directory, f"temp_{video.id}*"))

    if not path.exists(path.join(directory, f"temp_{video.id}.mp4")):
        # optional ffmpeg logging step
        # logging.error(stderr.decode("utf-8"))
        for file in generated_files:
            os.remove(file)
        raise Exception("Something went wrong with piecing the slideshow together")

    with open(path.join(directory, f"temp_{video.id}.mp4"), "rb") as f:
        ret = io.BytesIO(f.read())

    for file in generated_files:
        os.remove(file)

    return ret


async def save_video(video: Video, api: AsyncTikTokAPI):
    # Carrying over this cookie tricks TikTok into thinking this ClientSession was the Playwright instance
    # used by the AsyncTikTokAPI instance
    async with aiohttp.ClientSession(cookies={cookie["name"]: cookie["value"] for cookie in await api.context.cookies() if cookie["name"] == "tt_chain_token"}) as session:
        # Creating this header tricks TikTok into thinking it made the request itself
        async with session.get(video.video.download_addr, headers={"referer": "https://www.tiktok.com/"}) as resp:
            return io.BytesIO(await resp.read())
        
async def download_video(link, output_path):
    async with AsyncTikTokAPI() as api:
        video: Video = await api.video(link)
        if video.image_post:
            downloaded = await save_slideshow(video)
        else:
            downloaded = await save_video(video, api)
        # save the video to a file
        with open(path.join(output_path, f"{video.id}.mp4"), "wb") as f:
            f.write(downloaded.read())
        

async def down_video_tiktok_from_user(user_id, output_path, video_limit=None):
    async with AsyncTikTokAPI() as api:
        if video_limit is not None:
            user = await api.user(user_id, video_limit=video_limit)
            pbar = tqdm.tqdm(total=video_limit)
        else:
            user = await api.user(user_id)
            pbar = tqdm.tqdm(total=user.stats.video_count)
        
        async for video in user.videos:
            link = f"https://www.tiktok.com/@{user_id}/video/{video.id}"
            await download_video(link=link, output_path=output_path)
            pbar.update(1)
            pbar.set_description(f"Downloaded {video.id}")
        pbar.close()
        


if __name__ == "__main__":
    user_id = 'vtvthoitiet'
    directory = '/home/duytran/Downloads/new_raw_video'

    output_path = os.path.join(directory, user_id)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    asyncio.run(down_video_tiktok_from_user(user_id=user_id, output_path=output_path))