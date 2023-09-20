import time
import os
import requests
import asyncio
from bs4 import BeautifulSoup
from tiktokapipy.async_api import AsyncTikTokAPI
from urllib.request import urlopen
import tqdm

def downloadVideo(link, id, output_path):
    cookies = {
    '_gid': 'GA1.2.1696976353.1695097931',
    '_ga': 'GA1.2.1298546876.1695097931',
    '_gat_UA-3524196-6': '1',
    '__gads': 'ID=57ffb4abac092a46-2263b1ebe1e30043:T=1695097945:RT=1695183294:S=ALNI_MZAYVSXcFESioNjN20c3oB088KSPg',
    '__gpi': 'UID=00000c4b7d9c7a22:T=1695097945:RT=1695183294:S=ALNI_MaoFpElMja6w0EwatwWmSYqk_fZ7g',
    '_ga_ZSF3D6YSLC': 'GS1.1.1695183279.5.0.1695183286.0.0.0',
    }

    headers = {
        'authority': 'ssstik.io',
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        # 'cookie': '_gid=GA1.2.1696976353.1695097931; _ga=GA1.2.1298546876.1695097931; _gat_UA-3524196-6=1; __gads=ID=57ffb4abac092a46-2263b1ebe1e30043:T=1695097945:RT=1695183294:S=ALNI_MZAYVSXcFESioNjN20c3oB088KSPg; __gpi=UID=00000c4b7d9c7a22:T=1695097945:RT=1695183294:S=ALNI_MaoFpElMja6w0EwatwWmSYqk_fZ7g; _ga_ZSF3D6YSLC=GS1.1.1695183279.5.0.1695183286.0.0.0',
        'hx-current-url': 'https://ssstik.io/vi',
        'hx-request': 'true',
        'hx-target': 'target',
        'hx-trigger': '_gcaptcha_pt',
        'origin': 'https://ssstik.io',
        'referer': 'https://ssstik.io/vi',
        'sec-ch-ua': '"Microsoft Edge";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.31',
    }

    params = {
        'url': 'dl',
    }

    data = {
        'id': link,
        'locale': 'vi',
        'tt': 'Y252eFVh',
    }
    
    response = requests.post('https://ssstik.io/abc', params=params, cookies=cookies, headers=headers, data=data)
    time.sleep(1)
    downloadSoup = BeautifulSoup(response.text, "html.parser")

    try:
        downloadLink = downloadSoup.a["href"]

        mp4File = urlopen(downloadLink)
        # Feel free to change the download directory
        with open(os.path.join(output_path, f"{id}.mp4"), "wb") as output:
            while True:
                data = mp4File.read(4096)
                if data:
                    output.write(data)
                else:
                    break
    except:
        print(f"Error: Unable to download video from {link}")

async def do_video_tiktok(user_id, video_limit=None):
    async with AsyncTikTokAPI() as api:
        result = []
        if video_limit is not None:
            user = await api.user(user_id, video_limit=video_limit)
        else:
            user = await api.user(user_id)
        async for video in user.videos:
            print(video.id)
            result.append(video.id)
        return result

if __name__ == "__main__":
    user_id = 'zing.podcast'
    output_path = os.path.join('D:/LIP READING/new_raw_video', user_id)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    urlDownloads = asyncio.run(do_video_tiktok(user_id=user_id))
    
    for idx, id in enumerate(tqdm.tqdm(urlDownloads)):
        url = f'https://www.tiktok.com/@{user_id}/video/{id}'
        downloadVideo(link=url, id=id, output_path=output_path)
        time.sleep(4)