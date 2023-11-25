from tiktokapipy.async_api import AsyncTikTokAPI
import asyncio

async def do_something():
    async with AsyncTikTokAPI(headless="new") as api:
        user = await api.user("huydao")
        async for video in user.videos:
            print(video)

asyncio.run(do_something())