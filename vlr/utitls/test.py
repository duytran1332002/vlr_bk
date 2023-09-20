from tiktokapipy.async_api import AsyncTikTokAPI

async def do_something():
    async with AsyncTikTokAPI() as api:
        result = []
        user = await api.user('spiderumcareerguide', video_limit=None)
        async for video in user.videos:
            print(video.id)
            result.append(video.id)
        print(len(result))

if __name__ == "__main__":
    import asyncio
    asyncio.run(do_something())