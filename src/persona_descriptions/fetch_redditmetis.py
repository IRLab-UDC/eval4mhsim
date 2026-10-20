import asyncio
import base64
import gzip
import json
import os

import praw
import requests
from playwright.async_api import async_playwright

USERNAMES_FILE = "data/usernames"
OUTPUT = "data/redditmetis.jsonl"
HEADERS = {"Origin": "https://redditmetis.com", "Referer": "https://redditmetis.com/", "User-Agent": "Mozilla/5.0"}

reddit = praw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],
    client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    user_agent="linux:reddit.user.collector:v1.0.0",
)


def try_cache(username):
    url = f"https://gm117fneof.execute-api.us-east-2.amazonaws.com/check?username={username}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    if data.get("exists") and data.get("result"):
        data["result"] = json.loads(gzip.decompress(base64.b64decode(data["result"])))
        return data
    return None


async def trigger_browser(username):
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--disable-blink-features=AutomationControlled"])
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 900},
        )
        page = await ctx.new_page()
        await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        await page.goto(f"https://redditmetis.com/user/{username}", wait_until="networkidle", timeout=120000)
        await page.wait_for_timeout(90000)
        await browser.close()


with open(USERNAMES_FILE) as f:
    usernames = [line.strip() for line in f if line.strip()]

with open(OUTPUT, "a") as out:
    for username in usernames:
        data = try_cache(username)
        if data is None:
            asyncio.run(trigger_browser(username))
            data = try_cache(username)

        if not data:
            print(f"No data for {username}, skipping.")
            continue

        about = reddit.request(method="GET", path=f"user/{username}/about")["data"]
        data["about"] = about
        data["username"] = username

        out.write(json.dumps(data) + "\n")
        print(f"Saved {username}")
