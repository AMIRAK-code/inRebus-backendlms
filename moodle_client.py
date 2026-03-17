import aiohttp
import asyncio
import time

class MoodleClient:
    def __init__(self, base_url=None, token=None):
        self.base_url = base_url or "${MOODLE_BASE_URL}/webservice/rest/server.php"
        self.token = token
        self.cache = {}  # In-memory cache
        self.cache_ttl = 60  # Cache TTL in seconds
        self.last_fetched = {}  # Track last fetched time for cache

    async def fetch_courses(self):
        current_time = time.time()
        if "courses" in self.cache and (current_time - self.last_fetched["courses"] < self.cache_ttl):
            return self.cache["courses"]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params={
                    'wstoken': self.token,
                    'wsfunction': 'core_course_get_courses',
                    'moodlewsrestformat': 'json'
                }) as response:
                    response.raise_for_status()
                    data = await response.json()
                    self.cache["courses"] = data
                    self.last_fetched["courses"] = current_time
                    return data
        except Exception as e:
            # Handle error and return stale data if available
            if "courses" in self.cache:
                return self.cache["courses"]  # Return stale data
            raise e

    def course_url(self, course_id):
        return f"${{MOODLE_BASE_URL}}/course/view.php?id={{course_id}}"