import os
import httpx
from datetime import datetime, timedelta
from functools import lru_cache

class MoodleClient:
    def __init__(self):
        self.base_url = os.getenv('MOODLE_BASE_URL')
        self.rest_url = os.getenv('MOODLE_REST_URL', '')
        self.ws_token = os.getenv('MOODLE_WS_TOKEN')
        
        if not self.base_url or not self.ws_token:
            raise ValueError('Missing configuration: MOODLE_BASE_URL and MOODLE_WS_TOKEN must be set')

    @lru_cache(maxsize=128)
    def fetch_courses(self):
        url = f'{self.base_url}/api/moodle/courses?ws_token={self.ws_token}'
        expiration_time = timedelta(seconds=60)
        last_cached_time = datetime.now() - expiration_time

        try:
            response = httpx.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError:
            if (datetime.now() - last_cached_time).seconds < 60:
                return self.fetch_courses.cache_get(last_cached_time).cache
            else:
                return {'error': 'Failed to fetch from Moodle'}

# Usage example
if __name__ == '__main__':
    client = MoodleClient()
    courses = client.fetch_courses()
    print(courses)
