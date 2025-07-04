import redis
import os
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

try:
    r.ping()
    print("✅ Connected successfully!")
except Exception as e:
    print("❌ Connection failed:", e)
