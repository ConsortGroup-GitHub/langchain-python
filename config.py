# config.py
import os
from dotenv import load_dotenv

load_dotenv()

UPSTASH_REDIS_URL = os.getenv('UPSTASH_REDIS_URL')
UPSTASH_REST_TOKEN = os.getenv('UPSTASH_REST_TOKEN')