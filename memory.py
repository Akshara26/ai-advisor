import os
import json
from upstash_redis import Redis
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

def get_redis():
    try:
        url = st.secrets["UPSTASH_REDIS_REST_URL"]
        token = st.secrets["UPSTASH_REDIS_REST_TOKEN"]
    except:
        url = os.getenv("UPSTASH_REDIS_REST_URL")
        token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    return Redis(url=url, token=token)

def load_history(session_id: str) -> list:
    """Load conversation history for a session from Redis."""
    try:
        redis = get_redis()
        data = redis.get(f"history:{session_id}")
        if data:
            return json.loads(data)
        return []
    except Exception as e:
        print(f"Redis load error: {e}")
        return []

def _serialize(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def save_history(session_id: str, history: list):
    """Save conversation history to Redis. Expires after 7 days."""
    try:
        redis = get_redis()
        redis.set(f"history:{session_id}", json.dumps(history, default=_serialize), ex=604800)
    except Exception as e:
        print(f"Redis save error: {e}")