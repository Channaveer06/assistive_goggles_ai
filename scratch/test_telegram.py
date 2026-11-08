import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import urllib.parse
import requests
import config

url = config.GUARDIAN_WEBHOOK_URL
print(f"Testing Telegram webhook URL: {url}")
parsed = urllib.parse.urlparse(url)
params = urllib.parse.parse_qs(parsed.query)
chat_id = params.get("chat_id", [None])[0]
print(f"Extracted chat_id: {chat_id}")

payload = {
    "chat_id": chat_id,
    "text": "🧪 [TEST] Assistive Goggles Telegram Bot Connection Test! If you see this, Telegram notifications are working."
}

try:
    resp = requests.post(url, json=payload, timeout=8)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")
except Exception as e:
    print(f"Error sending to Telegram: {e}")
