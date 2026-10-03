import requests
import json

# Target your local server endpoint
URL = "http://localhost:8000/webhook"

print("--- RUNNING TEST 1: Arabic Handoff Intercept ---")
payload_1 = {
    "phone_number": "+971509998877",
    "message": "أريد التحدث مع مدير لو سمحت"
}
try:
    res1 = requests.post(URL, json=payload_1)
    print("Status Code:", res1.status_code)
    print("Response JSON:", json.dumps(res1.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print("Test 1 Failed:", e)

print("\n--- RUNNING TEST 2: Pinecone Vector RAG Query ---")
payload_2 = {
    "phone_number": "+971505554433",
    "message": "Hi Layla, do you have any properties available on Jumeirah Bay Island?"
}
try:
    res2 = requests.post(URL, json=payload_2)
    print("Status Code:", res2.status_code)
    print("Response JSON:", json.dumps(res2.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print("Test 2 Failed:", e)
