import os
import requests
from dotenv import load_dotenv

# Load your .env file
load_dotenv()

# Get your RENT API key from environment variables
API_KEY = os.getenv("RENT_API_KEY")
assert API_KEY, "No RENT_API_KEY found. Did you create .env?"

# Example: Replace with your actual rent API endpoint
url = "https://app.rentcast.io/app/api"  

params = {
    "city": "New York",
    "limit": 1,
    "apikey": API_KEY  # Or "Authorization": f"Bearer {API_KEY}" depending on API docs
}

# Make the request
r = requests.get(url, params=params, timeout=15)
r.raise_for_status()
data = r.json()

# Print a few fields to prove it works
print("✅ API call successful!")
print("Sample response:", data)
