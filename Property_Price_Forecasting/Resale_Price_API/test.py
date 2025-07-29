import requests
import json

# API Gateway
url = 

# JSON
payload = {
    "flat_type": 2,
    "block": 120,
    "storey_range": 4,
    "floor_area": 100.0,
    "lease_commence": 1990,
    "remaining_lease": 900,
    "town": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    "flat_model": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    "year": 2024,
    "month": 3
}

# Send the request
response = requests.post(url, json=payload)


print("Status code:", response.status_code)
print("Response:", response.json())
