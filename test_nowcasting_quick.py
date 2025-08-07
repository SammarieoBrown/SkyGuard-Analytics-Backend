#!/usr/bin/env python3
"""Quick test of all nowcasting endpoints."""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1/nowcasting"

endpoints = [
    ("GET", "/sites", None, None),
    ("GET", "/health", None, None),
    ("GET", "/data-status", None, None),
    ("GET", "/cache-stats", None, None),
    ("GET", "/current-conditions/KAMX", None, None),
    ("GET", "/radar-data/KAMX", {"hours_back": 1, "max_frames": 2}, None),
    ("GET", "/radar-timeseries/KAMX", {"hours": 1}, None),
    ("GET", "/radar-frame/KAMX", None, None),
    ("GET", "/radar-data/multi-site", {"sites": "KAMX,KATX", "hours_back": 1, "max_frames_per_site": 2}, None),
    ("GET", "/visualization/KAMX", None, None),
    ("GET", "/mosaic", None, None),
    ("GET", "/mosaic/info", None, None),
    ("POST", "/predict", None, {"site_id": "KAMX", "hours_back": 1}),
    ("POST", "/batch", None, {"sites": ["KAMX"], "hours_back": 1}),
    ("POST", "/refresh-data", None, {"site_id": "KAMX", "hours_back": 1}),
    ("POST", "/warm-cache", None, {"site_id": "KAMX"}),
]

print("Testing Nowcasting Endpoints:")
print("="*50)

passed = 0
failed = 0

for method, endpoint, params, data in endpoints:
    try:
        if method == "GET":
            r = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=10)
        else:
            r = requests.post(f"{BASE_URL}{endpoint}", json=data, timeout=10)
        
        if r.status_code == 200:
            print(f"✅ {method:4} {endpoint:30} - OK")
            passed += 1
        else:
            print(f"❌ {method:4} {endpoint:30} - {r.status_code}")
            if r.headers.get('content-type') == 'application/json':
                error = r.json()
                print(f"     Error: {error.get('detail', 'Unknown')}")
            failed += 1
    except Exception as e:
        print(f"❌ {method:4} {endpoint:30} - Exception: {str(e)[:50]}")
        failed += 1

print("="*50)
print(f"Results: {passed} passed, {failed} failed out of {len(endpoints)} total")