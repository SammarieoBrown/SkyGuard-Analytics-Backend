#!/usr/bin/env python3
"""Test API performance to verify optimizations."""
import requests
import time

# Test endpoint
url = "http://localhost:8000/api/v1/nowcasting/radar-data/KAMX"

print("Testing API performance with optimizations...")
print("="*50)

# First call (cold start)
print("\nFirst call (cold start):")
start = time.time()
r = requests.get(url, params={"hours_back": 1, "max_frames": 3})
first_time = time.time() - start
print(f"  Status: {r.status_code}")
print(f"  Time: {first_time:.2f}s")
if r.status_code == 200:
    data = r.json()
    print(f"  Frames returned: {len(data.get('frames', []))}")

# Second call (should use cache)
print("\nSecond call (should be faster with cache):")
start = time.time()
r = requests.get(url, params={"hours_back": 1, "max_frames": 3})
second_time = time.time() - start
print(f"  Status: {r.status_code}")
print(f"  Time: {second_time:.2f}s")
if r.status_code == 200:
    data = r.json()
    print(f"  Frames returned: {len(data.get('frames', []))}")

# Performance improvement
if second_time < first_time:
    improvement = ((first_time - second_time) / first_time) * 100
    print(f"\n✅ Performance improved by {improvement:.1f}%")
    print(f"   Saved {first_time - second_time:.2f} seconds")
else:
    print("\n⚠️  No performance improvement detected")

print("\n" + "="*50)
print("Note: Full benefits visible on Render with persistent disk")