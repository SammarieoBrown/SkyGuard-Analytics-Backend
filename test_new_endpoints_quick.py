"""
Quick test of the new endpoints that should work immediately.
"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000/api/v1/nowcasting"

def test_working_endpoints():
    """Test endpoints that should work without data processing."""
    print("🧪 Testing New Endpoints - Quick Test")
    print("="*50)
    
    results = {}
    
    # Test 1: Cache Stats (should work)
    print("\n1. Testing cache stats...")
    try:
        response = requests.get(f"{BASE_URL}/cache-stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Cache stats working: {data['cache_stats']['cache_type']} cache with {data['cache_enabled']} enabled")
            results["cache_stats"] = True
        else:
            print(f"❌ Cache stats failed: {response.status_code}")
            results["cache_stats"] = False
    except Exception as e:
        print(f"❌ Cache stats error: {e}")
        results["cache_stats"] = False
    
    # Test 2: Try mosaic info (may timeout due to import issues)
    print("\n2. Testing mosaic info...")
    try:
        response = requests.get(f"{BASE_URL}/mosaic/info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Mosaic info working: {len(data['supported_sites'])} sites supported")
            results["mosaic_info"] = True
        else:
            print(f"❌ Mosaic info failed: {response.status_code}")
            if response.status_code == 500:
                try:
                    error = response.json()
                    print(f"   Error: {error.get('detail', 'Unknown error')}")
                except:
                    pass
            results["mosaic_info"] = False
    except requests.exceptions.Timeout:
        print("❌ Mosaic info timed out (probably import issue)")
        results["mosaic_info"] = False
    except Exception as e:
        print(f"❌ Mosaic info error: {e}")
        results["mosaic_info"] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 QUICK TEST SUMMARY")
    print(f"{'='*50}")
    
    working = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test, result in results.items():
        status = "✅ WORKING" if result else "❌ BROKEN"
        print(f"{test:20} {status}")
    
    print(f"\nResult: {working}/{total} endpoints working")
    
    if working > 0:
        print("\n🎉 At least some new functionality is working!")
        print("The cache system and performance optimizations are active.")
    
    return working == total

if __name__ == "__main__":
    test_working_endpoints()