"""
Test script for new nowcasting endpoints added for performance optimization.
Tests the visualization, mosaic, cache warming, and cache stats endpoints.
"""
import requests
import json
import time
from datetime import datetime

# Base URL for the API
BASE_URL = "http://127.0.0.1:8000/api/v1/nowcasting"

def test_endpoint(method, endpoint, data=None, expect_image=False):
    """Test a single endpoint and return results."""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n{'='*60}")
    print(f"Testing: {method} {endpoint}")
    print(f"URL: {url}")
    
    try:
        start_time = time.time()
        
        if method == "GET":
            response = requests.get(url, timeout=120)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=120)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
        
        elapsed_time = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {elapsed_time:.2f} seconds")
        print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        
        if response.status_code == 200:
            if expect_image:
                if 'image/png' in response.headers.get('Content-Type', ''):
                    print(f"✅ SUCCESS - Received PNG image ({len(response.content)} bytes)")
                    return True
                else:
                    print(f"❌ FAILED - Expected image but got: {response.headers.get('Content-Type')}")
                    return False
            else:
                try:
                    data = response.json()
                    print(f"✅ SUCCESS - Response:")
                    print(json.dumps(data, indent=2)[:500] + "..." if len(str(data)) > 500 else json.dumps(data, indent=2))
                    return True
                except:
                    print(f"✅ SUCCESS - Non-JSON response: {response.text[:200]}...")
                    return True
        else:
            print(f"❌ FAILED - Status: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Error text: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ TIMEOUT - Request took longer than 120 seconds")
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ CONNECTION ERROR - Could not connect to server")
        return False
    except Exception as e:
        print(f"❌ ERROR - {str(e)}")
        return False

def main():
    """Test all new endpoints."""
    print("🚀 Testing New Nowcasting Endpoints")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: Cache Stats (should work even with no data)
    results["cache_stats"] = test_endpoint("GET", "/cache-stats")
    
    # Test 2: Mosaic Info (should work without data)
    results["mosaic_info"] = test_endpoint("GET", "/mosaic/info")
    
    # Test 3: Cache Warming for KAMX site
    results["warm_cache_kamx"] = test_endpoint("POST", "/warm-cache?site_id=KAMX&hours_back=6")
    
    # Test 4: Cache Warming for KATX site
    results["warm_cache_katx"] = test_endpoint("POST", "/warm-cache?site_id=KATX&hours_back=6")
    
    # Test 5: Single site visualization for KAMX
    results["visualization_kamx"] = test_endpoint("GET", "/visualization/KAMX", expect_image=True)
    
    # Test 6: Single site visualization for KATX
    results["visualization_katx"] = test_endpoint("GET", "/visualization/KATX", expect_image=True)
    
    # Test 7: Composite radar mosaic
    results["composite_mosaic"] = test_endpoint("GET", "/mosaic", expect_image=True)
    
    # Test 8: Cache stats after operations
    print(f"\n{'='*60}")
    print("Final cache stats after all operations:")
    results["final_cache_stats"] = test_endpoint("GET", "/cache-stats")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All new endpoints are working correctly!")
    else:
        print("⚠️  Some endpoints need attention")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)