#!/usr/bin/env python3
"""
Comprehensive test script for all nowcasting API endpoints.
Tests all 16 endpoints to ensure they work correctly with GCS integration.
"""
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any


BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1/nowcasting"


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_endpoint(method: str, endpoint: str, data: Dict[str, Any] = None, params: Dict[str, Any] = None) -> tuple:
    """Test an API endpoint and return status."""
    url = f"{BASE_URL}{API_PREFIX}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, params=params, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            return False, f"Unsupported method: {method}"
        
        if response.status_code == 200:
            # Try to parse JSON response
            try:
                result = response.json()
                return True, result
            except:
                # For binary responses (images)
                return True, f"Binary response ({len(response.content)} bytes)"
        else:
            try:
                error = response.json()
                return False, f"Status {response.status_code}: {error.get('detail', 'Unknown error')}"
            except:
                return False, f"Status {response.status_code}: {response.text[:200]}"
                
    except requests.exceptions.Timeout:
        return False, "Request timeout (30s)"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def test_all_endpoints():
    """Test all 16 nowcasting endpoints."""
    
    print_header("NOWCASTING API ENDPOINT TESTS")
    print(f"Testing at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base URL: {BASE_URL}{API_PREFIX}")
    
    tests = [
        # 1. Get supported sites
        {
            "name": "Get Supported Sites",
            "method": "GET",
            "endpoint": "/sites",
            "params": None,
            "description": "List all supported radar sites"
        },
        
        # 2. Get nowcasting health
        {
            "name": "Get Nowcasting Health",
            "method": "GET",
            "endpoint": "/health",
            "params": None,
            "description": "Check health of nowcasting service"
        },
        
        # 3. Get data pipeline status
        {
            "name": "Get Data Pipeline Status",
            "method": "GET",
            "endpoint": "/data-status",
            "params": None,
            "description": "Check status of data pipeline"
        },
        
        # 4. Get cache statistics
        {
            "name": "Get Cache Statistics",
            "method": "GET",
            "endpoint": "/cache-stats",
            "params": None,
            "description": "Get Redis cache statistics"
        },
        
        # 5. Get current conditions for KAMX
        {
            "name": "Get Current Radar Conditions (KAMX)",
            "method": "GET",
            "endpoint": "/current-conditions/KAMX",
            "params": None,
            "description": "Get current radar conditions for Miami"
        },
        
        # 6. Get raw radar data
        {
            "name": "Get Raw Radar Data (KAMX)",
            "method": "GET",
            "endpoint": "/radar-data/KAMX",
            "params": {"hours_back": 2, "max_frames": 5},
            "description": "Get raw radar arrays for frontend rendering"
        },
        
        # 7. Get radar timeseries
        {
            "name": "Get Radar Timeseries (KAMX)",
            "method": "GET",
            "endpoint": "/radar-timeseries/KAMX",
            "params": {"hours": 2, "interval_minutes": 30},
            "description": "Get historical radar timeseries data"
        },
        
        # 8. Get single radar frame
        {
            "name": "Get Single Radar Frame (KAMX)",
            "method": "GET",
            "endpoint": "/radar-frame/KAMX",
            "params": {"frame_index": 0},
            "description": "Get a single processed radar frame"
        },
        
        # 9. Get multi-site radar data
        {
            "name": "Get Multi-Site Radar Data",
            "method": "GET",
            "endpoint": "/radar-data/multi-site",
            "params": {"sites": "KAMX,KATX", "hours_back": 1, "max_frames_per_site": 3},
            "description": "Get radar data from multiple sites"
        },
        
        # 10. Get radar visualization
        {
            "name": "Get Radar Visualization (KAMX)",
            "method": "GET",
            "endpoint": "/visualization/KAMX",
            "params": None,
            "description": "Generate NWS-style radar visualization"
        },
        
        # 11. Get radar mosaic
        {
            "name": "Get Radar Mosaic",
            "method": "GET",
            "endpoint": "/mosaic",
            "params": None,
            "description": "Get composite radar mosaic"
        },
        
        # 12. Get mosaic info
        {
            "name": "Get Mosaic Info",
            "method": "GET",
            "endpoint": "/mosaic/info",
            "params": None,
            "description": "Get information about mosaic generation"
        },
        
        # 13. Predict weather nowcast
        {
            "name": "Predict Weather Nowcast",
            "method": "POST",
            "endpoint": "/predict",
            "data": {
                "site_id": "KAMX",
                "hours_back": 1,
                "prediction_steps": 3
            },
            "description": "Generate weather predictions"
        },
        
        # 14. Batch weather nowcast
        {
            "name": "Batch Weather Nowcast",
            "method": "POST",
            "endpoint": "/batch",
            "data": {
                "sites": ["KAMX", "KATX"],
                "hours_back": 1,
                "prediction_steps": 3
            },
            "description": "Generate predictions for multiple sites"
        },
        
        # 15. Refresh radar data
        {
            "name": "Refresh Radar Data",
            "method": "POST",
            "endpoint": "/refresh-data",
            "data": {
                "site_id": "KAMX",
                "hours_back": 1
            },
            "description": "Refresh radar data from source"
        },
        
        # 16. Warm cache for site
        {
            "name": "Warm Cache for Site",
            "method": "POST",
            "endpoint": "/warm-cache",
            "data": {
                "site_id": "KAMX"
            },
            "description": "Warm Redis cache with recent data"
        }
    ]
    
    # Run all tests
    results = []
    passed = 0
    failed = 0
    
    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/16] Testing: {test['name']}")
        print(f"    {test['description']}")
        print(f"    {test['method']} {API_PREFIX}{test['endpoint']}")
        
        # Run test
        start_time = time.time()
        success, result = test_endpoint(
            test['method'],
            test['endpoint'],
            data=test.get('data'),
            params=test.get('params')
        )
        elapsed = time.time() - start_time
        
        # Store result
        results.append({
            "test": test['name'],
            "success": success,
            "time": elapsed,
            "result": result
        })
        
        # Print result
        if success:
            print(f"    ✅ SUCCESS ({elapsed:.2f}s)")
            if isinstance(result, dict):
                # Print key information from response
                if 'status' in result:
                    print(f"       Status: {result['status']}")
                if 'total_files' in result:
                    print(f"       Files: {result['total_files']}")
                if 'frames' in result and isinstance(result['frames'], list):
                    print(f"       Frames: {len(result['frames'])}")
                if 'sites' in result and isinstance(result['sites'], dict):
                    print(f"       Sites: {list(result['sites'].keys())}")
            elif isinstance(result, str):
                print(f"       {result}")
            passed += 1
        else:
            print(f"    ❌ FAILED ({elapsed:.2f}s)")
            print(f"       Error: {result}")
            failed += 1
    
    # Summary
    print_header("TEST SUMMARY")
    print(f"Total Tests: 16")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success Rate: {(passed/16)*100:.1f}%")
    
    # List failed tests
    if failed > 0:
        print("\nFailed Tests:")
        for r in results:
            if not r['success']:
                print(f"  - {r['test']}: {r['result']}")
    
    return passed, failed


def main():
    """Main test runner."""
    print("\n" + "="*60)
    print("  NOWCASTING API COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"✅ Server is running at {BASE_URL}")
    except:
        print(f"❌ Server is not running at {BASE_URL}")
        print("Please start the server with: uvicorn main:app --reload")
        return 1
    
    # Run tests
    passed, failed = test_all_endpoints()
    
    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())