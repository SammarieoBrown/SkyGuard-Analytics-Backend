#!/usr/bin/env python3
"""
Test all API endpoints to ensure they're working correctly.
Shows complete response output without truncation.
"""
import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_endpoint(method, path, data=None, description=""):
    """Test a single endpoint."""
    url = f"{BASE_URL}{path}"
    print(f"\n{'='*60}")
    print(f"Testing: {description or path}")
    print(f"Method: {method} {url}")
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            print(f"Unsupported method: {method}")
            return False
            
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS")
            # Print full response without truncation
            print("Full Response:")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("Testing SkyGuard Analytics API Endpoints - COMPLETE")
    print("="*60)
    
    success_count = 0
    total_count = 0
    
    # Test data
    tests = [
        # Default endpoints
        ("GET", "/", None, "Root endpoint"),
        ("GET", "/health", None, "Health check"),
        ("GET", "/hello/TestUser", None, "Say hello endpoint"),
        
        # Impact endpoints
        ("POST", "/api/v1/impact/property-damage", {
            "event_type": "Thunderstorm Wind",
            "state": "TX",
            "magnitude": 65.0,
            "duration_hours": 1.5
        }, "Property damage prediction"),
        
        ("POST", "/api/v1/impact/casualty-risk", {
            "event_type": "Tornado",
            "state": "OK",
            "magnitude": 150.0,
            "tor_f_scale": "EF3"
        }, "Casualty risk prediction"),
        
        ("POST", "/api/v1/impact/severity", {
            "event_type": "Tornado",
            "state": "OK",
            "magnitude": 150.0,
            "property_damage": 500000.0,
            "injuries": 15,
            "deaths": 2
        }, "Severity prediction"),
        
        ("POST", "/api/v1/impact/comprehensive-assessment", {
            "event_type": "Hurricane",
            "state": "FL",
            "magnitude": 130.0,
            "duration_hours": 5.0
        }, "Comprehensive assessment"),
        
        ("GET", "/api/v1/impact/healthcheck", None, "Impact healthcheck"),
        
        # Risk endpoints
        ("POST", "/api/v1/risk/state", {
            "state_code": "TX"
        }, "State risk assessment (POST)"),
        
        ("GET", "/api/v1/risk/state/TX", None, "State risk assessment (GET)"),
        
        ("POST", "/api/v1/risk/multi-state", {
            "state_codes": ["TX", "OK", "FL", "KS"]
        }, "Multi-state risk assessment"),
        
        ("POST", "/api/v1/risk/rankings", {
            "limit": 5,
            "ascending": False
        }, "Risk rankings (POST)"),
        
        ("GET", "/api/v1/risk/rankings?limit=3", None, "Risk rankings (GET)"),
        
        ("POST", "/api/v1/risk/event-type", {
            "event_type": "Tornado"
        }, "Risk by event type (POST)"),
        
        ("GET", "/api/v1/risk/event-type/Hurricane", None, "Risk by event type (GET)"),
        
        ("GET", "/api/v1/risk/summary", None, "Risk summary"),
        
        ("GET", "/api/v1/risk/healthcheck", None, "Risk healthcheck"),
        
        # Simulation endpoints
        ("POST", "/api/v1/simulation/scenario", {
            "base_event": {
                "event_type": "Tornado",
                "state": "OK",
                "magnitude": 150.0
            },
            "modifications": [{
                "parameter": "magnitude",
                "modification_type": "multiply",
                "value": 1.5
            }],
            "include_uncertainty": True
        }, "Scenario simulation"),
        
        ("POST", "/api/v1/simulation/batch", {
            "base_event": {
                "event_type": "Tornado",
                "state": "OK",
                "magnitude": 150.0
            },
            "scenario_sets": [
                [{
                    "parameter": "magnitude",
                    "modification_type": "multiply",
                    "value": 1.2
                }],
                [{
                    "parameter": "magnitude",
                    "modification_type": "multiply",
                    "value": 1.5
                }],
                [{
                    "parameter": "magnitude",
                    "modification_type": "multiply",
                    "value": 2.0
                }]
            ],
            "include_uncertainty": False
        }, "Batch simulation"),
        
        ("POST", "/api/v1/simulation/sensitivity", {
            "base_event": {
                "event_type": "Tornado",
                "state": "OK",
                "magnitude": 150.0,
                "duration_hours": 2.0
            },
            "parameters": ["magnitude", "duration_hours"],
            "variation_range": 0.3
        }, "Sensitivity analysis"),
        
        # First run a scenario to get an ID
        ("GET", "/api/v1/simulation/scenario/scenario_test_123", None, "Get saved scenario"),
        
        ("GET", "/api/v1/simulation/healthcheck", None, "Simulation healthcheck"),
    ]
    
    # Run tests
    for method, path, data, description in tests:
        total_count += 1
        if test_endpoint(method, path, data, description):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Test Summary: {success_count}/{total_count} tests passed")
    print(f"{'='*60}\n")
    
    if success_count < total_count:
        print(f"❌ {total_count - success_count} tests failed!")
        sys.exit(1)
    else:
        print("✅ All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()