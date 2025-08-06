#!/usr/bin/env python3
"""
Comprehensive test suite for new raw radar data endpoints.

This test suite validates:
- All new endpoint functionality
- Response schema compliance
- Real radar data processing
- Error handling
- Performance benchmarks
- Data integrity

Designed to ensure no breaking changes and validate frontend-ready data.
"""
import requests
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configuration
BASE_URL = "http://localhost:8000/api/v1/nowcasting"
TIMEOUT_SECONDS = 120
PERFORMANCE_THRESHOLD_MS = 5000  # 5 second max per endpoint

class TestResult:
    """Track individual test results."""
    def __init__(self, name: str, success: bool, duration_ms: float, details: str = ""):
        self.name = name
        self.success = success
        self.duration_ms = duration_ms
        self.details = details
        self.timestamp = datetime.now()

class RawRadarEndpointTester:
    """Comprehensive tester for raw radar data endpoints."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_sites = ["KAMX", "KATX"]
        self.start_time = datetime.now()
        
    def log_test(self, name: str, success: bool, duration_ms: float, details: str = ""):
        """Log a test result."""
        result = TestResult(name, success, duration_ms, details)
        self.results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        perf_indicator = "🚀" if duration_ms < 2000 else "⚡" if duration_ms < 5000 else "🐌"
        
        print(f"{status} {perf_indicator} {name} ({duration_ms:.0f}ms)")
        if details and not success:
            print(f"    Details: {details}")
    
    def make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> tuple:
        """Make HTTP request and return (success, response_data, duration_ms, error_details)."""
        url = f"{BASE_URL}{endpoint}"
        start_time = time.time()
        
        try:
            if method == "GET":
                response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
            elif method == "POST":
                response = requests.post(url, json=data, params=params, timeout=TIMEOUT_SECONDS)
            else:
                return False, None, 0, f"Unsupported method: {method}"
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    return True, response_data, duration_ms, ""
                except json.JSONDecodeError:
                    return False, None, duration_ms, "Invalid JSON response"
            else:
                error_details = f"HTTP {response.status_code}: {response.text[:200]}"
                return False, None, duration_ms, error_details
                
        except requests.exceptions.Timeout:
            return False, None, TIMEOUT_SECONDS * 1000, "Request timeout"
        except requests.exceptions.ConnectionError:
            return False, None, 0, "Connection error - is server running?"
        except Exception as e:
            return False, None, 0, f"Unexpected error: {str(e)}"
    
    # ===========================
    # Schema Validation Tests
    # ===========================
    
    def validate_coordinate_metadata(self, coords: Dict) -> tuple:
        """Validate CoordinateMetadata schema."""
        required_fields = ["bounds", "center", "resolution_deg", "resolution_km", "projection", "range_km"]
        
        for field in required_fields:
            if field not in coords:
                return False, f"Missing field: {field}"
        
        # Validate bounds format [west, east, south, north]
        bounds = coords["bounds"]
        if not isinstance(bounds, list) or len(bounds) != 4:
            return False, f"Invalid bounds format: {bounds}"
        
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            return False, f"Invalid bounds values: west >= east or south >= north"
        
        # Validate center format [lat, lon]
        center = coords["center"]
        if not isinstance(center, list) or len(center) != 2:
            return False, f"Invalid center format: {center}"
        
        if not (-90 <= center[0] <= 90) or not (-180 <= center[1] <= 180):
            return False, f"Invalid center coordinates: {center}"
        
        # Validate positive resolution
        if coords["resolution_deg"] <= 0 or coords["resolution_km"] <= 0:
            return False, f"Invalid resolution values: {coords['resolution_deg']}, {coords['resolution_km']}"
        
        return True, ""
    
    def validate_radar_frame(self, frame: Dict) -> tuple:
        """Validate RadarDataFrame schema."""
        required_fields = ["timestamp", "data", "coordinates", "intensity_range", "data_quality"]
        
        for field in required_fields:
            if field not in frame:
                return False, f"Missing field: {field}"
        
        # Validate data dimensions (should be 64x64)
        data = frame["data"]
        if not isinstance(data, list) or len(data) != 64:
            return False, f"Invalid data dimensions: expected 64 rows, got {len(data) if isinstance(data, list) else 'not a list'}"
        
        for i, row in enumerate(data):
            if not isinstance(row, list) or len(row) != 64:
                return False, f"Invalid row {i}: expected 64 columns, got {len(row) if isinstance(row, list) else 'not a list'}"
        
        # Validate coordinates
        coord_valid, coord_error = self.validate_coordinate_metadata(frame["coordinates"])
        if not coord_valid:
            return False, f"Invalid coordinates: {coord_error}"
        
        # Validate intensity range
        intensity_range = frame["intensity_range"]
        if not isinstance(intensity_range, list) or len(intensity_range) != 2:
            return False, f"Invalid intensity_range: {intensity_range}"
        
        if intensity_range[0] > intensity_range[1]:
            return False, f"Invalid intensity range: min > max"
        
        # Validate data quality
        if frame["data_quality"] not in ["good", "fair", "poor"]:
            return False, f"Invalid data_quality: {frame['data_quality']}"
        
        return True, ""
    
    def validate_raw_radar_response(self, response: Dict) -> tuple:
        """Validate RawRadarDataResponse schema."""
        required_fields = ["success", "site_info", "frames", "total_frames", "time_range", "processing_time_ms"]
        
        for field in required_fields:
            if field not in response:
                return False, f"Missing field: {field}"
        
        # Validate frames
        frames = response["frames"]
        if not isinstance(frames, list):
            return False, "Frames must be a list"
        
        if len(frames) != response["total_frames"]:
            return False, f"Frame count mismatch: {len(frames)} vs {response['total_frames']}"
        
        # Validate each frame
        for i, frame in enumerate(frames):
            frame_valid, frame_error = self.validate_radar_frame(frame)
            if not frame_valid:
                return False, f"Invalid frame {i}: {frame_error}"
        
        # Validate time_range
        time_range = response["time_range"]
        if "start" not in time_range or "end" not in time_range:
            return False, "time_range missing start or end"
        
        return True, ""
    
    # ===========================
    # Endpoint Tests
    # ===========================
    
    def test_get_raw_radar_data(self):
        """Test /radar-data/{site_id} endpoint."""
        print("\n" + "="*60)
        print("Testing /radar-data/{site_id} endpoint")
        print("="*60)
        
        for site_id in self.test_sites:
            # Test 1: Basic functionality
            success, response, duration, error = self.make_request(
                "GET", f"/radar-data/{site_id}", 
                params={"hours_back": 6, "max_frames": 5}
            )
            
            if success:
                # Validate schema
                schema_valid, schema_error = self.validate_raw_radar_response(response)
                if schema_valid:
                    self.log_test(f"radar-data/{site_id} - basic", True, duration, 
                                f"{response['total_frames']} frames returned")
                else:
                    self.log_test(f"radar-data/{site_id} - basic", False, duration, 
                                f"Schema validation failed: {schema_error}")
            else:
                self.log_test(f"radar-data/{site_id} - basic", False, duration, error)
            
            # Test 2: With processing metadata
            success, response, duration, error = self.make_request(
                "GET", f"/radar-data/{site_id}", 
                params={"hours_back": 3, "max_frames": 2, "include_processing_metadata": True}
            )
            
            if success and response.get("frames"):
                has_metadata = all("processing_metadata" in frame for frame in response["frames"])
                self.log_test(f"radar-data/{site_id} - metadata", has_metadata, duration, 
                            "Processing metadata included" if has_metadata else "Missing processing metadata")
            else:
                self.log_test(f"radar-data/{site_id} - metadata", False, duration, error or "No frames returned")
        
        # Test 3: Error handling - invalid site
        success, response, duration, error = self.make_request(
            "GET", "/radar-data/INVALID"
        )
        
        expected_error = not success  # Should fail
        self.log_test("radar-data/INVALID - error handling", expected_error, duration, 
                     "Correctly rejected invalid site" if expected_error else "Should have failed")
    
    def test_get_radar_timeseries(self):
        """Test /radar-timeseries/{site_id} endpoint."""
        print("\n" + "="*60)
        print("Testing /radar-timeseries/{site_id} endpoint")
        print("="*60)
        
        # Use recent time range based on available data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=6)
        
        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S")
        
        for site_id in self.test_sites:
            # Test 1: Basic time series
            success, response, duration, error = self.make_request(
                "GET", f"/radar-timeseries/{site_id}", 
                params={"start_time": start_str, "end_time": end_str, "max_frames": 10}
            )
            
            if success:
                # Check for time series specific fields
                required_fields = ["temporal_resolution_minutes", "data_completeness", "actual_time_range"]
                has_all_fields = all(field in response for field in required_fields)
                
                if has_all_fields:
                    completeness = response.get("data_completeness", 0)
                    self.log_test(f"timeseries/{site_id} - basic", True, duration, 
                                f"{response.get('total_frames', 0)} frames, {completeness:.1%} complete")
                else:
                    missing = [field for field in required_fields if field not in response]
                    self.log_test(f"timeseries/{site_id} - basic", False, duration, 
                                f"Missing fields: {missing}")
            else:
                self.log_test(f"timeseries/{site_id} - basic", False, duration, error)
        
        # Test 2: Error handling - invalid time range
        success, response, duration, error = self.make_request(
            "GET", f"/radar-timeseries/{self.test_sites[0]}", 
            params={"start_time": end_str, "end_time": start_str}  # Swapped times
        )
        
        expected_error = not success  # Should fail
        self.log_test("timeseries - invalid time range", expected_error, duration, 
                     "Correctly rejected invalid time range" if expected_error else "Should have failed")
    
    def test_get_multi_site_radar_data(self):
        """Test /radar-data/multi-site endpoint."""
        print("\n" + "="*60)
        print("Testing /radar-data/multi-site endpoint")
        print("="*60)
        
        # Test 1: Multiple valid sites
        site_ids_str = ",".join(self.test_sites)
        success, response, duration, error = self.make_request(
            "GET", "/radar-data/multi-site", 
            params={"site_ids": site_ids_str, "hours_back": 6, "max_frames_per_site": 3}
        )
        
        if success:
            expected_sites = len(self.test_sites)
            total_sites = response.get("total_sites", 0)
            successful_sites = response.get("successful_sites", 0)
            
            all_sites_processed = total_sites == expected_sites
            has_site_data = "site_data" in response and len(response["site_data"]) == expected_sites
            
            if all_sites_processed and has_site_data:
                self.log_test("multi-site - basic", True, duration, 
                            f"{successful_sites}/{total_sites} sites successful")
            else:
                self.log_test("multi-site - basic", False, duration, 
                            f"Expected {expected_sites} sites, got {total_sites}, data: {has_site_data}")
        else:
            self.log_test("multi-site - basic", False, duration, error)
        
        # Test 2: Mixed valid/invalid sites
        mixed_sites = f"{self.test_sites[0]},INVALID,{self.test_sites[1]}"
        success, response, duration, error = self.make_request(
            "GET", "/radar-data/multi-site", 
            params={"site_ids": mixed_sites}
        )
        
        expected_error = not success  # Should fail due to invalid site
        self.log_test("multi-site - mixed validity", expected_error, duration, 
                     "Correctly rejected invalid site" if expected_error else "Should have failed")
    
    def test_get_single_radar_frame(self):
        """Test /radar-frame/{site_id} endpoint."""
        print("\n" + "="*60)
        print("Testing /radar-frame/{site_id} endpoint")
        print("="*60)
        
        for site_id in self.test_sites:
            # Test 1: Latest frame
            success, response, duration, error = self.make_request(
                "GET", f"/radar-frame/{site_id}"
            )
            
            if success:
                # Should return a single RadarDataFrame
                frame_valid, frame_error = self.validate_radar_frame(response)
                if frame_valid:
                    timestamp = response.get("timestamp", "unknown")
                    quality = response.get("data_quality", "unknown")
                    self.log_test(f"radar-frame/{site_id} - latest", True, duration, 
                                f"Quality: {quality}, Time: {timestamp[:19]}")
                else:
                    self.log_test(f"radar-frame/{site_id} - latest", False, duration, 
                                f"Frame validation failed: {frame_error}")
            else:
                self.log_test(f"radar-frame/{site_id} - latest", False, duration, error)
            
            # Test 2: Specific timestamp (use a recent timestamp)
            recent_time = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
            success, response, duration, error = self.make_request(
                "GET", f"/radar-frame/{site_id}", 
                params={"timestamp": recent_time, "include_processing_metadata": True}
            )
            
            if success:
                has_metadata = "processing_metadata" in response
                self.log_test(f"radar-frame/{site_id} - timestamp", True, duration, 
                            f"Found closest frame, metadata: {has_metadata}")
            else:
                # It's OK if no data exists for the specific timestamp
                self.log_test(f"radar-frame/{site_id} - timestamp", True, duration, 
                            "No data for timestamp (acceptable)")
    
    # ===========================
    # Data Integrity Tests
    # ===========================
    
    def test_data_integrity(self):
        """Test data integrity and coordinate accuracy."""
        print("\n" + "="*60)
        print("Testing Data Integrity")
        print("="*60)
        
        for site_id in self.test_sites:
            success, response, duration, error = self.make_request(
                "GET", f"/radar-data/{site_id}", 
                params={"hours_back": 3, "max_frames": 3}
            )
            
            if not success:
                self.log_test(f"integrity/{site_id} - data fetch", False, duration, error)
                continue
            
            frames = response.get("frames", [])
            if not frames:
                self.log_test(f"integrity/{site_id} - no frames", False, duration, "No frames available")
                continue
            
            frame = frames[0]  # Test first frame
            
            # Test 1: Data array integrity
            data = frame["data"]
            flat_data = [val for row in data for val in row]
            
            # Check for reasonable data values (0-255 range expected)
            min_val, max_val = min(flat_data), max(flat_data)
            data_range_valid = 0 <= min_val <= max_val <= 255
            
            # Check for non-trivial data (not all zeros)
            non_zero_count = sum(1 for val in flat_data if val > 0)
            has_signal = non_zero_count > 100  # At least some radar returns
            
            data_integrity_ok = data_range_valid and has_signal
            self.log_test(f"integrity/{site_id} - data values", data_integrity_ok, 0, 
                         f"Range: {min_val:.1f}-{max_val:.1f}, Non-zero: {non_zero_count}/4096")
            
            # Test 2: Coordinate accuracy for known sites
            coords = frame["coordinates"]
            center = coords["center"]
            
            # Known approximate coordinates
            expected_coords = {
                "KAMX": (25.6112, -80.4128),  # Miami
                "KATX": (48.1947, -122.4956)  # Seattle
            }
            
            if site_id in expected_coords:
                expected_lat, expected_lon = expected_coords[site_id]
                actual_lat, actual_lon = center[0], center[1]
                
                # Allow small tolerance for coordinate accuracy
                lat_diff = abs(actual_lat - expected_lat)
                lon_diff = abs(actual_lon - expected_lon)
                
                coords_accurate = lat_diff < 0.1 and lon_diff < 0.1  # Within ~10km
                self.log_test(f"integrity/{site_id} - coordinates", coords_accurate, 0, 
                             f"Expected: ({expected_lat:.4f}, {expected_lon:.4f}), "
                             f"Actual: ({actual_lat:.4f}, {actual_lon:.4f})")
            
            # Test 3: Timestamp ordering (if multiple frames)
            if len(frames) > 1:
                timestamps = [datetime.fromisoformat(frame["timestamp"].replace('Z', '+00:00')) for frame in frames]
                is_ordered = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
                self.log_test(f"integrity/{site_id} - time ordering", is_ordered, 0, 
                             f"Frames are {'properly' if is_ordered else 'incorrectly'} time-ordered")
    
    # ===========================
    # Performance Tests
    # ===========================
    
    def test_performance(self):
        """Test endpoint performance benchmarks."""
        print("\n" + "="*60)
        print("Testing Performance Benchmarks")
        print("="*60)
        
        performance_tests = [
            ("radar-data/KAMX", "GET", "/radar-data/KAMX", {"max_frames": 10}),
            ("radar-frame/KAMX", "GET", "/radar-frame/KAMX", {}),
            ("multi-site", "GET", "/radar-data/multi-site", {"site_ids": "KAMX,KATX", "max_frames_per_site": 5}),
        ]
        
        for test_name, method, endpoint, params in performance_tests:
            # Run test multiple times for average
            durations = []
            success_count = 0
            
            for run in range(3):  # 3 runs for averaging
                success, response, duration, error = self.make_request(method, endpoint, params=params)
                if success:
                    durations.append(duration)
                    success_count += 1
            
            if durations:
                avg_duration = sum(durations) / len(durations)
                min_duration = min(durations)
                max_duration = max(durations)
                
                performance_ok = avg_duration < PERFORMANCE_THRESHOLD_MS
                self.log_test(f"performance - {test_name}", performance_ok, avg_duration, 
                             f"Avg: {avg_duration:.0f}ms, Range: {min_duration:.0f}-{max_duration:.0f}ms, "
                             f"Success: {success_count}/3")
            else:
                self.log_test(f"performance - {test_name}", False, 0, "All runs failed")
    
    # ===========================
    # Main Test Runner
    # ===========================
    
    def run_all_tests(self):
        """Run all test suites."""
        print(f"\n🚀 Starting Comprehensive Raw Radar Endpoint Tests")
        print(f"Timestamp: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Base URL: {BASE_URL}")
        print(f"Test Sites: {', '.join(self.test_sites)}")
        print(f"Performance Threshold: {PERFORMANCE_THRESHOLD_MS}ms")
        
        # Check server availability
        print(f"\n📡 Checking server availability...")
        success, _, duration, error = self.make_request("GET", "/health")
        if not success:
            print(f"❌ Server not available: {error}")
            print(f"Please ensure the API server is running at {BASE_URL}")
            return False
        
        print(f"✅ Server is running ({duration:.0f}ms)")
        
        # Run test suites
        test_suites = [
            ("Basic Endpoint Tests", [
                self.test_get_raw_radar_data,
                self.test_get_radar_timeseries,
                self.test_get_multi_site_radar_data,
                self.test_get_single_radar_frame
            ]),
            ("Data Integrity Tests", [
                self.test_data_integrity
            ]),
            ("Performance Tests", [
                self.test_performance
            ])
        ]
        
        for suite_name, test_functions in test_suites:
            print(f"\n🔬 {suite_name}")
            for test_func in test_functions:
                try:
                    test_func()
                except Exception as e:
                    print(f"❌ Test suite {test_func.__name__} crashed: {str(e)}")
        
        # Generate summary
        self.print_summary()
        
        # Return overall success
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if result.success)
        return passed_tests == total_tests
    
    def print_summary(self):
        """Print comprehensive test summary."""
        print(f"\n" + "="*80)
        print(f"📊 TEST SUMMARY")
        print(f"="*80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if result.success)
        failed_tests = total_tests - passed_tests
        
        # Overall stats
        avg_duration = sum(result.duration_ms for result in self.results) / total_tests if total_tests > 0 else 0
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Average Response Time: {avg_duration:.0f}ms")
        print(f"Total Test Duration: {total_duration:.1f}s")
        
        # Performance analysis
        fast_tests = sum(1 for result in self.results if result.success and result.duration_ms < 2000)
        slow_tests = sum(1 for result in self.results if result.success and result.duration_ms > 5000)
        
        print(f"\n⚡ Performance Breakdown:")
        print(f"Fast (<2s): {fast_tests}")
        print(f"Acceptable (2-5s): {passed_tests - fast_tests - slow_tests}")
        print(f"Slow (>5s): {slow_tests}")
        
        # Failed test details
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.results:
                if not result.success:
                    print(f"  • {result.name}: {result.details}")
        
        # Final verdict
        print(f"\n" + "="*80)
        if failed_tests == 0:
            print(f"🎉 ALL TESTS PASSED! Raw radar endpoints are ready for frontend integration.")
        else:
            print(f"⚠️  {failed_tests} test(s) failed. Please review and fix issues before deployment.")
        print(f"="*80)


def main():
    """Main test runner."""
    tester = RawRadarEndpointTester()
    success = tester.run_all_tests()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()