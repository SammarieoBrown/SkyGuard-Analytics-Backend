#!/usr/bin/env python3
"""
Test script for GCS (Google Cloud Storage) implementation.
Tests storage operations for NEXRAD radar data.
"""
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import GCS_BUCKET_NAME, GCS_CREDENTIALS, USE_GCS_STORAGE
from app.services.gcs_storage_service import GCSStorageService
from app.services.nexrad_data_service import NEXRADDataService


def test_gcs_connection():
    """Test basic GCS connection and bucket access."""
    print("\n" + "="*60)
    print("TEST 1: GCS Connection")
    print("="*60)
    
    if not USE_GCS_STORAGE:
        print("❌ GCS storage is disabled in configuration")
        return False
    
    if not GCS_CREDENTIALS:
        print("❌ GCS credentials not found in environment")
        return False
    
    try:
        # Initialize GCS service
        gcs_service = GCSStorageService(GCS_BUCKET_NAME, GCS_CREDENTIALS)
        print(f"✅ Connected to GCS bucket: {GCS_BUCKET_NAME}")
        
        # Get storage info
        info = gcs_service.get_storage_info()
        print(f"   Total files: {info.get('total_files', 0)}")
        print(f"   Total size: {info.get('total_size_mb', 0):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to GCS: {e}")
        return False


def test_file_operations():
    """Test file upload, download, and delete operations."""
    print("\n" + "="*60)
    print("TEST 2: File Operations")
    print("="*60)
    
    try:
        gcs_service = GCSStorageService(GCS_BUCKET_NAME, GCS_CREDENTIALS)
        
        # Test data
        test_blob_name = "test/nexrad_test_file.bin"
        test_data = b"This is test NEXRAD data"
        test_metadata = {
            "site_id": "TEST",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "test": "true"
        }
        
        # Test upload
        print(f"Testing upload to {test_blob_name}...")
        if gcs_service.upload_file(test_data, test_blob_name, test_metadata):
            print("✅ Upload successful")
        else:
            print("❌ Upload failed")
            return False
        
        # Test file exists
        print("Testing file existence check...")
        if gcs_service.file_exists(test_blob_name):
            print("✅ File exists check passed")
        else:
            print("❌ File exists check failed")
            return False
        
        # Test download
        print("Testing download...")
        downloaded_data = gcs_service.download_file(test_blob_name)
        if downloaded_data == test_data:
            print("✅ Download successful, data matches")
        else:
            print("❌ Download failed or data mismatch")
            return False
        
        # Test list files
        print("Testing list files...")
        files = gcs_service.list_files("test/")
        if any(f['name'] == test_blob_name for f in files):
            print(f"✅ List files successful, found {len(files)} file(s)")
        else:
            print("❌ List files failed")
            return False
        
        # Test delete
        print("Testing delete...")
        if gcs_service.delete_file(test_blob_name):
            print("✅ Delete successful")
        else:
            print("❌ Delete failed")
            return False
        
        # Verify deletion
        if not gcs_service.file_exists(test_blob_name):
            print("✅ Deletion verified")
        else:
            print("❌ File still exists after deletion")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ File operations test failed: {e}")
        return False


def test_nexrad_integration():
    """Test NEXRAD service integration with GCS."""
    print("\n" + "="*60)
    print("TEST 3: NEXRAD Service Integration")
    print("="*60)
    
    try:
        # Initialize NEXRAD service
        nexrad_service = NEXRADDataService()
        
        # Check if GCS is being used
        if nexrad_service.gcs_service:
            print("✅ NEXRAD service initialized with GCS storage")
            
            # Test downloading a small amount of recent data
            print("\nTesting NEXRAD data download to GCS...")
            site_id = "KAMX"
            
            # Download just 1 hour of data for testing
            results = nexrad_service.download_recent_data(site_id, hours_back=1)
            
            print(f"Download results for {site_id}:")
            print(f"   Hours successful: {results['hours_successful']}")
            print(f"   Total files: {results['total_files']}")
            print(f"   Duration: {results.get('duration_seconds', 0):.2f} seconds")
            
            if results['total_files'] > 0:
                print("✅ NEXRAD data successfully downloaded to GCS")
                
                # Check if files are in GCS
                gcs_service = GCSStorageService(GCS_BUCKET_NAME, GCS_CREDENTIALS)
                prefix = f"nexrad/{site_id}/"
                files = gcs_service.list_files(prefix, max_results=5)
                
                if files:
                    print(f"✅ Found {len(files)} file(s) in GCS under {prefix}")
                    for f in files[:3]:  # Show first 3 files
                        print(f"   - {f['name']} ({f['size']} bytes)")
                else:
                    print(f"⚠️  No files found in GCS under {prefix}")
                
                return True
            else:
                print("⚠️  No files were downloaded")
                return False
        else:
            print("⚠️  NEXRAD service is using local storage, not GCS")
            return False
            
    except Exception as e:
        print(f"❌ NEXRAD integration test failed: {e}")
        return False


def main():
    """Run all GCS implementation tests."""
    print("\n" + "="*60)
    print("GCS Implementation Test Suite")
    print("="*60)
    print(f"Bucket: {GCS_BUCKET_NAME}")
    print(f"GCS Enabled: {USE_GCS_STORAGE}")
    print(f"Credentials Present: {'Yes' if GCS_CREDENTIALS else 'No'}")
    
    # Run tests
    tests_passed = 0
    tests_total = 3
    
    if test_gcs_connection():
        tests_passed += 1
    
    if test_file_operations():
        tests_passed += 1
    
    if test_nexrad_integration():
        tests_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("✅ All tests passed! GCS implementation is working correctly.")
        return 0
    else:
        print(f"❌ {tests_total - tests_passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit(main())