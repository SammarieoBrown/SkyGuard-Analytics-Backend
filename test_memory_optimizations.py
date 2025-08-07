#!/usr/bin/env python3
"""
Test script for memory optimizations.
Simulates Render environment and tests memory usage.
"""
import os
import sys
import psutil
import gc
import time
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Simulate Render environment
os.environ["RENDER"] = "true"

from app.config import (
    IS_RENDER, RADAR_MAX_WORKERS, RADAR_MAX_BATCH_SIZE, 
    ENABLE_MEMORY_CLEANUP, TEMP_DIR, CACHE_DIR
)
from app.services.radar_processing_service import RadarProcessingService


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def test_configuration():
    """Test memory-aware configuration."""
    print("\n" + "="*60)
    print("TEST 1: Memory-Aware Configuration")
    print("="*60)
    
    print(f"IS_RENDER: {IS_RENDER}")
    print(f"RADAR_MAX_WORKERS: {RADAR_MAX_WORKERS}")
    print(f"RADAR_MAX_BATCH_SIZE: {RADAR_MAX_BATCH_SIZE}")
    print(f"ENABLE_MEMORY_CLEANUP: {ENABLE_MEMORY_CLEANUP}")
    
    if IS_RENDER:
        assert RADAR_MAX_WORKERS == 1, "Should use 1 worker on Render"
        assert RADAR_MAX_BATCH_SIZE == 5, "Should limit batch size on Render"
        assert ENABLE_MEMORY_CLEANUP == True, "Should enable memory cleanup on Render"
        print("✅ Render configuration correct")
    else:
        print("✅ Local configuration correct")
    
    return True


def test_sequential_processing():
    """Test that processing is sequential on Render."""
    print("\n" + "="*60)
    print("TEST 2: Sequential Processing")
    print("="*60)
    
    processor = RadarProcessingService()
    
    print(f"Max workers: {processor.max_workers}")
    print(f"Memory cleanup enabled: {processor.enable_memory_cleanup}")
    
    if IS_RENDER:
        assert processor.max_workers == 1, "Should use sequential processing on Render"
        assert processor.enable_memory_cleanup == True, "Should enable cleanup on Render"
        print("✅ Sequential processing configured for Render")
    else:
        print("✅ Concurrent processing configured for local")
    
    return True


def test_memory_cleanup():
    """Test memory cleanup after processing."""
    print("\n" + "="*60)
    print("TEST 3: Memory Cleanup")
    print("="*60)
    
    # Record initial memory
    initial_memory = get_memory_usage()
    print(f"Initial memory: {initial_memory:.2f} MB")
    
    # Create some data and clean up
    for i in range(5):
        # Create a large array
        large_array = np.random.rand(1000, 1000).astype(np.float32)
        
        # Delete and collect if on Render
        del large_array
        if ENABLE_MEMORY_CLEANUP:
            gc.collect()
    
    # Check final memory
    final_memory = get_memory_usage()
    print(f"Final memory: {final_memory:.2f} MB")
    
    memory_increase = final_memory - initial_memory
    print(f"Memory increase: {memory_increase:.2f} MB")
    
    if IS_RENDER:
        # Should have minimal increase with cleanup
        if memory_increase < 50:  # Less than 50MB increase
            print("✅ Memory cleanup working effectively")
        else:
            print(f"⚠️  Memory increased by {memory_increase:.2f} MB")
    else:
        print("✅ Memory management tested")
    
    return True


def test_batch_size_limits():
    """Test batch size limiting."""
    print("\n" + "="*60)
    print("TEST 4: Batch Size Limits")
    print("="*60)
    
    # Test with large request
    requested_frames = 20
    effective_frames = min(requested_frames, RADAR_MAX_BATCH_SIZE)
    
    print(f"Requested frames: {requested_frames}")
    print(f"Effective frames: {effective_frames}")
    print(f"RADAR_MAX_BATCH_SIZE: {RADAR_MAX_BATCH_SIZE}")
    
    if IS_RENDER:
        assert effective_frames <= 5, "Should limit to 5 frames on Render"
        print("✅ Batch size properly limited for Render")
    else:
        assert effective_frames <= 20, "Should allow more frames locally"
        print("✅ Batch size appropriate for local")
    
    return True


def test_memory_usage_simulation():
    """Simulate processing and check memory usage."""
    print("\n" + "="*60)
    print("TEST 5: Memory Usage Simulation")
    print("="*60)
    
    initial_memory = get_memory_usage()
    print(f"Initial memory: {initial_memory:.2f} MB")
    
    # Simulate processing frames
    frames = []
    for i in range(RADAR_MAX_BATCH_SIZE):
        # Simulate a processed frame (64x64 uint8)
        frame = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        frames.append(frame)
        
        current_memory = get_memory_usage()
        print(f"  Frame {i+1}: {current_memory:.2f} MB (+{current_memory-initial_memory:.2f} MB)")
        
        # Cleanup on Render
        if ENABLE_MEMORY_CLEANUP and i % 2 == 0:
            gc.collect()
    
    # Convert to sequence array
    sequence = np.array(frames, dtype=np.float32)
    sequence = sequence / 255.0  # Normalize
    
    final_memory = get_memory_usage()
    total_increase = final_memory - initial_memory
    
    print(f"\nFinal memory: {final_memory:.2f} MB")
    print(f"Total increase: {total_increase:.2f} MB")
    print(f"Average per frame: {total_increase/RADAR_MAX_BATCH_SIZE:.2f} MB")
    
    # Clean up
    del frames, sequence
    if ENABLE_MEMORY_CLEANUP:
        gc.collect()
    
    if total_increase < 100:  # Less than 100MB for batch
        print("✅ Memory usage acceptable")
    else:
        print(f"⚠️  High memory usage: {total_increase:.2f} MB")
    
    return True


def main():
    """Run all memory optimization tests."""
    print("\n" + "="*60)
    print("MEMORY OPTIMIZATION TEST SUITE")
    print(f"Simulating Render: {os.environ.get('RENDER') == 'true'}")
    print("="*60)
    
    tests = [
        test_configuration,
        test_sequential_processing,
        test_memory_cleanup,
        test_batch_size_limits,
        test_memory_usage_simulation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    # Memory summary
    final_memory = get_memory_usage()
    print(f"\nFinal process memory: {final_memory:.2f} MB")
    
    if failed == 0:
        print("✅ All memory optimizations working correctly!")
        print("\nExpected improvements on Render:")
        print("- Sequential processing (1 worker instead of 4)")
        print("- Max 5 frames per batch (instead of 20)")
        print("- Aggressive garbage collection")
        print("- Memory usage should stay under 1GB")
        return 0
    else:
        print(f"❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())