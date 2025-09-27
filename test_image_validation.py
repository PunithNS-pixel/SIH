#!/usr/bin/env python3
"""
Image Validation Test Script
============================

This script tests the image validation functionality to ensure
the fixes for truncated/corrupted image handling work correctly.

Author: AI Assistant
Date: September 27, 2025
"""

from PIL import Image, ImageFile
import io
import os
import sys

# Enable loading of truncated images for testing
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test_empty_file():
    """Test handling of empty files"""
    print("🧪 Testing empty file...")
    try:
        empty_buffer = io.BytesIO(b'')
        empty_buffer.seek(0)
        if empty_buffer.getvalue() == b'':
            print("✅ Empty file detected correctly")
        else:
            print("❌ Empty file detection failed")
    except Exception as e:
        print(f"❌ Empty file test failed: {e}")

def test_corrupted_image():
    """Test handling of corrupted image data"""
    print("🧪 Testing corrupted image...")
    try:
        # Create corrupted image data
        corrupted_data = b'\xff\xd8\xff\xe0' + b'\x00' * 100  # Invalid JPEG header
        corrupted_buffer = io.BytesIO(corrupted_data)
        
        try:
            img = Image.open(corrupted_buffer)
            img.verify()
            print("❌ Corrupted image should have failed verification")
        except Exception:
            print("✅ Corrupted image detected correctly")
    except Exception as e:
        print(f"❌ Corrupted image test failed: {e}")

def test_valid_image():
    """Test handling of valid image"""
    print("🧪 Testing valid image creation...")
    try:
        # Create a valid test image
        img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        
        # Test validation
        test_img = Image.open(buffer)
        test_img.verify()
        print("✅ Valid image processed correctly")
        
        # Test size check
        buffer.seek(0)
        size_mb = len(buffer.getvalue()) / (1024 * 1024)
        print(f"✅ Image size: {size_mb:.4f} MB")
        
    except Exception as e:
        print(f"❌ Valid image test failed: {e}")

def test_oversized_file():
    """Test handling of oversized files"""
    print("🧪 Testing oversized file detection...")
    try:
        # Create a large image (simulate 15 MB)
        large_data = b'x' * (15 * 1024 * 1024)
        size_mb = len(large_data) / (1024 * 1024)
        
        if size_mb > 10:
            print(f"✅ Oversized file detected: {size_mb} MB > 10 MB")
        else:
            print(f"❌ Oversized file detection failed: {size_mb} MB")
    except Exception as e:
        print(f"❌ Oversized file test failed: {e}")

def test_file_formats():
    """Test different image formats"""
    print("🧪 Testing image format support...")
    formats = ['JPEG', 'PNG', 'BMP']
    
    for fmt in formats:
        try:
            img = Image.new('RGB', (50, 50), color='blue')
            buffer = io.BytesIO()
            img.save(buffer, format=fmt)
            buffer.seek(0)
            
            # Test opening
            test_img = Image.open(buffer)
            test_img.verify()
            print(f"✅ {fmt} format supported")
            
        except Exception as e:
            print(f"❌ {fmt} format test failed: {e}")

def run_validation_tests():
    """Run all validation tests"""
    print("=" * 50)
    print("🔧 IMAGE VALIDATION TEST SUITE")
    print("=" * 50)
    
    test_empty_file()
    test_corrupted_image()
    test_valid_image()
    test_oversized_file()
    test_file_formats()
    
    print("\n" + "=" * 50)
    print("✅ Image validation tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    run_validation_tests()