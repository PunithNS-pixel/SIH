#!/usr/bin/env python3
"""
Deployment Verification Script for KRISHI AI
Checks if all dependencies are properly installed and compatible
"""

import sys
import importlib
from datetime import datetime

def check_package(package_name, import_name=None):
    """Check if a package can be imported and get its version"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Version not available')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: Import failed - {e}")
        return False

def main():
    print("🌾 KRISHI AI - Deployment Verification")
    print("=" * 50)
    print(f"Python Version: {sys.version}")
    print(f"Check Time: {datetime.now()}")
    print("-" * 50)
    
    # List of required packages
    packages = [
        ('streamlit', 'streamlit'),
        ('pandas', 'pandas'), 
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('plotly', 'plotly'),
        ('Pillow', 'PIL'),
        ('requests', 'requests'),
        ('google-generativeai', 'google.generativeai')
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package_name, import_name in packages:
        if check_package(package_name, import_name):
            success_count += 1
    
    print("-" * 50)
    print(f"Summary: {success_count}/{total_count} packages successfully imported")
    
    if success_count == total_count:
        print("🎉 All dependencies are properly installed!")
        print("🚀 Ready for deployment!")
        return 0
    else:
        print("⚠️  Some dependencies are missing or incompatible")
        print("📋 Please check the requirements.txt file")
        return 1

if __name__ == "__main__":
    exit(main())