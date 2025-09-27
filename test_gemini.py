#!/usr/bin/env python3
"""
Test Gemini API and available models
"""
import google.generativeai as genai
import os
import sys

# Configure API
GOOGLE_API_KEY = "AIzaSyAplOzyIFzlNkYkCKRhw6O08d1ev6mhaGY"
genai.configure(api_key=GOOGLE_API_KEY)

def test_gemini_models():
    """Test different Gemini models"""
    
    models_to_try = [
        'gemini-pro',
        'gemini-pro-vision', 
        'gemini-1.5-pro',
        'gemini-1.5-flash',
        'models/gemini-pro',
        'models/gemini-pro-vision'
    ]
    
    print("🤖 Testing Gemini Models...")
    print("=" * 50)
    
    for model_name in models_to_try:
        try:
            print(f"Testing: {model_name}")
            model = genai.GenerativeModel(model_name)
            
            # Simple text test
            response = model.generate_content("Say hello")
            print(f"✅ {model_name} - Working!")
            print(f"   Response: {response.text[:50]}...")
            print()
            
        except Exception as e:
            print(f"❌ {model_name} - Error: {str(e)[:100]}")
            print()

def list_available_models():
    """List all available models"""
    try:
        print("📋 Available Models:")
        print("=" * 50)
        
        models = genai.list_models()
        for model in models:
            print(f"• {model.name}")
            if hasattr(model, 'supported_generation_methods'):
                print(f"  Methods: {model.supported_generation_methods}")
            print()
            
    except Exception as e:
        print(f"❌ Error listing models: {e}")

if __name__ == "__main__":
    print("🚀 Gemini API Test Suite")
    print("=" * 50)
    
    try:
        list_available_models()
        print("\n" + "="*50)
        test_gemini_models()
        
    except Exception as e:
        print(f"❌ General error: {e}")
        print("💡 Make sure you have internet connection and valid API key")