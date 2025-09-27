#!/usr/bin/env python3
"""
Google Gemini Vision API - Comprehensive Image Analysis Demo
============================================================

This script demonstrates how to use Google Gemini AI to perform comprehensive 
image analysis on uploaded images. It provides detailed insights about the 
image content, objects, environment, and technical characteristics.

Author: AI Assistant
Date: September 27, 2025
"""

import google.generativeai as genai
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import datetime
import json

# Configuration
GOOGLE_API_KEY = "AIzaSyAplOzyIFzlNkYkCKRhw6O08d1ev6mhaGY"

def enhance_image_for_analysis(image):
    """
    Enhance image quality for better AI analysis
    """
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (max 2048x2048 for Gemini)
        max_size = 2048
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Enhance brightness and contrast
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Sharpen the image
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        return image
        
    except Exception as e:
        print(f"Image enhancement error: {e}")
        return image

def comprehensive_image_analysis(image_path):
    """
    Perform comprehensive image analysis using Google Gemini Vision API
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Analysis results with success status and detailed information
    """
    try:
        # Configure Gemini API
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        # Load and enhance image
        image = Image.open(image_path)
        enhanced_image = enhance_image_for_analysis(image)
        
        # Comprehensive analysis prompt
        analysis_prompt = """🔬 COMPREHENSIVE IMAGE ANALYSIS EXPERT 🔬

You are an expert image analyst with advanced computer vision capabilities. Analyze this image thoroughly and provide detailed insights.

**ANALYSIS FRAMEWORK:**

🎯 **PRIMARY IDENTIFICATION**
- Main subject/object in the image
- Species/category classification (if animal/plant)
- Confidence level (1-10)

👁️ **VISUAL CHARACTERISTICS**
- Physical appearance (size, shape, color, texture)
- Distinctive features and markings
- Body structure and proportions
- Facial features and expressions (if applicable)

🌍 **ENVIRONMENTAL CONTEXT**
- Setting and background details
- Lighting conditions and quality
- Weather/atmospheric conditions
- Surrounding objects or elements

🎬 **BEHAVIORAL ANALYSIS** (if applicable)
- Subject's pose and position
- Apparent activity or behavior
- Body language and mood
- Interaction with environment

📸 **TECHNICAL ASSESSMENT**
- Image quality and resolution
- Composition and framing
- Clarity and focus
- Any technical limitations

🔍 **DETAILED OBSERVATIONS**
- Unique characteristics
- Noteworthy details
- Potential concerns or interests
- Additional context clues

**STRUCTURED OUTPUT:**
PRIMARY_SUBJECT: [Main subject identification]
CONFIDENCE_SCORE: [1-10 rating]
PHYSICAL_FEATURES: [Detailed physical description]
ENVIRONMENT: [Setting and context]
BEHAVIOR: [Actions or pose observed]
TECHNICAL_QUALITY: [Image quality assessment]
KEY_INSIGHTS: [Important observations]
RECOMMENDATIONS: [Suggestions for better analysis if applicable]

Provide comprehensive, professional analysis covering all aspects above.
"""
        
        # Generate analysis
        print("🔍 Analyzing image with Google Gemini AI...")
        response = model.generate_content([analysis_prompt, enhanced_image])
        
        if response and response.text:
            return {
                'success': True,
                'analysis': response.text,
                'model': 'Google Gemini 2.5 Flash',
                'timestamp': datetime.datetime.now().isoformat(),
                'image_path': image_path
            }
        else:
            return {
                'success': False,
                'error': 'No response from Gemini API'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def parse_structured_analysis(analysis_text):
    """
    Parse structured analysis response into organized components
    
    Args:
        analysis_text (str): Raw analysis text from Gemini
        
    Returns:
        dict: Parsed analysis components
    """
    parsed = {
        'primary_subject': 'Not specified',
        'confidence_score': 'Not specified',
        'physical_features': 'Not specified',
        'environment': 'Not specified',
        'behavior': 'Not specified',
        'technical_quality': 'Not specified',
        'key_insights': 'Not specified',
        'recommendations': 'Not specified'
    }
    
    lines = analysis_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            
            if key in parsed:
                parsed[key] = value
    
    return parsed

def display_analysis_results(results):
    """
    Display analysis results in a formatted way
    
    Args:
        results (dict): Analysis results from comprehensive_image_analysis
    """
    if not results['success']:
        print(f"❌ Analysis failed: {results['error']}")
        return
    
    print("\n" + "="*60)
    print("🤖 GOOGLE GEMINI VISION ANALYSIS RESULTS")
    print("="*60)
    
    print(f"📅 Timestamp: {results['timestamp']}")
    print(f"🔧 Model: {results['model']}")
    print(f"📁 Image: {results['image_path']}")
    
    print("\n" + "-"*60)
    print("📋 COMPLETE ANALYSIS:")
    print("-"*60)
    print(results['analysis'])
    
    # Try to parse structured components
    parsed = parse_structured_analysis(results['analysis'])
    
    print("\n" + "-"*60)
    print("📊 PARSED COMPONENTS:")
    print("-"*60)
    
    for key, value in parsed.items():
        if value != 'Not specified':
            print(f"🔹 {key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "="*60)

def main():
    """
    Main function to demonstrate comprehensive image analysis
    """
    print("🔬 Google Gemini Vision API - Comprehensive Image Analysis Demo")
    print("=" * 65)
    
    # Example usage
    image_path = input("📁 Enter image path (or press Enter for demo): ").strip()
    
    if not image_path:
        print("ℹ️  No path provided. Please provide a valid image path.")
        return
    
    try:
        # Perform analysis
        results = comprehensive_image_analysis(image_path)
        
        # Display results
        display_analysis_results(results)
        
        # Save results to JSON file
        output_file = f"analysis_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()