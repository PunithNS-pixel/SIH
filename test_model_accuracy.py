#!/usr/bin/env python3
"""
Test script to validate animal classification model accuracy
"""

import os
import sys
import numpy as np
from PIL import Image
import random

# Add the current directory to the Python path
sys.path.append('/Users/punithns/Desktop/SIH')

def test_model_accuracy(num_samples=50):
    """Test model accuracy with random samples from the dataset"""
    
    # Import the classification function
    from streamlit_app import classify_animal_simple, get_class_labels
    
    train_dir = '/Users/punithns/Desktop/SIH/Farm Harmful Animal Dataset/train'
    
    if not os.path.exists(train_dir):
        print("❌ Dataset not found!")
        return
    
    print(f"🧪 Testing model accuracy with {num_samples} random samples...")
    print("=" * 60)
    
    # Get all categories
    categories = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    class_labels = get_class_labels()
    
    print(f"📊 Dataset categories: {len(categories)}")
    print(f"🏷️ Model classes: {len(class_labels)}")
    print()
    
    correct_predictions = 0
    total_predictions = 0
    results = []
    
    for i in range(num_samples):
        try:
            # Pick random category and image
            category = random.choice(categories)
            category_path = os.path.join(train_dir, category)
            
            images = [f for f in os.listdir(category_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not images:
                continue
                
            image_file = random.choice(images)
            image_path = os.path.join(category_path, image_file)
            
            # Test prediction
            with open(image_path, 'rb') as f:
                predicted_animal, confidence = classify_animal_simple(f)
            
            # Normalize names for comparison
            true_label = category.lower().replace(' ', '_').replace('-', '_')
            pred_label = predicted_animal.lower().replace(' ', '_').replace('-', '_')
            
            is_correct = (true_label == pred_label or 
                         true_label.replace('_', '') == pred_label.replace('_', '') or
                         true_label in pred_label or pred_label in true_label)
            
            if is_correct:
                correct_predictions += 1
            
            total_predictions += 1
            
            result = {
                'true': category,
                'predicted': predicted_animal,
                'confidence': confidence,
                'correct': is_correct,
                'image': image_file
            }
            results.append(result)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                current_accuracy = (correct_predictions / total_predictions) * 100
                print(f"Progress: {i + 1}/{num_samples} - Current accuracy: {current_accuracy:.1f}%")
        
        except Exception as e:
            print(f"⚠️ Error testing sample {i + 1}: {e}")
            continue
    
    # Calculate final results
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        
        print("=" * 60)
        print("🎯 FINAL RESULTS")
        print("=" * 60)
        print(f"✅ Correct predictions: {correct_predictions}")
        print(f"❌ Incorrect predictions: {total_predictions - correct_predictions}")
        print(f"📊 Total tested: {total_predictions}")
        print(f"🎯 Accuracy: {accuracy:.2f}%")
        print()
        
        # Show some example results
        print("📋 Sample Results:")
        print("-" * 40)
        for i, result in enumerate(results[:10]):  # Show first 10 results
            status = "✅" if result['correct'] else "❌"
            print(f"{status} {result['true']} → {result['predicted']} ({result['confidence']:.1%})")
        
        if len(results) > 10:
            print(f"... and {len(results) - 10} more results")
        
        print()
        
        # Performance categories
        if accuracy >= 90:
            print("🌟 Excellent performance! Model is highly accurate.")
        elif accuracy >= 75:
            print("👍 Good performance! Model works well.")
        elif accuracy >= 60:
            print("⚠️ Moderate performance. Consider more training.")
        else:
            print("🔴 Poor performance. Model needs significant improvement.")
    else:
        print("❌ No successful predictions made!")

if __name__ == "__main__":
    test_model_accuracy()