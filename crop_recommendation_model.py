#!/usr/bin/env python3
"""
Crop Recommendation System using Machine Learning
This script builds and evaluates multiple ML models to recommend crops based on soil and weather conditions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load and explore the crop recommendation dataset"""
    print("Loading Crop Recommendation Dataset...")
    
    # Load the dataset
    cropdf = pd.read_csv("Crop_recommendation.csv.xls")
    
    print(f"Dataset shape: {cropdf.shape}")
    print(f"Number of different crops: {len(cropdf['label'].unique())}")
    print(f"Crops available: {sorted(cropdf['label'].unique())}")
    print("\nFirst few rows:")
    print(cropdf.head())
    
    print("\nDataset statistics:")
    print(cropdf.describe())
    
    print("\nMissing values:")
    print(cropdf.isnull().sum())
    
    print("\nCrop distribution:")
    print(cropdf['label'].value_counts())
    
    return cropdf

def prepare_data(cropdf):
    """Prepare features and target variables"""
    print("\nPreparing data for machine learning...")
    
    # Features: N, P, K, temperature, humidity, ph, rainfall
    X = cropdf.drop('label', axis=1)
    # Target: crop label  
    y = cropdf['label']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    return X, y

def train_and_evaluate_models(X, y):
    """Train multiple ML models and compare their performance"""
    print("\n" + "="*60)
    print("TRAINING AND EVALUATING MULTIPLE MODELS")
    print("="*60)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    # Train and evaluate models
    model_results = {}
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        
        # Calculate accuracies
        test_accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        
        # Store results
        model_results[name] = {
            'model': model,
            'test_accuracy': test_accuracy,
            'train_accuracy': train_accuracy,
            'predictions': y_pred
        }
        
        print(f"{name} Results:")
        print(f"  Training Accuracy: {train_accuracy:.4f}")
        print(f"  Testing Accuracy: {test_accuracy:.4f}")
        print(f"  Overfitting Check: {train_accuracy - test_accuracy:.4f}")
        
        # Keep track of best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = model
            best_model_name = name
    
    print(f"\n" + "="*60)
    print(f"BEST MODEL: {best_model_name}")
    print(f"BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy:.1%})")
    print("="*60)
    
    return best_model, best_model_name, X_test, y_test, model_results

def evaluate_best_model(best_model, best_model_name, X_test, y_test, model_results):
    """Provide detailed evaluation of the best model"""
    print(f"\nDetailed evaluation of {best_model_name}:")
    
    # Classification report
    y_pred = model_results[best_model_name]['predictions']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(14, 12))
    crop_labels = sorted(y_test.unique())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=crop_labels, yticklabels=crop_labels)
    plt.title(f'Confusion Matrix - {best_model_name}\nAccuracy: {model_results[best_model_name]["test_accuracy"]:.4f}')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        print(f"\nFeature importance analysis for {best_model_name}:")
        feature_names = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        print(feature_importance)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.title(f'Feature Importance - {best_model_name}')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_prediction_function(best_model):
    """Create a function to predict crops for new data"""
    def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
        """
        Predict the best crop for given soil and weather conditions
        
        Parameters:
        - N: Nitrogen content in soil
        - P: Phosphorus content in soil  
        - K: Potassium content in soil
        - temperature: Temperature in Celsius
        - humidity: Relative humidity in %
        - ph: pH value of soil
        - rainfall: Rainfall in mm
        
        Returns:
        - predicted_crop: The recommended crop
        - probabilities: Dictionary of all crops with their probabilities
        """
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = best_model.predict(input_data)
        
        probabilities = None
        if hasattr(best_model, 'predict_proba'):
            proba = best_model.predict_proba(input_data)[0]
            crop_labels = best_model.classes_
            probabilities = dict(zip(crop_labels, proba))
            # Sort by probability (highest first)
            probabilities = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
        
        return prediction[0], probabilities
    
    return predict_crop

def test_predictions(predict_crop_func):
    """Test the prediction function with sample data"""
    print("\n" + "="*60)
    print("TESTING CROP PREDICTIONS")
    print("="*60)
    
    # Test cases with different soil and weather conditions
    test_cases = [
        {
            'name': 'Rice-optimal conditions',
            'params': (90, 42, 43, 20.9, 82, 6.5, 203),
            'description': 'High N, moderate P&K, cool temp, high humidity, neutral pH, high rainfall'
        },
        {
            'name': 'Maize-optimal conditions', 
            'params': (80, 50, 20, 25, 65, 6.2, 90),
            'description': 'High N, moderate P, low K, warm temp, moderate humidity, slightly acidic pH, moderate rainfall'
        },
        {
            'name': 'Chickpea-optimal conditions',
            'params': (40, 70, 80, 18, 17, 7.5, 75),
            'description': 'Low N, high P&K, cool temp, low humidity, alkaline pH, low rainfall'
        },
        {
            'name': 'Cotton-optimal conditions',
            'params': (120, 45, 30, 28, 70, 6.8, 120),
            'description': 'Very high N, moderate P, low K, hot temp, moderate humidity, neutral pH, moderate rainfall'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print(f"Input: N={test_case['params'][0]}, P={test_case['params'][1]}, K={test_case['params'][2]}, "
              f"Temp={test_case['params'][3]}°C, Humidity={test_case['params'][4]}%, "
              f"pH={test_case['params'][5]}, Rainfall={test_case['params'][6]}mm")
        print(f"Description: {test_case['description']}")
        
        crop, probabilities = predict_crop_func(*test_case['params'])
        print(f"Predicted crop: {crop}")
        
        if probabilities:
            print("Top 5 recommendations with confidence:")
            for i, (crop_name, prob) in enumerate(list(probabilities.items())[:5]):
                print(f"  {i+1}. {crop_name}: {prob:.3f} ({prob*100:.1f}%)")

def main():
    """Main function to run the complete crop recommendation system"""
    print("="*60)
    print("CROP RECOMMENDATION SYSTEM")
    print("="*60)
    
    # Load and explore data
    cropdf = load_and_explore_data()
    
    # Prepare data
    X, y = prepare_data(cropdf)
    
    # Train and evaluate models
    best_model, best_model_name, X_test, y_test, model_results = train_and_evaluate_models(X, y)
    
    # Detailed evaluation
    evaluate_best_model(best_model, best_model_name, X_test, y_test, model_results)
    
    # Create prediction function
    predict_crop = create_prediction_function(best_model)
    
    # Test predictions
    test_predictions(predict_crop)
    
    print(f"\n" + "="*60)
    print("CROP RECOMMENDATION SYSTEM READY!")
    print("="*60)
    print(f"Best Model: {best_model_name}")
    print(f"Accuracy: {model_results[best_model_name]['test_accuracy']:.1%}")
    print("Use the predict_crop() function to get crop recommendations")
    print("Example: predict_crop(90, 42, 43, 20.9, 82, 6.5, 203)")
    
    return best_model, predict_crop

if __name__ == "__main__":
    model, predict_function = main()