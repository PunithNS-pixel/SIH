#!/usr/bin/env python3
"""
Complete Working Crop Recommendation System
This is the final, clean version that combines all functionality
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
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load and perform basic analysis of the crop dataset"""
    print("="*80)
    print("🌾 CROP RECOMMENDATION SYSTEM")
    print("="*80)
    
    # Load the dataset
    print("Loading dataset...")
    cropdf = pd.read_csv("Crop_recommendation.csv.xls")
    
    print(f"📊 Dataset shape: {cropdf.shape}")
    print(f"🌱 Number of different crops: {len(cropdf['label'].unique())}")
    print(f"🌾 Available crops: {sorted(cropdf['label'].unique())}")
    
    print("\n📈 Dataset Statistics:")
    print(cropdf.describe().round(2))
    
    print(f"\n🔍 Missing values: {cropdf.isnull().sum().sum()}")
    
    print("\n📊 Crop Distribution:")
    crop_counts = cropdf['label'].value_counts()
    print(f"Each crop has {crop_counts.iloc[0]} samples (balanced dataset)")
    
    return cropdf

def visualize_correlations(cropdf):
    """Create correlation heatmap"""
    print("\n📊 Creating correlation analysis...")
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    correlation_matrix = cropdf.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, linewidths=0.5)
    plt.title('Feature Correlations in Crop Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('crop_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_matrix

def train_models(X, y):
    """Train multiple ML models and find the best one"""
    print("\n" + "="*80)
    print("🤖 MACHINE LEARNING MODEL TRAINING")
    print("="*80)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    
    print(f"📚 Training set: {X_train.shape[0]} samples")
    print(f"📝 Testing set: {X_test.shape[0]} samples")
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42, kernel='rbf'),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
    }
    
    # Train and evaluate models
    print(f"\n🏃‍♂️ Training {len(models)} different models...")
    model_results = {}
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
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
        
        # Display results
        print(f"   ✅ Training Accuracy: {train_accuracy:.4f} ({train_accuracy:.1%})")
        print(f"   📊 Testing Accuracy: {test_accuracy:.4f} ({test_accuracy:.1%})")
        print(f"   🎯 Overfitting Check: {train_accuracy - test_accuracy:.4f}")
        
        # Keep track of best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = model
            best_model_name = name
    
    print(f"\n" + "🏆" + "="*60)
    print(f"🥇 BEST MODEL: {best_model_name}")
    print(f"🎯 ACCURACY: {best_accuracy:.4f} ({best_accuracy:.1%})")
    print("🏆" + "="*60)
    
    return best_model, best_model_name, X_test, y_test, model_results

def evaluate_model(best_model, best_model_name, X_test, y_test, model_results, crop_labels):
    """Detailed evaluation of the best model"""
    print(f"\n📋 Detailed Evaluation of {best_model_name}:")
    
    # Classification report
    y_pred = model_results[best_model_name]['predictions']
    print("\n📊 Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Print key metrics
    accuracy = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")
    
    # Confusion Matrix
    print(f"\n🎨 Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(crop_labels), yticklabels=sorted(crop_labels))
    plt.title(f'Confusion Matrix - {best_model_name}\\nAccuracy: {accuracy:.3f}', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Crop', fontsize=12)
    plt.ylabel('Actual Crop', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n🎯 Feature Importance Analysis:")
        feature_names = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 
                        'Temperature', 'Humidity', 'pH', 'Rainfall']
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        print(feature_importance.to_string(index=False))
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
        bars = plt.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
        plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_prediction_system(best_model):
    """Create the crop prediction system"""
    def predict_crop(N, P, K, temperature, humidity, ph, rainfall, show_details=True):
        """
        🌾 Predict the best crop for given soil and weather conditions
        
        Parameters:
        -----------
        N : float
            Nitrogen content (0-140)
        P : float  
            Phosphorus content (0-145)
        K : float
            Potassium content (0-205)
        temperature : float
            Temperature in Celsius (8-44)
        humidity : float
            Relative humidity percentage (14-100)
        ph : float
            Soil pH level (3.5-10)
        rainfall : float
            Rainfall in mm (20-300)
        show_details : bool
            Whether to show detailed output
            
        Returns:
        --------
        prediction : str
            Recommended crop
        probabilities : dict
            All crops with their confidence scores
        """
        
        # Validate inputs
        validations = [
            (N, 0, 140, "Nitrogen"),
            (P, 0, 145, "Phosphorus"),
            (K, 0, 205, "Potassium"),
            (temperature, 8, 44, "Temperature"),
            (humidity, 14, 100, "Humidity"),
            (ph, 3.5, 10, "pH"),
            (rainfall, 20, 300, "Rainfall")
        ]
        
        for value, min_val, max_val, name in validations:
            if not (min_val <= value <= max_val):
                print(f"⚠️  Warning: {name} value {value} is outside typical range [{min_val}-{max_val}]")
        
        # Make prediction
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = best_model.predict(input_data)[0]
        
        # Get probabilities
        probabilities = None
        if hasattr(best_model, 'predict_proba'):
            proba = best_model.predict_proba(input_data)[0]
            crop_labels = best_model.classes_
            probabilities = dict(zip(crop_labels, proba))
            probabilities = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
        
        if show_details:
            print(f"\n{'='*60}")
            print("🌾 CROP RECOMMENDATION RESULTS")
            print(f"{'='*60}")
            print(f"📋 Input Conditions:")
            print(f"   🧪 Nitrogen (N): {N}")
            print(f"   🧪 Phosphorus (P): {P}")
            print(f"   🧪 Potassium (K): {K}")
            print(f"   🌡️  Temperature: {temperature}°C")
            print(f"   💧 Humidity: {humidity}%")
            print(f"   ⚗️  pH Level: {ph}")
            print(f"   🌧️  Rainfall: {rainfall}mm")
            
            print(f"\n🏆 PRIMARY RECOMMENDATION: {prediction.upper()}")
            
            if probabilities:
                print(f"\n📊 Top 5 Crop Recommendations:")
                for i, (crop, prob) in enumerate(list(probabilities.items())[:5]):
                    confidence = ("🟢 Very High" if prob > 0.8 else 
                                "🔵 High" if prob > 0.5 else 
                                "🟡 Medium" if prob > 0.2 else "🔴 Low")
                    print(f"   {i+1}. {crop.title()}: {prob:.3f} ({prob*100:.1f}%) - {confidence}")
        
        return prediction, probabilities
    
    return predict_crop

def run_prediction_tests(predict_crop):
    """Run comprehensive tests with different scenarios"""
    print(f"\n{'='*80}")
    print("🧪 TESTING CROP PREDICTIONS - COMPREHENSIVE SCENARIOS")
    print(f"{'='*80}")
    
    test_scenarios = [
        {
            'name': '🌾 Rice Optimal Conditions',
            'params': (90, 42, 43, 20.9, 82, 6.5, 203),
            'description': 'High nitrogen, moderate P&K, cool temperature, high humidity, neutral pH, high rainfall'
        },
        {
            'name': '🌽 Maize Growing Conditions', 
            'params': (80, 50, 20, 25, 65, 6.2, 90),
            'description': 'High nitrogen, moderate phosphorus, low potassium, warm temperature'
        },
        {
            'name': '🫘 Chickpea Conditions',
            'params': (40, 70, 80, 18, 17, 7.5, 75),
            'description': 'Low nitrogen, high P&K, cool temperature, low humidity, alkaline pH'
        },
        {
            'name': '🍎 Apple Orchard Conditions',
            'params': (20, 125, 200, 22, 60, 6.8, 180),
            'description': 'Low nitrogen, very high phosphorus & potassium, moderate temperature'
        },
        {
            'name': '☕ Coffee Plantation',
            'params': (100, 30, 30, 28, 75, 6.0, 150),
            'description': 'Very high nitrogen, low P&K, warm temperature, high humidity'
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'-'*60}")
        print(f"Test {i}/5: {scenario['name']}")
        print(f"Scenario: {scenario['description']}")
        print(f"Parameters: N={scenario['params'][0]}, P={scenario['params'][1]}, K={scenario['params'][2]}, "
              f"T={scenario['params'][3]}°C, H={scenario['params'][4]}%, pH={scenario['params'][5]}, R={scenario['params'][6]}mm")
        
        crop, _ = predict_crop(*scenario['params'])

def save_model(model, model_name):
    """Save the trained model"""
    filename = 'best_crop_recommendation_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump({
            'model': model,
            'model_name': model_name,
            'features': ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        }, f)
    print(f"\n💾 Model saved as '{filename}'")
    return filename

def main():
    """Main function - Complete Crop Recommendation System"""
    
    # Load and analyze data
    cropdf = load_and_analyze_data()
    
    # Create correlation visualization
    correlation_matrix = visualize_correlations(cropdf)
    
    # Prepare features and target
    X = cropdf.drop('label', axis=1)
    y = cropdf['label']
    
    print(f"\n🎯 Features: {list(X.columns)}")
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"🏷️  Target variable shape: {y.shape}")
    
    # Train models
    best_model, best_model_name, X_test, y_test, model_results = train_models(X, y)
    
    # Evaluate best model
    evaluate_model(best_model, best_model_name, X_test, y_test, model_results, y.unique())
    
    # Create prediction system
    predict_crop = create_prediction_system(best_model)
    
    # Run tests
    run_prediction_tests(predict_crop)
    
    # Save model
    model_file = save_model(best_model, best_model_name)
    
    # Final summary
    print(f"\n{'🎉'*60}")
    print("CROP RECOMMENDATION SYSTEM - READY FOR PRODUCTION!")
    print(f"{'🎉'*60}")
    print(f"✅ Best Model: {best_model_name}")
    print(f"✅ Accuracy: {model_results[best_model_name]['test_accuracy']:.1%}")
    print(f"✅ Total Crops Supported: {len(y.unique())}")
    print(f"✅ Model Saved: {model_file}")
    print(f"✅ Prediction Function: predict_crop() ready to use")
    
    print(f"\n📚 Usage Example:")
    print("prediction, probabilities = predict_crop(90, 42, 43, 20.9, 82, 6.5, 203)")
    print("\n🔧 For custom predictions, call predict_crop with your soil/weather data!")
    
    return best_model, predict_crop

if __name__ == "__main__":
    model, predict_function = main()