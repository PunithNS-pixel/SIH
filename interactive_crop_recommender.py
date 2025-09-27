#!/usr/bin/env python3
"""
Interactive Crop Recommendation System
Simple interface to get crop recommendations based on soil and weather conditions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class CropRecommender:
    def __init__(self):
        self.model = None
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.is_trained = False
        
    def load_and_train(self, data_file='Crop_recommendation.csv.xls'):
        """Load data and train the model"""
        print("Loading and training the crop recommendation model...")
        
        # Load dataset
        df = pd.read_csv(data_file)
        
        # Prepare features and target
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train Naive Bayes model (best performing)
        self.model = GaussianNB()
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        print(f"Model trained successfully! Accuracy: {accuracy:.2%}")
        print(f"Available crops: {sorted(df['label'].unique())}")
        
        return accuracy
    
    def predict(self, N, P, K, temperature, humidity, ph, rainfall):
        """Predict the best crop for given conditions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call load_and_train() first.")
        
        # Prepare input
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Get prediction
        prediction = self.model.predict(input_data)[0]
        
        # Get probabilities
        probabilities = self.model.predict_proba(input_data)[0]
        crop_labels = self.model.classes_
        
        # Create probability dictionary
        crop_probabilities = dict(zip(crop_labels, probabilities))
        crop_probabilities = dict(sorted(crop_probabilities.items(), key=lambda x: x[1], reverse=True))
        
        return prediction, crop_probabilities
    
    def get_recommendations(self, N, P, K, temperature, humidity, ph, rainfall, top_n=5):
        """Get top N crop recommendations with detailed info"""
        prediction, probabilities = self.predict(N, P, K, temperature, humidity, ph, rainfall)
        
        print(f"\n{'='*60}")
        print("CROP RECOMMENDATION RESULTS")
        print(f"{'='*60}")
        print(f"Input Conditions:")
        print(f"  Nitrogen (N): {N}")
        print(f"  Phosphorus (P): {P}")
        print(f"  Potassium (K): {K}")
        print(f"  Temperature: {temperature}°C")
        print(f"  Humidity: {humidity}%")
        print(f"  pH: {ph}")
        print(f"  Rainfall: {rainfall}mm")
        print(f"\nPRIMARY RECOMMENDATION: {prediction.upper()}")
        print(f"\nTop {top_n} Recommendations:")
        
        for i, (crop, prob) in enumerate(list(probabilities.items())[:top_n]):
            confidence = "Very High" if prob > 0.8 else "High" if prob > 0.5 else "Medium" if prob > 0.2 else "Low"
            print(f"  {i+1}. {crop.title()}: {prob:.3f} ({prob*100:.1f}%) - {confidence} confidence")
        
        return prediction, list(probabilities.items())[:top_n]

def interactive_mode():
    """Run interactive mode for getting recommendations"""
    recommender = CropRecommender()
    
    # Train the model
    try:
        recommender.load_and_train()
    except FileNotFoundError:
        print("Error: Crop_recommendation.csv.xls file not found!")
        print("Please ensure the data file is in the same directory.")
        return
    
    print(f"\n{'='*60}")
    print("INTERACTIVE CROP RECOMMENDATION SYSTEM")
    print(f"{'='*60}")
    print("Enter soil and weather conditions to get crop recommendations")
    print("Type 'quit' to exit, 'example' for sample inputs")
    
    while True:
        print(f"\n{'-'*40}")
        user_input = input("Enter command (recommend/example/quit): ").strip().lower()
        
        if user_input == 'quit':
            print("Thank you for using the Crop Recommendation System!")
            break
        
        elif user_input == 'example':
            print("\nRunning example predictions...")
            
            examples = [
                {"name": "Rice conditions", "params": (90, 42, 43, 20.9, 82, 6.5, 203)},
                {"name": "Wheat conditions", "params": (80, 50, 20, 25, 65, 6.2, 90)},
                {"name": "Chickpea conditions", "params": (40, 70, 80, 18, 17, 7.5, 75)},
            ]
            
            for example in examples:
                print(f"\n--- {example['name']} ---")
                recommender.get_recommendations(*example['params'])
        
        elif user_input == 'recommend':
            try:
                print("\nEnter soil and weather conditions:")
                N = float(input("Nitrogen (N) content (0-140): "))
                P = float(input("Phosphorus (P) content (0-145): "))
                K = float(input("Potassium (K) content (0-205): "))
                temperature = float(input("Temperature (°C) (8-44): "))
                humidity = float(input("Humidity (%) (14-100): "))
                ph = float(input("pH level (3.5-10): "))
                rainfall = float(input("Rainfall (mm) (20-300): "))
                
                # Validate inputs
                if not (0 <= N <= 140):
                    print("Warning: Nitrogen should be between 0-140")
                if not (0 <= P <= 145):
                    print("Warning: Phosphorus should be between 0-145")
                if not (0 <= K <= 205):
                    print("Warning: Potassium should be between 0-205")
                if not (8 <= temperature <= 44):
                    print("Warning: Temperature should be between 8-44°C")
                if not (14 <= humidity <= 100):
                    print("Warning: Humidity should be between 14-100%")
                if not (3.5 <= ph <= 10):
                    print("Warning: pH should be between 3.5-10")
                if not (20 <= rainfall <= 300):
                    print("Warning: Rainfall should be between 20-300mm")
                
                recommender.get_recommendations(N, P, K, temperature, humidity, ph, rainfall)
                
            except ValueError:
                print("Error: Please enter valid numbers for all inputs.")
            except Exception as e:
                print(f"Error: {e}")
        
        else:
            print("Invalid command. Use 'recommend', 'example', or 'quit'.")

def quick_predict(N, P, K, temperature, humidity, ph, rainfall):
    """Quick prediction function for direct use"""
    recommender = CropRecommender()
    recommender.load_and_train()
    return recommender.get_recommendations(N, P, K, temperature, humidity, ph, rainfall)

if __name__ == "__main__":
    # Run interactive mode
    interactive_mode()