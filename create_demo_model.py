# Quick Perfect Model Demo Data
import pickle
import numpy as np
from datetime import datetime

# Create demo trained features for common farm animals
demo_trained_features = {
    'Cow': {
        'brightness': {'mean': 140.5, 'std': 35.2, 'min': 80, 'max': 200},
        'mean_red': {'mean': 145.3, 'std': 30.1, 'min': 90, 'max': 190},
        'mean_green': {'mean': 138.7, 'std': 28.9, 'min': 85, 'max': 185},
        'mean_blue': {'mean': 135.2, 'std': 32.1, 'min': 80, 'max': 180},
        'aspect_ratio': {'mean': 1.4, 'std': 0.3, 'min': 1.0, 'max': 2.0},
        'edge_density': {'mean': 0.12, 'std': 0.05, 'min': 0.05, 'max': 0.25}
    },
    'Horse': {
        'brightness': {'mean': 110.2, 'std': 40.1, 'min': 60, 'max': 170},
        'mean_red': {'mean': 115.8, 'std': 35.2, 'min': 70, 'max': 160},
        'mean_green': {'mean': 108.3, 'std': 38.7, 'min': 65, 'max': 155},
        'mean_blue': {'mean': 106.9, 'std': 36.4, 'min': 60, 'max': 150},
        'aspect_ratio': {'mean': 1.6, 'std': 0.4, 'min': 1.2, 'max': 2.2},
        'edge_density': {'mean': 0.15, 'std': 0.06, 'min': 0.08, 'max': 0.30}
    },
    'Goat': {
        'brightness': {'mean': 135.7, 'std': 32.8, 'min': 85, 'max': 190},
        'mean_red': {'mean': 140.2, 'std': 28.5, 'min': 95, 'max': 180},
        'mean_green': {'mean': 134.8, 'std': 30.1, 'min': 90, 'max': 175},
        'mean_blue': {'mean': 132.1, 'std': 29.7, 'min': 85, 'max': 170},
        'aspect_ratio': {'mean': 1.1, 'std': 0.25, 'min': 0.8, 'max': 1.5},
        'edge_density': {'mean': 0.18, 'std': 0.07, 'min': 0.10, 'max': 0.35}
    },
    'Sheep': {
        'brightness': {'mean': 165.3, 'std': 25.4, 'min': 120, 'max': 210},
        'mean_red': {'mean': 170.5, 'std': 22.1, 'min': 130, 'max': 200},
        'mean_green': {'mean': 168.2, 'std': 24.3, 'min': 125, 'max': 195},
        'mean_blue': {'mean': 165.8, 'std': 26.7, 'min': 120, 'max': 190},
        'aspect_ratio': {'mean': 1.2, 'std': 0.2, 'min': 0.9, 'max': 1.6},
        'edge_density': {'mean': 0.22, 'std': 0.08, 'min': 0.12, 'max': 0.40}
    },
    'Pig': {
        'brightness': {'mean': 155.8, 'std': 28.9, 'min': 110, 'max': 200},
        'mean_red': {'mean': 160.3, 'std': 25.7, 'min': 120, 'max': 190},
        'mean_green': {'mean': 156.1, 'std': 27.2, 'min': 115, 'max': 185},
        'mean_blue': {'mean': 152.4, 'std': 29.8, 'min': 110, 'max': 180},
        'aspect_ratio': {'mean': 1.5, 'std': 0.35, 'min': 1.1, 'max': 2.0},
        'edge_density': {'mean': 0.10, 'std': 0.04, 'min': 0.05, 'max': 0.20}
    },
    'Chicken': {
        'brightness': {'mean': 180.2, 'std': 30.5, 'min': 130, 'max': 220},
        'mean_red': {'mean': 185.7, 'std': 28.1, 'min': 140, 'max': 210},
        'mean_green': {'mean': 182.3, 'std': 29.7, 'min': 135, 'max': 205},
        'mean_blue': {'mean': 175.8, 'std': 32.4, 'min': 125, 'max': 200},
        'aspect_ratio': {'mean': 0.85, 'std': 0.15, 'min': 0.6, 'max': 1.2},
        'edge_density': {'mean': 0.16, 'std': 0.06, 'min': 0.08, 'max': 0.28}
    },
    'Dog': {
        'brightness': {'mean': 125.4, 'std': 45.2, 'min': 70, 'max': 190},
        'mean_red': {'mean': 130.8, 'std': 42.1, 'min': 75, 'max': 185},
        'mean_green': {'mean': 127.2, 'std': 44.3, 'min': 70, 'max': 180},
        'mean_blue': {'mean': 122.9, 'std': 46.7, 'min': 65, 'max': 175},
        'aspect_ratio': {'mean': 1.3, 'std': 0.4, 'min': 0.8, 'max': 1.9},
        'edge_density': {'mean': 0.20, 'std': 0.08, 'min': 0.10, 'max': 0.35}
    },
    'Cat': {
        'brightness': {'mean': 142.6, 'std': 38.7, 'min': 90, 'max': 200},
        'mean_red': {'mean': 148.3, 'std': 35.4, 'min': 95, 'max': 190},
        'mean_green': {'mean': 144.1, 'std': 37.8, 'min': 90, 'max': 185},
        'mean_blue': {'mean': 138.4, 'std': 40.2, 'min': 85, 'max': 180},
        'aspect_ratio': {'mean': 1.0, 'std': 0.2, 'min': 0.7, 'max': 1.4},
        'edge_density': {'mean': 0.25, 'std': 0.09, 'min': 0.15, 'max': 0.40}
    }
}

# Create demo model data
model_data = {
    'classes': ['Cow', 'Horse', 'Goat', 'Sheep', 'Pig', 'Chicken', 'Dog', 'Cat'],
    'trained_features': demo_trained_features,
    'classification_rules': {},
    'training_date': datetime.now().isoformat(),
    'model_type': 'Perfect Animal Classifier v2.0 (Demo)'
}

# Save demo model
with open('perfect_animal_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✅ Demo perfect model created!")