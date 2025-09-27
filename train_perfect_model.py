#!/usr/bin/env python3
"""
Perfect Animal Classification Model Training System
Using Farm Harmful Animal Dataset with Advanced Deep Learning
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7, ResNet152V2, InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

class PerfectAnimalClassifier:
    """Advanced animal classification system with ensemble learning"""
    
    def __init__(self, dataset_path="/Users/punithns/Desktop/SIH/Farm Harmful Animal Dataset"):
        self.dataset_path = Path(dataset_path)
        self.input_size = (224, 224, 3)
        self.batch_size = 32
        self.classes = []
        self.models = {}
        self.ensemble_model = None
        
        # Initialize GPU if available
        self.setup_gpu()
        
    def setup_gpu(self):
        """Configure GPU for optimal performance"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU acceleration enabled: {len(gpus)} GPU(s) found")
            except RuntimeError as e:
                print(f"⚠️ GPU setup warning: {e}")
        else:
            print("💻 Using CPU for training")
            
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset with advanced augmentation"""
        print("🔍 Loading Farm Harmful Animal Dataset...")
        
        # Get class names from training directories
        train_path = self.dataset_path / "train"
        self.classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
        print(f"📋 Found {len(self.classes)} animal classes: {self.classes}")
        
        # Advanced data augmentation for robust training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.8, 1.2],
            channel_shift_range=20,
            fill_mode='nearest'
        )
        
        # Validation data (minimal augmentation)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            horizontal_flip=True
        )
        
        # Test data (no augmentation)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create data generators
        self.train_generator = train_datagen.flow_from_directory(
            self.dataset_path / "train",
            target_size=self.input_size[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        self.validation_generator = val_datagen.flow_from_directory(
            self.dataset_path / "validation",
            target_size=self.input_size[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            self.dataset_path / "test",
            target_size=self.input_size[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"📊 Training samples: {self.train_generator.samples}")
        print(f"📊 Validation samples: {self.validation_generator.samples}")
        print(f"📊 Test samples: {self.test_generator.samples}")
        
    def create_advanced_model(self, base_model_name="EfficientNetB7"):
        """Create state-of-the-art model with transfer learning"""
        print(f"🏗️ Building advanced model with {base_model_name}...")
        
        # Load pre-trained base model
        if base_model_name == "EfficientNetB7":
            base_model = EfficientNetB7(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_size
            )
        elif base_model_name == "ResNet152V2":
            base_model = ResNet152V2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_size
            )
        elif base_model_name == "InceptionV3":
            base_model = InceptionV3(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_size
            )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu', name='dense_1024')(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation='relu', name='dense_512')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu', name='dense_256')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(len(self.classes), activation='softmax', name='predictions')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile with advanced optimizer
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        return model
    
    def train_ensemble_models(self):
        """Train multiple models for ensemble learning"""
        print("🚀 Training ensemble of advanced models...")
        
        model_architectures = ["EfficientNetB7", "ResNet152V2", "InceptionV3"]
        
        for arch in model_architectures:
            print(f"\n🏋️ Training {arch} model...")
            
            # Create model
            model = self.create_advanced_model(arch)
            
            # Callbacks for optimal training
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=8,
                    min_lr=1e-7,
                    verbose=1
                ),
                ModelCheckpoint(
                    f'best_{arch.lower()}_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Stage 1: Train with frozen base model
            print(f"Stage 1: Training classifier head for {arch}...")
            history1 = model.fit(
                self.train_generator,
                epochs=20,
                validation_data=self.validation_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            # Stage 2: Fine-tune entire model
            print(f"Stage 2: Fine-tuning entire {arch} model...")
            model.trainable = True
            
            # Use lower learning rate for fine-tuning
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'top_3_accuracy']
            )
            
            history2 = model.fit(
                self.train_generator,
                epochs=30,
                validation_data=self.validation_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save trained model
            self.models[arch] = model
            model.save(f'perfect_{arch.lower()}_animal_classifier.h5')
            print(f"✅ {arch} model training completed!")
            
    def create_ensemble_predictor(self):
        """Create ensemble predictor combining all models"""
        print("🔗 Creating ensemble predictor...")
        
        def ensemble_predict(image):
            """Ensemble prediction using all trained models"""
            predictions = []
            
            # Preprocess image
            if isinstance(image, str):
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.array(image)
            
            img = cv2.resize(img, self.input_size[:2])
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Get predictions from each model
            for arch, model in self.models.items():
                pred = model.predict(img, verbose=0)
                predictions.append(pred)
            
            # Ensemble averaging
            ensemble_pred = np.mean(predictions, axis=0)
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(ensemble_pred)
            confidence = float(ensemble_pred[0][predicted_class_idx])
            predicted_class = self.classes[predicted_class_idx]
            
            return predicted_class, confidence * 100
        
        self.ensemble_predict = ensemble_predict
        
    def evaluate_models(self):
        """Comprehensive evaluation of all models"""
        print("📊 Evaluating model performance...")
        
        results = {}
        
        for arch, model in self.models.items():
            print(f"\n🔍 Evaluating {arch}...")
            
            # Get predictions
            test_predictions = model.predict(self.test_generator, verbose=1)
            test_pred_classes = np.argmax(test_predictions, axis=1)
            
            # True labels
            true_classes = self.test_generator.classes
            
            # Calculate accuracy
            accuracy = np.mean(test_pred_classes == true_classes)
            print(f"{arch} Test Accuracy: {accuracy:.4f}")
            
            # Classification report
            report = classification_report(
                true_classes, 
                test_pred_classes, 
                target_names=self.classes,
                output_dict=True
            )
            
            results[arch] = {
                'accuracy': accuracy,
                'report': report
            }
            
        return results
    
    def save_perfect_model(self):
        """Save the perfect model system"""
        print("💾 Saving perfect model system...")
        
        # Create model info
        model_info = {
            'classes': self.classes,
            'input_size': self.input_size,
            'trained_models': list(self.models.keys()),
            'training_date': datetime.now().isoformat(),
            'performance': 'Ensemble of EfficientNetB7, ResNet152V2, InceptionV3'
        }
        
        # Save model info
        joblib.dump(model_info, 'perfect_animal_classifier_info.pkl')
        
        # Save class mapping
        with open('animal_classes.txt', 'w') as f:
            for i, cls in enumerate(self.classes):
                f.write(f"{i}: {cls}\n")
        
        print("✅ Perfect model system saved!")
        
    def train_perfect_system(self):
        """Complete training pipeline for perfect animal classification"""
        print("🎯 Starting Perfect Animal Classification Training...")
        print("=" * 60)
        
        # Step 1: Load and preprocess data
        self.load_and_preprocess_data()
        
        # Step 2: Train ensemble models
        self.train_ensemble_models()
        
        # Step 3: Create ensemble predictor
        self.create_ensemble_predictor()
        
        # Step 4: Evaluate models
        results = self.evaluate_models()
        
        # Step 5: Save perfect model
        self.save_perfect_model()
        
        print("=" * 60)
        print("🎉 PERFECT ANIMAL CLASSIFICATION SYSTEM TRAINED!")
        print("✨ Ready for deployment with maximum accuracy!")
        
        return results

def main():
    """Main training function"""
    print("🚀 Perfect Animal Classifier Training System")
    print("Using Farm Harmful Animal Dataset")
    print("-" * 50)
    
    # Initialize trainer
    trainer = PerfectAnimalClassifier()
    
    # Train the perfect system
    results = trainer.train_perfect_system()
    
    # Display final results
    print("\n📈 FINAL RESULTS:")
    for arch, result in results.items():
        print(f"{arch}: {result['accuracy']:.4f} accuracy")
    
if __name__ == "__main__":
    main()