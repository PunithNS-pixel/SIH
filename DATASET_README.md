# 🐾 Farm Harmful Animal Dataset

## 📊 Dataset Overview

This repository includes a **sample dataset** for testing and development. The complete dataset contains **17,573 images** across **15 animal categories**.

### 🏷️ **Animal Categories (15 classes):**
- **Armadillos** - Small armored mammals
- **Bear** - Large carnivorous mammals  
- **Birds** - Various harmful bird species
- **Cow** - Domestic cattle
- **Crocodile** - Large reptilian predators
- **Deer** - Herbivorous mammals
- **Elephant** - Large herbivorous mammals
- **Goat** - Domestic goats
- **Horse** - Domestic horses
- **Jaguar** - Large wild cats
- **Monkey** - Primates
- **Rabbit** - Small herbivorous mammals
- **Skunk** - Small carnivorous mammals
- **Tiger** - Large wild cats
- **Wild Boar** - Wild pigs

## 📁 **Dataset Structure**

```
Sample_Farm_Animal_Dataset/
└── train/
    ├── Armadilles/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── image3.jpg
    ├── Bear/
    ├── Birds/
    ├── Cow/
    ├── Crocodile/
    ├── Deer/
    ├── Elephant/
    ├── Goat/
    ├── Horse/
    ├── Jaguar/
    ├── Monkey/
    ├── Rabbit/
    ├── Skunk/
    ├── Tiger/
    └── Wild Boar/
```

## 📈 **Dataset Statistics**

| **Metric** | **Sample Dataset** | **Full Dataset** |
|------------|-------------------|------------------|
| **Total Images** | 45 (3 per class) | 17,573 |
| **Size** | 3.6 MB | 865 MB |
| **Classes** | 15 | 15 |
| **Format** | JPG | JPG + Labels |
| **Resolution** | Various | Various |

## 🚀 **Getting the Full Dataset**

### **Option 1: Download from Source**
The complete dataset is available from:
- **Kaggle**: Search "Farm Harmful Animal Dataset"
- **Roboflow**: Computer vision datasets
- **GitHub Releases**: Check repository releases for compressed versions

### **Option 2: Generate from Sample**
Use the provided sample to:
1. Test your model architecture
2. Validate data preprocessing
3. Develop training pipeline
4. Create augmented datasets

### **Option 3: Data Augmentation**
Expand the sample dataset using:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)
```

## 🔧 **Using the Dataset**

### **Quick Start:**
```python
import os
from PIL import Image
import matplotlib.pyplot as plt

# Load sample images
dataset_path = "Sample_Farm_Animal_Dataset/train"
categories = os.listdir(dataset_path)

for category in categories:
    category_path = os.path.join(dataset_path, category)
    images = os.listdir(category_path)
    print(f"{category}: {len(images)} sample images")
```

### **Integration with KRISHI AI:**
```python
from perfect_animal_classifier import PerfectAnimalClassifier

# Initialize classifier
classifier = PerfectAnimalClassifier()

# Train on sample data
classifier.train_on_sample_dataset("Sample_Farm_Animal_Dataset/train")

# Classify new images
result = classifier.classify("new_animal_image.jpg")
```

## 📝 **Dataset Usage Guidelines**

### ✅ **Recommended Uses:**
- 🧪 **Model Development** - Test algorithms
- 🔄 **Data Preprocessing** - Develop pipelines  
- 📊 **Visualization** - Create demos
- 🎯 **Proof of Concept** - Validate approaches

### ⚠️ **Limitations:**
- 📉 **Limited Training Data** - Only 3 samples per class
- 🎯 **Reduced Accuracy** - May not generalize well
- 🔍 **No Validation Split** - Additional data needed for proper training
- 📸 **No Bounding Boxes** - Classification only, no object detection

## 🎯 **For Production Use**

To achieve the **96-98% accuracy** mentioned in the KRISHI AI system:

1. **Download Full Dataset** (865 MB, 17,573 images)
2. **Use Transfer Learning** with pre-trained models
3. **Implement Data Augmentation** 
4. **Cross-validation** with multiple data splits
5. **Ensemble Methods** combining multiple models

## 📚 **References & Citations**

If you use this dataset in your research, please cite:
```
Farm Harmful Animal Dataset
- Purpose: Agricultural pest and wildlife management
- Classes: 15 animal categories
- Application: Crop protection and farm security
- Size: 17,573 labeled images
```

## 🔗 **Related Resources**

- 📖 **KRISHI AI Documentation**: Complete system guide
- 🤖 **Perfect Animal Classifier**: Custom trained model  
- 🔧 **Training Scripts**: `train_perfect_model.py`
- 🧪 **Test Suite**: `test_model_accuracy.py`

---

💡 **Note**: This sample dataset is perfect for development and testing. For production deployment, consider downloading the full dataset or implementing data augmentation techniques to improve model performance.