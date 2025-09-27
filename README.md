# 🌾 Crop Recommendation System - Streamlit Web App

A modern, AI-powered web application for intelligent crop recommendations based on soil nutrients and weather conditions.

## 🚀 Features

- **🎯 Smart Predictions**: AI model with 99.4% accuracy
- **🌱 Fertilizer Recommendations**: Intelligent fertilizer suggestions for each crop
- **🌱 22 Crop Types**: Comprehensive crop database
- **📊 Interactive Dashboard**: Real-time analysis and visualization
- **🔍 Data Insights**: Detailed dataset analysis and correlations
- **💡 Soil Analysis**: NPK levels and pH optimization recommendations
- **📚 Fertilizer Database**: Complete fertilizer guide for all crops
- **📱 Responsive Design**: Works on desktop and mobile
- **⚡ Fast Performance**: Instant predictions with caching

## 📸 Screenshots

### Main Prediction Interface
![Prediction Interface](https://via.placeholder.com/800x400?text=Crop+Prediction+Interface)

### Dataset Analysis Dashboard
![Analysis Dashboard](https://via.placeholder.com/800x400?text=Dataset+Analysis+Dashboard)

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd SIH
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open in browser**
   - The app will automatically open in your browser
   - Or visit `http://localhost:8501`

## 🌐 Deployment Options

### 1. Streamlit Community Cloud (Recommended - FREE)

**Step-by-step deployment:**

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Crop Recommendation System"
   git branch -M main
   git remote add origin https://github.com/yourusername/crop-recommendation.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_app.py`
   - Click "Deploy!"

3. **Your app will be live at**: `https://yourusername-crop-recommendation-streamlit-app-xyz.streamlit.app`

### 2. Heroku Deployment

1. **Create additional files**:

   **Procfile**:
   ```
   web: sh setup.sh && streamlit run streamlit_app.py
   ```

   **setup.sh**:
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [general]\n\
   email = \"your-email@domain.com\"\n\
   " > ~/.streamlit/credentials.toml
   echo "\
   [server]\n\
   headless = true\n\
   enableCORS=false\n\
   port = $PORT\n\
   " > ~/.streamlit/config.toml
   ```

2. **Deploy to Heroku**:
   ```bash
   # Install Heroku CLI first
   heroku create your-app-name
   git push heroku main
   ```

### 3. Railway Deployment

1. **Connect your GitHub repo to Railway**
2. **Railway will auto-detect and deploy**
3. **Set start command**: `streamlit run streamlit_app.py --server.port=$PORT`

### 4. Google Cloud Platform

1. **Create app.yaml**:
   ```yaml
   runtime: python39
   env_variables:
     PORT: 8080
   ```

2. **Deploy**:
   ```bash
   gcloud app deploy
   ```

## 📁 Project Structure

```
SIH/
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── Crop_recommendation.csv.xls   # Dataset
├── final_crop_recommender.py     # Core ML model
├── interactive_crop_recommender.py # CLI version
├── README.md                     # This file
├── PROJECT_SUMMARY.md           # Project documentation
└── deployment/                  # Deployment configurations
    ├── Procfile                # Heroku config
    ├── setup.sh               # Streamlit config
    └── app.yaml              # GCP config
```

## 🎯 Usage Guide

### Making Predictions

1. **Navigate to "Crop Prediction" page**
2. **Enter soil conditions**:
   - Nitrogen (N): 0-140
   - Phosphorus (P): 0-145  
   - Potassium (K): 0-205
   - pH Level: 3.5-10

3. **Enter weather conditions**:
   - Temperature: 8-44°C
   - Humidity: 14-100%
   - Rainfall: 20-300mm

4. **Use Quick Presets** for common scenarios
5. **Click "Predict Best Crop"**
6. **View results** with confidence scores and recommendations

### Getting Fertilizer Recommendations

1. **Navigate to "Fertilizer Recommendation" page**
2. **Choose from three options**:
   - **Quick Recommendation**: Get instant crop + fertilizer suggestions
   - **Detailed Analysis**: Deep dive into specific crop requirements
   - **Fertilizer Database**: Browse complete fertilizer guide

3. **Enter soil conditions** (same as crop prediction)
4. **Get comprehensive recommendations**:
   - Primary fertilizers (Urea, DAP, MOP)
   - NPK ratios and application schedules
   - Organic alternatives
   - Micronutrient requirements
   - pH optimization tips
   - Expert recommendations

### Exploring Data

1. **Go to "Dataset Analysis" page**
2. **View dataset statistics and distributions**
3. **Analyze feature correlations**
4. **Explore crop-wise patterns**

## 🤖 Model Information

- **Algorithm**: Gaussian Naive Bayes
- **Accuracy**: 99.4% on test data
- **Training Data**: 2,200 samples
- **Features**: 7 input parameters
- **Crops Supported**: 22 different types

## 📊 Supported Crops

| Cereals | Legumes | Fruits | Cash Crops |
|---------|---------|---------|------------|
| Rice | Chickpea | Apple | Cotton |
| Maize | Lentil | Banana | Jute |
| - | Blackgram | Grapes | Coffee |
| - | Kidneybeans | Orange | - |
| - | Mothbeans | Mango | - |
| - | Mungbean | Papaya | - |
| - | Pigeonpeas | Pomegranate | - |
| - | - | Watermelon | - |
| - | - | Muskmelon | - |
| - | - | Coconut | - |

## 🔧 Configuration

### Environment Variables (Optional)

```bash
# For production deployment
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Customization

You can customize the app by modifying:

- **Colors & Styling**: Edit CSS in `streamlit_app.py`
- **Model Parameters**: Modify `train_model()` function
- **Input Ranges**: Update validation in `validate_inputs()`
- **Crop Database**: Add new crops to the dataset

## 🚀 Performance Optimization

The app includes several optimizations:

- **@st.cache_data**: Caches dataset loading
- **@st.cache_resource**: Caches model training
- **Lazy Loading**: Components load on demand
- **Efficient Plotting**: Uses Plotly for interactive charts

## 🔒 Security & Privacy

- **No Data Storage**: Input data is not stored or logged
- **Client-Side Processing**: Predictions made locally
- **Secure Deployment**: HTTPS enabled on all platforms
- **No External APIs**: Fully self-contained application

## 📞 Support & Contributing

### Getting Help
- **Issues**: Create a GitHub issue
- **Feature Requests**: Submit via GitHub discussions
- **Documentation**: Check PROJECT_SUMMARY.md

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📈 Roadmap

### Upcoming Features
- [ ] **Regional Models**: Location-specific recommendations
- [ ] **Weather API Integration**: Real-time weather data
- [ ] **Yield Prediction**: Expected crop yield estimates
- [ ] **Economic Analysis**: Crop profitability insights
- [ ] **Mobile App**: Native mobile application
- [ ] **Multi-language Support**: Localization for different regions

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🏆 Acknowledgments

- **Smart India Hackathon (SIH)** for the opportunity
- **Scikit-learn** for machine learning capabilities
- **Streamlit** for the amazing web framework
- **Agricultural Research Community** for domain knowledge

---

**Made with ❤️ for Smart Agriculture**

*Last updated: September 2025*