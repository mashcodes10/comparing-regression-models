# 🏠 Airbnb Price Prediction: Comparing Regression Models

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-green.svg)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-yellow.svg)](https://pandas.pydata.org)

A comprehensive comparison of multiple regression models for predicting Airbnb listing prices. This project demonstrates machine learning model selection, ensemble methods, and performance evaluation techniques.

## 🎯 Project Overview

This project systematically compares **5 different regression algorithms** to predict Airbnb listing prices, showcasing the power of ensemble methods over individual models. The analysis includes proper model evaluation, hyperparameter tuning, and performance visualization.

## 📊 Dataset

- **Airbnb Listings Dataset**: Pre-processed NYC Airbnb listings
- **File Location**: `data_regressors/airbnbData_train.csv`
- **Target Variable**: `price` (continuous)
- **Features**: Room type, location, amenities, host information, reviews
- **Data Split**: 70% training, 30% testing
- **Preprocessing**: One-hot encoding, scaling, missing value imputation already applied
- **Dataset Size**: Ready-to-use preprocessed data for immediate modeling

## 🤖 Models Compared

| Model | Type | RMSE | R² Score | Key Features |
|-------|------|------|----------|--------------|
| **Random Forest** 🏆 | Ensemble | **0.620** | **0.601** | Best overall performance |
| **Gradient Boosting** | Ensemble | 0.650 | 0.562 | Sequential learning |
| **Stacking Ensemble** | Meta-learner | 0.683 | 0.516 | Combines multiple models |
| **Decision Tree** | Tree-based | 0.714 | 0.470 | Interpretable |
| **Linear Regression** | Linear | 0.722 | 0.459 | Baseline model |

## 🏆 Key Results

### Performance Summary
- **Winner**: Random Forest Regressor
  - **RMSE**: $0.62 (lowest prediction error)
  - **R² Score**: 0.601 (explains 60% of price variance)
  - **Key Strength**: Handles non-linear relationships effectively

### Model Insights
1. **Ensemble methods outperformed individual models**
2. **Tree-based models** showed superior performance over linear models
3. **Hyperparameter tuning** improved Decision Tree performance significantly
4. **Stacking ensemble** provided good results but with increased complexity

## 📈 Detailed Analysis

### Hyperparameter Optimization
- **GridSearchCV** with 3-fold cross-validation
- **Decision Tree**: `max_depth=[4,8]`, `min_samples_leaf=[25,50]`
- **Best parameters**: `max_depth=8`, `min_samples_leaf=25`

### Ensemble Methods Explored

#### 1. Stacking Regressor
```python
estimators = [
    ("DT", DecisionTreeRegressor(max_depth=8, min_samples_leaf=25)),
    ("LR", LinearRegression())
]
stacking_model = StackingRegressor(estimators=estimators)
```

#### 2. Gradient Boosting
- **Configuration**: `max_depth=2`, `n_estimators=300`
- **Strategy**: Sequential weak learner improvement

#### 3. Random Forest
- **Configuration**: `max_depth=32`, `n_estimators=300`
- **Strategy**: Parallel tree averaging with bootstrap sampling

## 🛠️ Technical Implementation

### Data Preprocessing Pipeline
1. **Feature Selection**: Removed irrelevant columns
2. **Missing Values**: Mean imputation for numerical, mode for categorical
3. **Encoding**: One-hot encoding for categorical variables
4. **Scaling**: Applied where necessary for algorithm requirements

### Model Evaluation Framework
```python
# Consistent evaluation across all models
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
```

### Performance Visualization
- **Bar charts** comparing RMSE and R² across models
- **Side-by-side comparison** for easy interpretation
- **Statistical significance** assessment

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running the Analysis
1. Clone this repository
2. Verify the dataset file exists at: `data_regressors/airbnbData_train.csv`
3. Open `ComparingRegressionModels.ipynb`
4. Run all cells to reproduce the analysis

**Data Requirements**: The project expects the preprocessed Airbnb dataset in the `data_regressors/` folder.

### Project Structure
```
├── ComparingRegressionModels.ipynb    # Main analysis notebook
├── data_regressors/
│   └── airbnbData_train.csv          # Preprocessed Airbnb dataset (REQUIRED)
├── data_*/                           # Other project datasets
│   ├── data_NN/                     # Neural network datasets
│   ├── data_GBDT/                   # Gradient boosting datasets
│   ├── data_RF/                     # Random forest datasets
│   ├── data_clustering/             # Clustering datasets
│   └── ...                          # Additional ML datasets
├── README.md                         # This file
└── requirements.txt                  # Dependencies
```

**Important**: This project specifically uses the dataset in `data_regressors/airbnbData_train.csv`.

## 💡 Key Learning Outcomes

### Machine Learning Concepts
- **Model Selection**: Systematic comparison of algorithms
- **Ensemble Methods**: Bagging, boosting, and stacking techniques
- **Hyperparameter Tuning**: GridSearchCV for optimization
- **Cross-validation**: Proper model evaluation practices

### Business Applications
- **Pricing Strategy**: Data-driven approach to rental pricing
- **Feature Importance**: Understanding factors affecting price
- **Model Deployment**: Considerations for production systems

## 🔮 Future Enhancements

- [ ] **Feature Engineering**: Create new predictive features
- [ ] **Advanced Ensembles**: XGBoost, LightGBM implementation
- [ ] **Deep Learning**: Neural network comparison
- [ ] **Model Interpretability**: SHAP values for feature importance
- [ ] **Production Pipeline**: Model deployment with Flask/FastAPI
- [ ] **Real-time Pricing**: Dynamic pricing model updates

## 📊 Performance Visualization

The project includes comprehensive visualizations:
- **Model comparison bar charts**
- **RMSE vs R² trade-off analysis**
- **Residual plots** for model diagnostics
- **Feature importance** rankings

## 🎯 Business Impact

This analysis provides actionable insights for:
- **Airbnb Hosts**: Optimal pricing strategies
- **Platform Optimization**: Fair pricing recommendations
- **Market Analysis**: Understanding pricing factors
- **Revenue Optimization**: Data-driven pricing decisions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Additional model implementations
- Feature engineering improvements
- Visualization enhancements
- Documentation updates

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **If this project helped you learn about regression models and ensemble methods, please star this repository!**

### 🔗 Connect with Me
- **LinkedIn**: [Your LinkedIn Profile]
- **Portfolio**: [Your Portfolio Website]
- **Email**: [Your Email] 