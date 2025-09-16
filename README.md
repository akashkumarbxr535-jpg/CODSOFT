# CODSOFT
#  Sales Prediction using Machine Learning

This project predicts product **Sales** based on **advertising expenditure** across multiple channels (`TV`, `Radio`, `Newspaper`).  
It demonstrates how businesses can leverage **machine learning** to optimize marketing strategies and maximize ROI.

##  Project Workflow

1. **Dataset**  
   - Source: `advertising.csv` (200 rows × 4 columns)  
   - Features: `TV`, `Radio`, `Newspaper`  
   - Target: `Sales`  

2. **Data Preprocessing**  
   - Handled missing values (none in this dataset).  
   - Selected relevant numeric features.  

3. **Model Building**  
   - **Linear Regression** (baseline)  
   - **Ridge Regression** (regularized linear model)  

4. **Model Evaluation**  
   - Metrics: **RMSE**, **MAE**, **R²**  
   - Visualization: Actual vs Predicted plot  

5. **Feature Importance**  
   - Identified how much each advertising channel contributes to sales.  

6. **Predictions**  
   - Generated and saved predictions in `sales_predictions.csv`.

##  Results

*Linear Regression* achieved an **R² ≈ 0.91**, showing strong predictive ability.  
*TV and Radio* were the most influential features for driving sales, while `Newspaper` had little impact.  
