# Dynamic Pricing and Demand Forecasting 
## **Project Overview**  
The goal of this project is to build a dynamic pricing model that adjusts prices in real-time based on demand, competition, and other factors. Dynamic pricing helps businesses optimize pricing strategies to maximize revenue and improve customer satisfaction. This project demonstrates the use of **reinforcement learning**, **time series forecasting**, **demand estimation**, **pricing strategies**, and **A/B testing** techniques.

---

## **Components**  

### **1. Data Collection and Preprocessing**  
- **Objective**: The first step is to gather, clean, and preprocess the Walmart sales data and external datasets such as competitor pricing and weather data.  
- **Key Tasks**:
    - **Raw Data Loading**: Loaded datasets for Walmart sales, competitor prices, and weather data.
    - **External Data Integration**: Combined Walmart sales data with competitor pricing and weather data.
    - **Data Cleaning**: Handled missing values, removed duplicates, and filtered outliers.
    - **Feature Engineering**: Created additional features such as lagged sales, moving averages, and holiday effects.
    - **Data Normalization**: Applied normalization techniques like StandardScaler to scale numeric features.
    - **Data Splitting**: Split the dataset into training and testing sets using time-series split.

---

### **2. Exploratory Data Analysis (EDA)**  
- **Objective**: Analyze the dataset to identify patterns, trends, and correlations between various factors.  
- **Key Tasks**:
    - **Visualizations**: Created histograms, scatter plots, and heatmaps to explore the distribution of features.
    - **Trend and Seasonality Analysis**: Decomposed time-series data to extract trend, seasonality, and residuals using `seasonal_decompose`.
    - **Statistical Analysis**: Performed statistical tests (e.g., t-tests) to check for significant differences (e.g., holidays’ impact on sales).
    - **Key Results**:
        - **T-test** for Holiday Impact: T-statistic = `5.79`, P-value = `2.67e-08` (statistically significant).

---

### **3. Time Series Forecasting**  
- **Objective**: Build and optimize models to forecast future sales and demand.  
- **Key Tasks**:
    - **Model Selection**: Compared multiple forecasting models like ARIMA, SARIMA, Facebook Prophet, LSTM, and XGBoost.
    - **XGBoost Model**: Selected XGBoost due to its strong performance in forecasting with multivariate data.
    - **Hyperparameter Tuning**: Performed tuning using GridSearchCV to optimize the XGBoost model.
    - **Evaluation**: Used MAE to evaluate models' predictive accuracy:
        - **Before Tuning**: MAE = `70,020.03`
        - **After Tuning**: MAE = `68827.68`.

---

### **4. Demand Estimation**  
- **Objective**: Estimate price elasticity and demand using the trained XGBoost model.  
- **Key Tasks**:
    - **Price Elasticity**: Calculated price elasticity of demand (PED) using log-log regression.
    - **Elasticity Value**: `0.0075`, indicating low price sensitivity (inelastic demand).
    - **Modeling**: Estimated how changes in prices would impact sales using the trained XGBoost model.
    - **Results**:
        - Low elasticity (0.088) suggests that demand is not highly sensitive to price changes.

---

### **5. Reinforcement Learning for Pricing**  
- **Objective**: Apply reinforcement learning to dynamically adjust prices based on demand and external factors.  
- **Key Tasks**:
    - **Q-Learning**: Developed a Q-learning agent that learns the optimal pricing strategy through exploration and exploitation.
    - **Simulation**: Implemented a simulation environment where the agent adjusts prices to maximize revenue.
    - **Results**:
        - After 1000 training episodes, the total reward was `10,188,500`, indicating successful price optimization.
  
---

### **6. Pricing Strategies**  
- **Objective**: Evaluate the impact of different pricing strategies on revenue and customer satisfaction.  
- **Key Tasks**:
    - **Sensitivity Analysis**: Examined how price changes impact demand and revenue.
    - **Customer Segmentation**: Grouped customers into high-value and low-value segments for targeted pricing strategies.
    - **Key Insights**:
        - Price increases of around 2% led to a slight drop in demand but a net increase in revenue.
        - Found an optimal price point where revenue is maximized.

---

### **7. A/B Testing**  
- **Objective**: Perform A/B testing to compare the effect of different pricing strategies and determine which one performs better in terms of revenue.  
- **Key Tasks**:
    - **T-Tests**: Performed t-tests to check the statistical significance between the two groups (e.g., price changes).
    - **Results**: The difference between the two groups was statistically significant, with a **T-statistic = 5.79** and a **P-value = 2.67e-08**.

---

## **Technologies and Libraries Used**  
- **Python Libraries**:
    - `pandas`: Data loading, cleaning, and manipulation.
    - `numpy`: Numerical operations.
    - `matplotlib` & `seaborn`: Data visualization.
    - `sklearn`: For model building, hyperparameter tuning, and evaluation.
    - `xgboost`: XGBoost model for time-series forecasting.
    - `statsmodels`: Statistical modeling and time-series decomposition.
    - `scipy`: Statistical tests.

---

## **Conclusion and Future Work**  
- The project successfully built a dynamic pricing model using **reinforcement learning** and **XGBoost** for demand forecasting and price optimization.
- **Key Outcomes**:
    - Developed pricing strategies that optimize revenue while balancing demand.
    - Estimated price elasticity and demand sensitivity to inform pricing decisions.
- **Future Work**:
    - Explore additional reinforcement learning algorithms such as Deep Q-Learning.
    - Incorporate more advanced demand estimation models like GLMs or Poisson regression.

---

## **Files Included**
- **dynamic_pricing_model.ipynb**: Jupyter Notebook containing the entire code implementation, from data preprocessing to model building, reinforcement learning, and A/B testing.
- **dataset.csv**: The dataset used for training and testing the models, containing historical sales data, competitor prices, and external factors.


---
