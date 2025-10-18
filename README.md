# Predicting House Prices with ElasticNet Regression 

## About This Project

This is my first machine learning project where I built a model to predict house prices using the famous **Ames Housing dataset**.

The main goal was to go through all the steps of building a model: cleaning the data, preparing it, training a model, and checking how well it works. I chose the **ElasticNet regression** model and used **GridSearchCV** to find the best settings for it.

This repository also includes several practice notebooks where I learned important concepts like:
* Splitting data into training, validation, and test sets (`Train_Validation_Test.ipynb`).
* Using Cross-Validation to evaluate model performance (`Cross_Validation.ipynb`, `Cross_Val_Score.ipynb`).
* Using Grid Search to find the best model parameters (`Grid_Search.ipynb`).

---

## How I Did It

The project involved several key stages:

### 1. Finding and Removing Outliers (`Outliers.ipynb`)
First, I looked at the data visually to find any unusual data points (outliers) that might mess up the model.
* I used scatter plots to see the relationship between features like `OverallQual` (overall quality) and `GrLivArea` (living area size) and the `SalePrice`.
* I found and removed a few houses that had a very large area but surprisingly low prices, as these could confuse a linear model.

### 2. Cleaning the Data (`Missing_Data_L.ipynb`)
Next, I focused on handling missing information:
* **Checked for Missing Values:** I looked at all columns to see what percentage of data was missing.
* **Filled Missing Values (Imputation):**
    * For numbers like `MasVnrArea` (masonry veneer area) or `GarageYrBlt` (garage year built), I filled missing spots with **zero** or the **average** value where it made sense.
    * For categories where missing meant "doesn't have" (like no basement or no garage), I filled the gaps with the word **'None'**.
    * For `LotFrontage` (length of street connected to property), I filled missing values using the **average frontage for that specific neighborhood**, which is more accurate than using the overall average.
* **Removed Columns:** Columns with too much missing data (over 80%, like `Fence`, `Alley`, `MiscFeature`, `PoolQC`) were removed completely.

### 3. Preparing Data for the Model (Feature Engineering)
* **Changed Data Type:** The `MSSubClass` column (building type code) looked like a number but is really a category, so I changed it to text.
* **One-Hot Encoding:** I converted all text columns (categories) into numbers using `pd.get_dummies()`. This created many new columns (making a total of **245** features).
* **Split the Data:** I divided the data into a training set (90% for teaching the model) and a test set (10% for final evaluation).
* **Scaled the Features:** I scaled all the number columns using `StandardScaler` so they were all on a similar scale. I only used the training data to figure out how to scale, to avoid leaking information from the test set.

### 4. Training the Model (`Linear_Regression_Project.ipynb`)
* **Model Choice:** I used `ElasticNet`, a type of linear regression good for datasets with many features.
* **Finding Best Settings:** I used `GridSearchCV` with 10-fold cross-validation (`cv=10`) to automatically test different settings (`alpha` and `l1_ratio`) and find the best ones for the ElasticNet model.

---

## Results

The `GridSearchCV` found the best settings to be:
* **`alpha`**: 100
* **`l1_ratio`**: 1.0

> **Interesting Finding:** An `l1_ratio` of 1.0 means the best model was actually a pure **Lasso Regression**. Lasso is known for doing **feature selection** – it automatically sets the importance of less useful features to zero. This suggests that out of the 245 features, the model decided many weren't needed to predict the price well.

### How well did the model do on the test data?
* **RMSE (Root Mean Squared Error):** **~$23,759**
    * This means, on average, the model's price prediction is off by about $23,759. Since the average house price in the dataset is around $180,000, this is about a **13% error**, which is a solid starting point for a linear model.
* **MAE (Mean Absolute Error):** **~$17,083**
    * This shows the average absolute difference between the predicted price and the actual price.

---

## Technology Used
* Python 3
* Pandas & NumPy (for data handling)
* Matplotlib & Seaborn (for plotting)
* Scikit-learn (for splitting data, scaling, modeling, and grid search)
* Jupyter Notebook

---

## How to Run This Project

1.  **Clone the repository** to your computer:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    ```
2.  Go into the project folder.
3.  **Install the necessary libraries:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn jupyter
    ```
4.  **Run the Jupyter Notebooks** in this order:
    * Data cleaning and preparation:
        1.  `Outliers.ipynb`
        2.  `Missing_Data_L.ipynb`
    * Final model training and evaluation:
        3.  `Linear_Regression_Project.ipynb`
    * The other notebooks (`Grid_Search.ipynb`, `Cross_Validation.ipynb`, etc.) are practice notebooks showing the learning steps and can be run independently.
b78a725436964c564f18257424301de3cc7e5ea7
