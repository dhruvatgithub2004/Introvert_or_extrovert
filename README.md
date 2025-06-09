# Personality Prediction (Introvert/Extrovert) using Machine Learning

## Project Description

This project focuses on building a machine learning model to classify individuals as either Introvert or Extrovert based on behavioural attributes derived from survey-like data. The process involves data loading, cleaning, exploratory data analysis (EDA), feature engineering, training various classification models, evaluating their performance, and saving the best-performing model.

## Data

The analysis uses a dataset contained in the file `Introvert or extrovert data.csv`.
The raw dataset originally included 2900 entries. After cleaning steps, specifically dropping rows with missing values, the final dataset used for modelling comprises 2477 entries.
The key features in the dataset include:
*   `Time_spent_Alone`
*   `Stage_fear`
*   `Social_event_attendance`
*   `Going_outside`
*   `Drained_after_socializing'
*   `Friends_circle_size`
*   `Post_frequency`
*   `Personality` (the target variable - Introvert or Extrovert) 

## Libraries Used

The project was developed using Python and the following libraries:
*   `pandas`: For data loading, manipulation, and cleaning.
*   `seaborn` and `matplotlib.pyplot`: For data visualization, including plotting distributions and confusion matrices .
*   `numpy`: Used implicitly for numerical operations, specifically for calculating the mean of cross-validation scores.
*   `sklearn`: A comprehensive machine learning library used for:
    *   Preprocessing (`OneHotEncoder`, `LabelEncoder`, `ColumnTransformer`).
    *   Model selection (`RandomForestClassifier`, `SVC`, `LogisticRegression`, `KNeighborsClassifier`).
    *   Model evaluation (`train_test_split`, `accuracy_score`, `classification_report`, `recall_score`, `confusion_matrix`, `f1_score`, `cross_val_score`).
    *   Building a complete workflow (`Pipeline`).
*   `pickle`: For saving the trained machine learning pipeline object to a file.

## Project Steps

1.  **Data Loading and Initial Inspection:**
    *   The dataset was loaded using pandas.
    *   Initial checks were performed to understand the data types, non-null counts, and summary statistics. The presence of non-null counts less than the total entries indicated missing values in several columns.

2.  **Data Cleaning:**
    *   Rows containing any missing values were removed using the `dropna()` function. This reduced the dataset size from 2900 to 2477 entries.

3.  **Exploratory Data Analysis (EDA):**
    *   Descriptive statistics were generated for the numerical columns.
    *   Box plots were used to visualize the distribution and identify potential outliers in numerical features after dropping missing values.
    *   Key behavioural attributes were analysed by grouping the data by the `Personality` type to find average differences:
        *   Extroverts, on average, spent less `Time_spent_Alone` (2.13) compared to Introverts (7.04).
        *   Extroverts showed higher average `Social_event_attendance` (5.95) than Introverts (1.80).
        *   Extroverts had a significantly larger average `Friends_circle_size` (9.10) compared to Introverts (3.20).
        *   Extroverts posted more frequently (`Post_frequency`) on average (5.58) than Introverts (1.40).

4.  **Preprocessing and Feature Engineering:**
    *   Categorical features (`Stage_fear`, `Drained_after_socializing`) and numerical features were identified.
    *   A `ColumnTransformer` was set up to apply `OneHotEncoder` to the specified categorical features while passing through the numerical ones. `handle_unknown='ignore'` was used in the encoder.
    *   The target variable, `Personality`, was encoded into numerical form (likely 0 and 1) using `LabelEncoder`.

5.  **Model Training and Evaluation:**
    *   The cleaned and preprocessed data was split into training (80%) and testing (20%) sets. Note that a `random_state=0` was used initially, but `random_state=1` was used for the split when the final pipeline was created and saved.
    *   Several classification models (`RandomForestClassifier`, `SVC`, `LogisticRegression`, `KNeighborsClassifier`) were evaluated.
    *   A function `model_metrics` was used to automate the evaluation process, which included training the model on the training data, making predictions on the test data, and reporting performance metrics such as:
        *   Average cross-validation accuracy (calculated on the test data folds within the function).
        *   F1-score on the test set.
        *   Classification Report (precision, recall, f1-score, support) on the test set.
        *   Confusion Matrix (visualised using a heatmap) on the test set.
        *   The same metrics were also reported for the training data.

6.  **Model Selection and Saving:**
    *   Based on the evaluation metrics (details of which model performed best can be added here if the source explicitly stated a preference, otherwise mention they all performed well), a `RandomForestClassifier` was chosen as part of the final pipeline.
    *   A `Pipeline` object combining the preprocessing steps (`ColumnTransformer`) and the chosen model (`RandomForestClassifier`) was created and trained on the training data.
    *   This trained pipeline object was saved to a file named `Introvert_or_Extrovert1.pkl` using `pickle`.

## Key Findings

*   Distinct differences in behavioural traits (Time spent alone, Social event attendance, Friends circle size, Post frequency) are observed between Introverts and Extroverts in the dataset, aligning with typical understanding of these personality types.
*   Machine learning models trained on this data, including Logistic Regression, SVC, K-Nearest Neighbors, and Random Forest, demonstrated high predictive performance, with testing accuracies generally above 92%.

## FastAPI Integration
The saved model pipeline (`Introvert_or_Extrovert1.pkl`) can be loaded into a FastAPI application. This allows the trained model to be exposed as a web API endpoint, accepting input data (corresponding to the features used for training) and returning personality predictions (Introvert/Extrovert). This is a common way to deploy machine learning models for practical use.

## Getting Started

To run this project locally, you will need:
*   Python installed.
*   The dataset file `Introvert or extrovert data.csv`.
*   The saved model file `Introvert_or_Extrovert1.pkl` (if you only want to use the trained model).
*   The required Python libraries. You can install them using pip:
    ```bash
    pip install pandas seaborn matplotlib scikit-learn numpy scipy uvicorn fastapi python-multipart python-pickle
    ```
    *(Note: `uvicorn`, `fastapi`, `python-multipart` are needed for the FastAPI part, which is not in the source PDF but mentioned by the user. `scipy` and `python-pickle` are used in the analysis and loading).*

The analysis steps are typically executed sequentially in a script or Jupyter Notebook. The saved `.pkl` file contains the complete pipeline ready for making new predictions. To run the FastAPI application, you would need a separate Python script that loads this `.pkl` file and defines the API endpoints.
