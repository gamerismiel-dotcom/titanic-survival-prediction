# Titanic Survival Prediction 🚢

A Machine Learning project built with Python and Scikit-Learn. This model predicts whether a passenger survived the Titanic disaster based on features like age, passenger class, and sex.

## Technologies Used
*   **Python** (Core logic and data processing)
*   **Pandas** (Data cleaning and feature engineering)
*   **Scikit-Learn** (Decision Tree Classifier and Train/Test split)

## How it Works
1.  **Data Loading:** Creates a dataset of 10 passengers.
2.  **Data Cleaning:** Fills missing ages with the average age using `fillna()`.
3.  **Encoding:** Converts categorical text ('male'/'female') into numerical data (0/1) using `get_dummies()`.
4.  **Training:** Splits the data (80/20) and trains a Decision Tree Classifier.
5.  **Prediction:** Predicts survival for a new 25-year-old male in 3rd class.

## How to Run
1. Clone this repository.
2. Ensure you have `pandas` and `scikit-learn` installed (`pip install pandas scikit-learn`).
3. Run `python titanic.py`
