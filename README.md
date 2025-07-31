# California Housing Price Predictor

This is a Streamlit web application that predicts the median house value in California based on user input features.

## Features

- Interactive sliders for 8 key housing features:
  - Median income in tens of thousands of USD (MedInc)
  - Median house age in years (HouseAge)
  - Average number of rooms per household (AveRooms)
  - Average number of bedrooms per household (AveBedrms)
  - Block group population (Population)
  - Average number of household members (AveOccup)
  - Block group latitude (Latitude)
  - Block group longitude (Longitude)

- Responsive card layout displaying feature descriptions and sliders.

- Prediction of median house value based on input features using a pre-trained Random Forest model.

- Display of model performance metrics:
  - R² Score
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)

- Visualization of feature importance.

- Training and test graphs showing:
  - Train vs Actual values
  - Test vs Actual values
  - Residuals plot

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the pre-trained model file `housing_model.pkl` and `feature_importance.png` image are in the project directory.

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Open the URL provided by Streamlit in your browser to interact with the app.

## Notes

- The app uses cached loading for the model and data to improve performance.

- If the feature importance image is missing, please run the training script to generate it.

- The app includes interactive plots for model evaluation.

## Files

- `app.py`: Main Streamlit application code.

- `requirements.txt`: Python dependencies.

- `housing_model.pkl`: Pre-trained Random Forest model file.

- `feature_importance.png`: Image showing feature importance.

- `train_model.py`: Script to train the model and generate feature importance.

## License



This project is licensed under the MIT License.

---

Created by Srinivas R  
Email: srinivassrini14592@gmail.com
