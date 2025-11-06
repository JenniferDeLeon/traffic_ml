# App 2: Predict Traffic Volume 

# Import necessary libraries
# Import mapie for prediction intervals
from mapie.regression import MapieRegressor # To calculate prediction intervals
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings # Suppress warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Traffic Volume Predictor') 
st.write("Utilize our advanced Machine Learning application to predict traffic volume.")

# Display an image of traffic
st.image('traffic_sidebar.jpg', width = 600)

# Load the pre-trained model from the pickle file
dt_pickle = open('traffic_volume.pickle', 'rb') 
clf = pickle.load(dt_pickle) 
dt_pickle.close()

# Set up the sidebar for user inputs
st.sidebar.image('traffic_sidebar.jpg', caption="Traffic Volume Predictor", width=300)
st.sidebar.subheader("Input Features")
st.sidebar.write("You can either upload your data file or manually enter input features.")

# Input features expected by the model (should match .ipynb)
expected_cols = ['temp','rain_1h','snow_1h','clouds_all','hour',
    'holiday_Christmas Day','holiday_Columbus Day','holiday_Independence Day',
    'holiday_Labor Day','holiday_Martin Luther King Jr Day','holiday_Memorial Day',
    'holiday_New Years Day','holiday_None','holiday_State Fair',
    'holiday_Thanksgiving Day','holiday_Veterans Day','holiday_Washingtons Birthday',
    'weather_main_Clear','weather_main_Clouds','weather_main_Drizzle',
    'weather_main_Fog','weather_main_Haze','weather_main_Mist',
    'weather_main_Rain','weather_main_Smoke','weather_main_Snow',
    'weather_main_Squall','weather_main_Thunderstorm',
    'month_April','month_August','month_December','month_February',
    'month_January','month_July','month_June','month_March','month_May',
    'month_November','month_October','month_September',
    'weekday_Friday','weekday_Monday','weekday_Saturday','weekday_Sunday',
    'weekday_Thursday','weekday_Tuesday','weekday_Wednesday']


# Option 1: Upload CSV
with st.sidebar.expander("Option 1: Upload CSV file", expanded=False):
    st.write("Upload a CSV file containing the traffic details.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"]) # This is where the user uploads the file

    # Sample file
    st.write("### Sample Data Format for Upload")
    sample_df = pd.read_csv("Traffic_Volume.csv") # So the user knows what their csv file should be formatted like

    # I need to change date_time to month, weekday, hour (similar to the training)
    # I used ChatGPT here to help me figure out how to extract month, weekday, hour from date_time
    # Also, how to format the month and weekday to show the full names of the month/day
    sample_df['date_time'] = pd.to_datetime(sample_df['date_time'], format='%m/%d/%y %H:%M')
    sample_df['month'] = sample_df['date_time'].dt.month_name()
    sample_df['weekday'] = sample_df['date_time'].dt.day_name()
    sample_df['hour'] = sample_df['date_time'].dt.hour
    sample_df = sample_df.drop(columns=['date_time', 'traffic_volume']) 
    # Drop target column and original date_time because I made new columns for month, weekday, hour

    st.dataframe(sample_df.head()) # Show sample DF 
    st.warning("⚠️ Ensure your uploaded file has the same column names and data types as shown above.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Feature preprocessing
    X = pd.get_dummies(df, columns=['holiday', 'weather_main', 'month', 'weekday'], drop_first=False)
    # I need dummies for categorical columns

    # Add missing dummy columns
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_cols]  # reorder

    # Store in session_state for use later
    # I used ChatGPT here to help with the syntax of session_state
    st.session_state["uploaded_df"] = df
    st.session_state["X_uploaded"] = X


# Option 2: Manual Input Form
with st.sidebar.expander("Option 2: Fill Out Form", expanded=False): # False so that it is collapsed by default
    st.write("Enter the traffic details manually using the form below.")

    holiday_input = st.selectbox("Holiday", ["None", 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Christmas Day',
                                             'New Years Day', 'Washingtons Birthday', 'Memorial Day', 'Independence Day',
                                             'State Fair', 'Labor Day', 'Martin Luther King Jr Day'])
    temperature = st.number_input("Average temperature in Kelvin", value=281.21)
    rain = st.number_input("Amount in mm of rain that occurred in the hour", value=0.33)
    snow = st.number_input("Amount in mm of snow that occurred in the hour", value=0.00)
    cloud = st.number_input("Percentage of cloud cover", min_value=0, max_value=100, value=49)
    weather_input = st.selectbox("Choose the current weather", ['Clouds', 'Clear', 'Rain', 'Drizzle', 'Mist', 'Haze', 'Fog', 'Thunderstorm',
                                                                'Snow', 'Squall', 'Smoke'])
    month_input = st.selectbox("Choose month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                                                'September', 'October', 'November', 'December'])
    day_input = st.selectbox("Choose day of the week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    hour_input = st.selectbox("Hour of the day (0-23)", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], index=0)

    # Create DataFrame with one row for the model
    input_df = pd.DataFrame({
        'temp': [temperature],
        'rain_1h': [rain],
        'snow_1h': [snow],
        'clouds_all': [cloud],
        'month': [month_input],
        'weekday': [day_input],
        'hour': [hour_input],
        'holiday': [holiday_input],
        'weather_main': [weather_input]})

    # One-hot encode categorical columns
    input_df = pd.get_dummies(input_df, columns=['holiday', 'weather_main'], drop_first=False)

    # Add missing dummy columns
    # For safety, ensure all expected columns are present
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_cols]

    # Button inside the expander
    if st.button("Submit Form Data"):
        st.session_state["form_submitted"] = True
        st.session_state["input_features"] = input_df

# Initialize session state
# I used ChatGPT here to help with using the st.session_state function
if "input_features" not in st.session_state:
    st.session_state["input_features"] = None
if "form_submitted" not in st.session_state:
    st.session_state["form_submitted"] = False

# Load model once
with open('traffic_volume.pickle', 'rb') as f:
    mapie_model = pickle.load(f)

# Tell user the status of their input
if uploaded_file is not None:
    st.success("✅ CSV file uploaded successfully!")
elif st.session_state.get("form_submitted", False):
    st.success("✅ Form data submitted successfully!")
else:
    st.info("ℹ️ Please choose a data input method to proceed.")

# Slider for user input of alpha
alpha = st.slider("Select alpha for prediction interval", min_value=0.01, max_value=0.5, value=0.1)

# I want to show this section as a default even if no user input is given
# I repeated these same lines below after the user inputs data manually
# I want this feature to show up regardless of user input when the app first loads
if not st.session_state["form_submitted"] and uploaded_file is None:
    st.subheader("Predicting Traffic Volume...")
    st.metric(label="Predicted Traffic Volume", value="0")
    st.markdown(f"**Prediction Interval** ({(1-alpha)*100:.1f}%): [0, 664]")

# Manual Form Prediction
# I had a lot of issues with this section, so I used ChatGPT to help me debug and figure out how to properly use session_state to store user input data
if st.session_state["form_submitted"] and st.session_state["input_features"] is not None:

    # Retrieve from session state
    X_input = st.session_state["input_features"]
    y_pred, y_pis = mapie_model.predict(X_input, alpha=alpha)

    prediction = int(y_pred[0]) 
    lower_bound = int(y_pis[0, 0]) 
    upper_bound = int(y_pis[0, 1])

    st.subheader("Predicting Traffic Volume...")
    st.metric(label="Predicted Traffic Volume", value=f"{prediction:,.0f}")
    st.markdown(f"**Prediction Interval** ({(1-alpha)*100:.1f}%): [{lower_bound:,.0f}, {upper_bound:,.0f}]")


# I used ChatGPT to verify the st.session_state and the dataframe indexing syntax for the uploaded file  
elif uploaded_file is not None:

    # Retrieve from session state
    df = st.session_state["uploaded_df"] # Here, I want to show the original df uploaded by the user (after the user uploads)
    X = st.session_state["X_uploaded"] # Here, I want to use the preprocessed features for prediction (after the user uploads)

    # Predict (after the user uploads)
    y_pred, y_pis = mapie_model.predict(X, alpha=alpha)
    df["Predicted Volume"] = y_pred
    df["Lower Limit"] = y_pis[:, 0] # I couldn't remember how indexing worked, so I asked ChatGPT how this works
    df["Upper Limit"] = y_pis[:, 1]

    confidence = (1 - alpha) * 100 

    # These are the columns to display in the results
    display_cols = ["holiday", "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main", "month", "weekday", "hour",
                    "Predicted Volume", "Lower Limit", "Upper Limit"]

    st.subheader(f"Prediction Results with {confidence:.0f}% Prediction Interval")
    st.dataframe(df[display_cols])

# Model Insights Tabs
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", "Predicted Vs. Actual", "Coverage Plot"])

# Tab 1: Feature Importance 
with tab1:
    st.write("Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")

# Tab 2: Histogram of Residuals
with tab2:
    st.write("Histogram of Residuals")
    st.image('all_residuals.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")

# Tab 3: Predicted Vs. Actual
with tab3:
    st.write("Predicted Vs. Actual")
    st.image('predicted_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")

# Tab 4: Coverage Plot
with tab4:
    st.write("Coverage Plot")
    st.image('coverage_plot.svg')
    st.caption("Range of predictions with confidence intervals.")