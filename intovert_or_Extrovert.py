import streamlit as st
import pickle
import pandas as pd

# Load the trained model
try:
    with open(r'C:\pythonProject5\Introvert_or_Extrovert1.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.title("Are You an Introvert or Extrovert?")
st.markdown("Enter your details below to predict your personality type:")

# Input fields
Time_spent_Alone = st.number_input(
    "Time spent alone (hours per day)",
    min_value=0.1,
    max_value=24.0,
    value=5.0,
    step=0.1,
    help="Enter hours between 0.1 and 24"
)
Stage_fear = st.selectbox("Do you have stage fear?", options=["Yes", "No"])
Social_event_attendance = st.number_input(
    "How many social events do you attend per month?",
    min_value=0.0,
    value=3.0,
    step=0.1,
    help="Enter a number greater than 0"
)
Going_outside = st.number_input(
    "How much do you enjoy going outside? (0-10 scale)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1,
    help="Enter a value between 0 and 10"
)
Drained_after_socializing = st.selectbox("Do you feel drained after socializing?", options=["Yes", "No"])
Friends_circle_size = st.number_input(
    "How many close friends do you have? (0-10)",
    min_value=0.0,
    max_value=10.0,
    value=4.0,
    step=0.1,
    help="Enter a number between 0 and 10"
)
Post_frequency = st.number_input(
    "How many social media posts do you make per week? (0-10)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1,
    help="Enter a number between 0 and 10"
)

if st.button("Predict Personality"):
    try:
        # Prepare input data as a DataFrame
        input_data = pd.DataFrame([{
            'Time_spent_Alone': float(Time_spent_Alone),
            'Stage_fear': Stage_fear,  # Try strings first
            'Social_event_attendance': float(Social_event_attendance),
            'Going_outside': float(Going_outside),
            'Drained_after_socializing': Drained_after_socializing,
            'Friends_circle_size': float(Friends_circle_size),
            'Post_frequency': float(Post_frequency)
        }])

        # Alternative encoding: uncomment if model expects 1/0
        # input_data['Stage_fear'] = 1 if Stage_fear == "Yes" else 0
        # input_data['Drained_after_socializing'] = 1 if Drained_after_socializing == "Yes" else 0

        # Make prediction
        prediction = model.predict(input_data)[0]
        label = "Introvert" if str(prediction) == "1" else "Extrovert"  # Adjust based on your model
        st.success(f"Your Personality is: **{label}**")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Input data:", input_data.to_dict())
