import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("Are You an Introvert or Extrovert?")
st.markdown("Enter your details below to predict your personality type:")

# Input fields with constraints matching FastAPI UserInput model
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
    input_data = {
        "Time_spent_Alone": Time_spent_Alone,
        "Stage_fear": Stage_fear,
        "Social_event_attendance": Social_event_attendance,
        "Going_outside": Going_outside,
        "Drained_after_socializing": Drained_after_socializing,
        "Friends_circle_size": Friends_circle_size,
        "Post_frequency": Post_frequency
    }

    try:
        response = requests.post(API_URL, json=input_data)
        response.raise_for_status()  # Raises exception for non-200 status codes
        result = response.json()

        # Check for predicted_category in response
        if "predicted_category" in result:
            prediction = result["predicted_category"]
            # Map numeric prediction to label (adjust based on your model)
            label = "Introvert" if prediction == "1" else "Extrovert"
            st.success(f"Your Personality is: **{label}**")
        else:
            st.error(f"Unexpected response format: {result}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the FastAPI server. Make sure it's running at http://127.0.0.1:8000.")
    except requests.exceptions.HTTPError as http_err:
        st.error(f"API Error: {http_err}")
        if response.text:
            st.write(f"Details: {response.text}")
    except Exception as e:
        st.error(f"Error: {str(e)}")