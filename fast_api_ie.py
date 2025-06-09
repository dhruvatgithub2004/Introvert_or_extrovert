from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException

# import the ml model
try:
    with open(r'C:\pythonProject5\Introvert_or_Extrovert1.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

app = FastAPI()


# Pydantic model for input validation
class UserInput(BaseModel):
    Time_spent_Alone: Annotated[float, Field(..., gt=0, lt=24, description='Time spent alone')]
    Stage_fear: Annotated[str, Field(..., description='Does user have stage fear?')]
    Social_event_attendance: Annotated[
        float, Field(..., gt=0, description='How many social events does the user attend?')]
    Going_outside: Annotated[float, Field(..., description='Does user like going outside?')]
    Drained_after_socializing: Annotated[str, Field(..., description='Does user get tired after socializing?')]
    Friends_circle_size: Annotated[float, Field(..., lt=10, description='Number of friends the user has')]
    Post_frequency: Annotated[float, Field(..., lt=10, description='Number of posts user does')]


@app.post('/predict')
async def predict_personality(data: UserInput):
    try:
        # Create DataFrame from input
        input_df = pd.DataFrame([{
            'Time_spent_Alone': data.Time_spent_Alone,
            'Stage_fear': data.Stage_fear,
            'Social_event_attendance': data.Social_event_attendance,
            'Going_outside': data.Going_outside,
            'Drained_after_socializing': data.Drained_after_socializing,
            'Friends_circle_size': data.Friends_circle_size,
            'Post_frequency': data.Post_frequency
        }])

        # Debug: Print DataFrame
        print("Input DataFrame:")
        print(input_df)
        print(input_df.dtypes)

        # Make prediction
        prediction = model.predict(input_df)[0]

        return JSONResponse(status_code=200, content={'predicted_category': str(prediction)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")