import joblib
import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(base_dir, "house_rent_pipeline.pkl")

pipeline = joblib.load(pipeline_path)

new_house = pd.DataFrame({
    "BHK": [2],
    "Size": [1200],
    "Area Type": ["Super Area"],
    "City": ["Delhi"],
    "Furnishing Status": ["Semi-Furnished"],
    "Tenant Preferred": ["Family"],
    "Bathroom": [2],
    "Point of Contact": ["Contact Owner"]
})

prediction = pipeline.predict(new_house)
print("Predicted Rent:", float(prediction[0]))