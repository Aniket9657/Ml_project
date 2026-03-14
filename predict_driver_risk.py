import joblib
import numpy as np

model = joblib.load("driver_risk_model.pkl")

# example driver data
sample_driver = np.array([[30,10,1,0,2,1,2,95,110,7,3.5,2,10,12]])

prediction = model.predict(sample_driver)

risk_levels = ["Safe", "Moderate Risk", "High Risk"]

print("Predicted Driver Risk:", risk_levels[prediction[0]])