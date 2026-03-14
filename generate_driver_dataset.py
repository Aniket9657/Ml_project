import pandas as pd
import numpy as np

# number of rows
n = 5000

np.random.seed(42)

car_types = ["SUV", "Sedan", "Hatchback", "Truck"]
localities = ["Urban", "Semi-Urban", "Rural"]
professions = ["Engineer", "Driver", "Business", "Student", "Teacher"]

data = {
    "driver_age": np.random.randint(18, 65, n),
    "experience_years": np.random.randint(0, 40, n),
    "car_type": np.random.choice(car_types, n),
    "locality": np.random.choice(localities, n),
    "profession": np.random.choice(professions, n),
    "previous_accidents": np.random.randint(0, 5, n),
    "traffic_violations": np.random.randint(0, 10, n),

    # driving pattern first 1000m
    "avg_speed": np.random.normal(85, 20, n).clip(40, 140),
    "max_speed": np.random.normal(100, 25, n).clip(50, 160),
    "lane_changes": np.random.randint(0, 15, n),
    "acceleration_rate": np.random.uniform(1.0, 6.0, n),
    "harsh_brakes": np.random.randint(0, 6, n),
    "following_distance": np.random.uniform(5, 30, n),
    "speed_variation": np.random.uniform(1, 20, n)
}

df = pd.DataFrame(data)

# create risk label using simple rule logic
conditions = [
    (df["avg_speed"] > 110) | (df["lane_changes"] > 10) | (df["harsh_brakes"] > 3),
    (df["avg_speed"] > 90) | (df["lane_changes"] > 6)
]

choices = [2, 1]

df["risk_label"] = np.select(conditions, choices, default=0)

# save dataset
df.to_csv("driver_risk_dataset_5000.csv", index=False)

print("Dataset generated successfully with 5000 rows")