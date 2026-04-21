import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Load data
df = pd.read_csv("data/data.csv")

# Model
model = IsolationForest(contamination=0.15, random_state=42)
df['anomaly'] = model.fit_predict(df[['value']])

# Plot
plt.scatter(range(len(df)), df['value'], c=df['anomaly'])
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Anomaly Detection")

# Save output
plt.savefig("outputs/anomalies.png")
plt.show()

# Print anomalies
print("Detected anomalies:")
print(df[df['anomaly'] == -1])