import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("data/customers.csv", sep="\t")
print(df.head())
print(df.columns)

# Select features
X = df.iloc[:, [3, 4]]

# Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Plot clusters
plt.scatter(
    X['Annual Income (k$)'],
    X['Spending Score (1-100)'],
    c=df['Cluster']
)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")

# Save output
plt.savefig("outputs/clusters.png")
plt.show()
