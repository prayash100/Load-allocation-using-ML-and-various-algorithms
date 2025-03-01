import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("corrected_gujarat_load_demand_2024.csv")  # Replace with your file path
df["Index"] = range(len(df))
df['Load_Demand_MW'] = df['Load_Demand_MW'].round(2)
df_w = df.iloc[20000:40000]  # First 300 rows


# Plot the data (Assuming the CSV has 'Time' and 'Value' columns)
plt.figure(figsize=(10, 8))
plt.plot(df_w["Index"], df_w["Load_Demand_MW"], marker='None', linestyle='-')

# Labels and title
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("CSV Data Plot")

# Show the plot
plt.show()
