import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from functools import reduce

#import ipywidgets as widgets
#from IPython.display import display

st.title("STE1 Manufacturing Analyzer")

st.write(
    "This app allows you to interactively visualize and analyze key performance metrics and error patterns of a STE1 production run."
)

st.write(
    "TO DO: Converter that takes error, error code and continuous data file in XLSX format as input, and outputs MasterCSVs."
)

# Use Streamlit's file uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    st.write(data)

data["datetime"] = pd.to_datetime(data["datetime"])

data["derivative_good_objects"] = data[data["good_objects"].notna()]["good_objects"].diff() / data[data["good_objects"].notna()]["datetime"].diff().dt.total_seconds()
data["derivative_good_objects"] = data["derivative_good_objects"].fillna(0)

data

data["zero_time"] = datetime.timedelta()
prev_der = None
first_time = data.loc[data["good_objects"].first_valid_index(), "datetime"]
start_from = data.loc[data["good_objects"].first_valid_index(), "good_objects"]

for i, row in enumerate(data.itertuples()):
    # Transition from 0 to positive derivative
    if prev_der is not None and row.derivative_good_objects > 0 and prev_der == 0 and row.good_objects != start_from:
        zero_time = (row.datetime - first_time)
        data.at[i, "zero_time"] = zero_time

    # Transition from positive to 0 derivative
    elif prev_der is not None and row.derivative_good_objects == 0 and prev_der > 0:
        first_time = row.datetime  # Start the zero time interval

    # Update the previous derivative for the next iteration
    prev_der = row.derivative_good_objects

sorted_times = data[data["zero_time"] > datetime.timedelta()]["zero_time"].sort_values(ascending = False)


fig, ax1 = plt.subplots()
fig.set_size_inches(28.5, 15.5)

ax1.scatter(data[data["zero_time"] > datetime.timedelta()]["datetime"], data[data["zero_time"] > datetime.timedelta()]["zero_time"].dt.total_seconds())

st.pyplot(fig)




# Change the quantile value here, affects the lower limit for the pause duration.
q99 = data[data["zero_time"] != datetime.timedelta()]["zero_time"].quantile(0.99)
q99_df = data[data["zero_time"].dt.total_seconds() >= q99.total_seconds()]
data.loc[data["zero_time"] >= q99, "99q"] = True
data.fillna({"99q" : False}, inplace = True)

errors = []
indices = []
good_counter = []
for i in (q99_df["datetime"] - q99_df["zero_time"]):
    # This controls the amount of seconds before and after the start of the pause that we look for errors in.
    error_code = data[(data["datetime"] >= (i - datetime.timedelta(seconds = 10))) & (data["datetime"] <= (i + datetime.timedelta(seconds = 10)))]["description"]
    errors.extend(error_code[error_code.notna()].values)
    indices.extend(error_code[error_code.notna()].index.values.astype(int))

pd.DataFrame(errors).value_counts()

fig, ax1 = plt.subplots()
fig.set_size_inches(28.5, 15.5)

ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', -40))  # Move the 3rd axis outward
ax1.plot(data[data["good_objects"] >= 5]["datetime"], data[data["good_objects"] >= 5]["good_objects"].ffill(), c='b')
ax1.plot(data[data["good_objects"] >= 5]["datetime"], data[data["good_objects"] >= 5]["inactive_time"].ffill(), c='g')
ax2.scatter(data.iloc[indices]["datetime"],
            data.iloc[indices]["description"], c = 'r')
ax3.stem(q99_df["datetime"] - q99_df["zero_time"], q99_df["zero_time"])
# ax2.scatter(data[(data["code"] == 1) & (data["datetime"] >= "2024-05-15 09:00:00") & (data["derivative"] == 0)]["datetime"],
#             data[(data["code"] == 1) & (data["datetime"] >= "2024-05-15 09:00:00") & (data["derivative"] == 0)]["error_code"], c = "orange")
# ax1.bar(x = q99_df["datetime"] - q99_df["zero_time"],
#         height = data.iloc[data[data["datetime"].isin(start)].index.values.astype(int) - 1]["good_objects"].dropna(),
#         width = data[(data["datetime"] >= "2024-05-15 09:00:00") & data["99q"]]["zero_time"], align = "edge")

