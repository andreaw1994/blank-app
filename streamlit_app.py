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
