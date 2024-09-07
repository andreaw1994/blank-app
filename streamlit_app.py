import streamlit as st
#import pandas as pd
#import matplotlib.pyplot as plt
#import ipywidgets as widgets
#from IPython.display import display

st.title("STE1 Manufacturing Analyzer")

st.write(
    "This app allows you to interactively visualize and analyze key performance metrics and error patterns of a STE1 production run."
)

st.write(
    "TO DO: Converter that takes error, error code and continuous data file in XLSX format as input, and outputs MasterCSVs."
)

import pandas as pd

# Use Streamlit's file uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    st.write(data)

import datetime

data["datetime"] = pd.to_datetime(data["datetime"])

data["derivative_good_objects"] = data[data["good_objects"].notna()]["good_objects"].diff() / data[data["good_objects"].notna()]["datetime"].diff().dt.total_seconds()
data["derivative_good_objects"] = data["derivative_good_objects"].fillna(0)

data
