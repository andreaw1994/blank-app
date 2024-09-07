import streamlit as st
#import pandas as pd
#import matplotlib.pyplot as plt
#import ipywidgets as widgets
#from IPython.display import display

st.title("STE1 Manufacturing Analyzer")
st.write(
    "This app allows you to interactively visualize and analyze key performance metrics and error patterns of a STE1 production run."
)

import pandas as pd

# Use Streamlit's file uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    st.write(data)
