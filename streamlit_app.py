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
    "This app allows you to interactively visualize and analyze key performance metrics and error patterns in STE1 production run."
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

st.pyplot(fig)

#--------------------------------
#Lukas script:

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Part 2: Upload CSV Files
st.write("### Step 1: Upload Your CSV Files")
uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

# Part 3: Read and Process Uploaded Files
def process_csv(file, column_index):
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv(file)

    # Copy the specified column to a new column at the end
    data['good_copy'] = data.iloc[:, column_index]  # Assuming 'D' is the column to copy

    # Fill any blanks in the new column 'good_copy' with the previous value using ffill()
    data['good_copy'] = data['good_copy'].ffill()

    # Calculate the difference between the current value and the previous value in 'good_copy'
    data['difference'] = data['good_copy'].diff()

    return data

if uploaded_files:
    # Part 4: Process and store each CSV file in a dictionary, using the original filename
    data_dict = {}
    for file in uploaded_files:
        file_name = file.name  # Get the original filename
        data_dict[file_name] = process_csv(file, 3)

    # Show a preview of the first uploaded file
    st.write("### Step 2: Data Preview")
    first_file = uploaded_files[0].name
    st.write(f"Preview of the first uploaded file: {first_file}")
    st.dataframe(data_dict[first_file].head())

    # Part 5: Plotting Function
    def plot_data(data, dataset_name, num_to_plot):
        try:
            # Filter out the rows where 'error_code' is 1705
            filtered_data = data[data['error_code'] != 1705]

            # Count the occurrences of each unique error message in the 'description' column
            error_counts = filtered_data['description'].value_counts()

            # Limit the number of error messages to plot
            error_counts = error_counts.head(num_to_plot)

            # Plot the occurrences using a bar chart
            plt.figure(figsize=(14, 8))  # Increase figure size for better readability
            error_counts.plot(kind='bar', color='#3498db', edgecolor='black')  # Add edge color for clarity

            plt.title(f'Occurrences of Each Error Message in {dataset_name} (Excluding error_code 1705)', fontsize=16)
            plt.xlabel('Error Message', fontsize=14)
            plt.ylabel('Number of Occurrences', fontsize=14)

            # Wrap x-axis labels to improve readability
            plt.xticks(rotation=45, ha='right', fontsize=12)

            # Add horizontal gridlines for better readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()  # Adjust layout to fit labels
            st.pyplot(plt)
            plt.close()  # Close the plot after rendering to avoid memory issues

        except Exception as e:
            st.error(f"An error occurred while plotting: {e}")

    # Part 6: User Interface for Plotting
    st.write("### Step 3: Plot Data")

    # Dynamically create a selection box with the original filenames
    dataset_name = st.selectbox("Select Dataset", list(data_dict.keys()))

    # Adding a slider to select the number of error messages to plot
    num_to_plot = st.slider("Select the number of error messages to plot", min_value=1, max_value=50, value=10)

    # Plot the selected dataset when the button is pressed
    if st.button("Plot Data"):
        plot_data(data_dict[dataset_name], dataset_name, num_to_plot)

else:
    st.error("Please upload at least one CSV file.")

