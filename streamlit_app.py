#Imports

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime
from functools import reduce
from utils import process_csv
from error_code_transition_matrix import show_error_code_transition_matrix
from error_message_occurrences import show_error_message_occurrences
from complex_analysis import show_complex_analysis, process_csv

st.title("STE1 Manufacturing Analyzer")

st.write(
    "This app allows you to interactively visualize and analyze key performance metrics and error patterns of a STE1 production run."
)

#----------------------------

#Plot the distribution of pause length as an interactive zoomable plot

st.write("### Module 1: Pause Length Analysis")
uploaded_file = st.file_uploader("Choose a CSV file with pause lengths", type="csv", key="file_uploader")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Part 2: Read the CSV into a pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Exclude rows where 'length_seconds' is 0
    df_filtered = df[df['length_seconds'] > 0]

    # Preview the first few rows of the filtered DataFrame
    st.write("#### Preview of the uploaded data (Excluding pauses with length 0)")
    st.dataframe(df_filtered.head())

    # Part 3: Calculate the 99th percentile for 'length_seconds'
    percentile_99 = df_filtered['length_seconds'].quantile(0.99)

    # Part 4: Calculate bin size and ensure the 99th percentile aligns with a bin edge
    min_value = df_filtered['length_seconds'].min()
    max_value = df_filtered['length_seconds'].max()

    # Fixed bin size to ensure 99th percentile aligns with a bin edge
    bin_size = (percentile_99 - min_value) / np.ceil((percentile_99 - min_value) / 20)  # Choose 20 as an arbitrary bin count

    # Create bins starting from min value to max value
    bins = np.arange(min_value, max_value + bin_size, bin_size)

    # Create the Plotly histogram figure
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df_filtered['length_seconds'],
        xbins=dict(start=min_value, end=max_value, size=bin_size),  # Use calculated bin size
        marker=dict(color='lightblue', line=dict(color='black', width=1)),
        hovertemplate='Pause Duration: %{x:.2f}s<br>Count: %{y}<extra></extra>'
    ))

    # Add a vertical line at the 99th percentile
    fig.add_vline(
        x=percentile_99,
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text="99th Percentile",
        annotation_position="top right"
    )

    # Update layout to allow zooming and set initial view
    fig.update_layout(
        xaxis_title="Pause Duration (seconds)",
        yaxis_title="Frequency",
        title="Distribution of Pause Durations (Excluding Zero-Length Pauses)",
        title_x=0.5,  # Center the title
        dragmode="zoom",  # Enable zooming
        template='plotly_white',
        xaxis=dict(
            range=[0, 500]  # Set initial x-axis range
        ),
        yaxis=dict(
            range=[0, 30]  # Set initial y-axis range
        ),
        # Ensure toolbar buttons are visible
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Zoom",
                         method="relayout",
                         args=[{"xaxis.type": "linear", "yaxis.type": "linear"}]),
                    dict(label="Reset",
                         method="relayout",
                         args=[{"xaxis.autorange": True, "yaxis.autorange": True}])
                ],
                direction="down",
                showactive=False,
                x=0.17,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ]
    )

    # Display the plot with toolbar
    st.plotly_chart(fig, use_container_width=True)

    # Display the calculated 99th percentile value
    st.write(f"#### The 99th percentile of pause durations is {percentile_99:.2f} seconds (Excluding 0-length pauses).")

#------------

st.write("### Module 2: Error Message Analysis")
uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    data_dict = {}
    for file in uploaded_files:
        file_name = file.name
        data_dict[file_name] = process_csv(file)

    st.write("#### Preview of the uploaded data")
    first_file = uploaded_files[0].name
    st.write(f"Preview of the first uploaded file: {first_file}")
    st.dataframe(data_dict[first_file].head())

    st.write("#### Analysis Options")

    dataset_name = st.selectbox("Select Dataset", list(data_dict.keys()))
    data = data_dict[dataset_name]

    analysis_type = st.selectbox("Select Analysis Type", [
        "Error Code Transition Matrix",
        "Error Message Occurrences",
        "Complex Analysis"
    ])

    if analysis_type == "Error Code Transition Matrix":
        show_error_code_transition_matrix(data, dataset_name)
    elif analysis_type == "Error Message Occurrences":
        show_error_message_occurrences(data, dataset_name)
    elif analysis_type == "Complex Analysis":
        show_complex_analysis(data, dataset_name)
else:
    st.error("Please upload at least one CSV file.")



#--------------------------------
#Adapted version of Lukas' script below

#import streamlit as st
#import pandas as pd
#import matplotlib.pyplot as plt

# Part 2: Upload CSV Files
#st.write("### Step 1: Upload Your CSV Files")
#uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

# Part 3: Read and Process Uploaded Files
#def process_csv(file, column_index):
    # Load the CSV file into a pandas DataFrame
#    data = pd.read_csv(file)

    # Copy the specified column to a new column at the end
#    data['good_copy'] = data.iloc[:, column_index]  # Assuming 'D' is the column to copy

    # Fill any blanks in the new column 'good_copy' with the previous value using ffill()
#    data['good_copy'] = data['good_copy'].ffill()

    # Calculate the difference between the current value and the previous value in 'good_copy'
#    data['difference'] = data['good_copy'].diff()

#    return data

#if uploaded_files:
    # Part 4: Process and store each CSV file in a dictionary, using the original filename
#    data_dict = {}
#    for file in uploaded_files:
#        file_name = file.name  # Get the original filename
#        data_dict[file_name] = process_csv(file, 3)

    # Show a preview of the first uploaded file
#    st.write("### Step 2: Data Preview")
#    first_file = uploaded_files[0].name
#    st.write(f"Preview of the first uploaded file: {first_file}")
#    st.dataframe(data_dict[first_file].head())

    # Part 5: Plotting Function
#    def plot_data(data, dataset_name, num_to_plot):
#        try:
            # Filter out the rows where 'error_code' is 1705
#            filtered_data = data[data['error_code'] != 1705]

            # Count the occurrences of each unique error message in the 'description' column
 #           error_counts = filtered_data['description'].value_counts()

            # Limit the number of error messages to plot
  #          error_counts = error_counts.head(num_to_plot)

            # Plot the occurrences using a bar chart
   #         plt.figure(figsize=(14, 8))  # Increase figure size for better readability
   #         error_counts.plot(kind='bar', color='#3498db', edgecolor='black')  # Add edge color for clarity

   #         plt.title(f'Occurrences of Each Error Message in {dataset_name} (Excluding error_code 1705)', fontsize=16)
   #         plt.xlabel('Error Message', fontsize=14)
   #         plt.ylabel('Number of Occurrences', fontsize=14)

            # Wrap x-axis labels to improve readability
 #           plt.xticks(rotation=45, ha='right', fontsize=12)

            # Add horizontal gridlines for better readability
 #           plt.grid(axis='y', linestyle='--', alpha=0.7)

 #           plt.tight_layout()  # Adjust layout to fit labels
 #           st.pyplot(plt)
 #           plt.close()  # Close the plot after rendering to avoid memory issues

 #       except Exception as e:
 #           st.error(f"An error occurred while plotting: {e}")

    # Part 6: User Interface for Plotting
 #   st.write("### Step 3: Plot Occurrences of All Error Messages")

    # Dynamically create a selection box with the original filenames
#    dataset_name = st.selectbox("Select Dataset", list(data_dict.keys()))

    # Adding a slider to select the number of error messages to plot
#    num_to_plot = st.slider("Select the number of error messages to plot", min_value=1, max_value=50, value=10)

    # Plot the selected dataset when the button is pressed
#    if st.button("Plot Data"):
#        plot_data(data_dict[dataset_name], dataset_name, num_to_plot)

#else:
#    st.error("Please upload at least one CSV file.")

#--------------------------------
#Adapted version of Josh's script below

# Part 6: User Interface for Plotting
st.write("### Step 3: Duration of long pauses (top 1%), surrounding errors and good object production")

# Step 7: Use the currently selected dataset (`data_dict[dataset_name]`)
data = data_dict[dataset_name]  # Set `data` to the selected dataset

# Convert the "datetime" column to datetime format
data["datetime"] = pd.to_datetime(data["datetime"])

# Calculate the derivative of "good_objects" over time
data["derivative_good_objects"] = (
    data[data["good_objects"].notna()]["good_objects"].diff() / 
    data[data["good_objects"].notna()]["datetime"].diff().dt.total_seconds()
)
data["derivative_good_objects"] = data["derivative_good_objects"].fillna(0)

# Initialize the "zero_time" column with 0 timedelta
data["zero_time"] = datetime.timedelta()
prev_der = None
first_time = data.loc[data["good_objects"].first_valid_index(), "datetime"]
start_from = data.loc[data["good_objects"].first_valid_index(), "good_objects"]

# Loop through each row and calculate "zero_time" for transitions
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

# Sort the "zero_time" values
sorted_times = data[data["zero_time"] > datetime.timedelta()]["zero_time"].sort_values(ascending=False)

# Plot Pause Durations (top 1%)
fig, ax1 = plt.subplots()
fig.set_size_inches(28.5, 15.5)

# Scatter plot for Pause Durations (top 1%)
ax1.scatter(data[data["zero_time"] > datetime.timedelta()]["datetime"], 
            data[data["zero_time"] > datetime.timedelta()]["zero_time"].dt.total_seconds())

# Add labels and title
ax1.set_xlabel('Datetime', fontsize=14)
ax1.set_ylabel('Pause Duration (top 1%) (seconds)', fontsize=14)
ax1.set_title('Occurrences of Pause Durations (top 1%)', fontsize=16)
ax1.grid(True)

# Display the plot in Streamlit
st.pyplot(fig)

# Calculate the 99th percentile of non-zero "zero_time"
q99 = data[data["zero_time"] != datetime.timedelta()]["zero_time"].quantile(0.99)
q99_df = data[data["zero_time"].dt.total_seconds() >= q99.total_seconds()]
data.loc[data["zero_time"] >= q99, "99q"] = True
data.fillna({"99q": False}, inplace=True)

# Find errors surrounding the high pause durations
errors = []
indices = []
good_counter = []

for i in (q99_df["datetime"] - q99_df["zero_time"]):
    # Adjust the window (seconds before and after the pause) to look for errors
    error_code = data[(data["datetime"] >= (i - datetime.timedelta(seconds=10))) & 
                      (data["datetime"] <= (i + datetime.timedelta(seconds=10)))]["description"]
    errors.extend(error_code[error_code.notna()].values)
    indices.extend(error_code[error_code.notna()].index.values.astype(int))

# Display error counts in a DataFrame
st.write(pd.DataFrame(errors).value_counts())

# Plot results with multiple axes
fig, ax1 = plt.subplots()
fig.set_size_inches(28.5, 15.5)

# Create multiple y-axes
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', -40))  # Move the 3rd axis outward

# Plot good_objects and inactive_time on the first axis
ax1.plot(data[data["good_objects"] >= 5]["datetime"], 
         data[data["good_objects"] >= 5]["good_objects"].ffill(), c='b', label='Good Objects')
ax1.plot(data[data["good_objects"] >= 5]["datetime"], 
         data[data["good_objects"] >= 5]["inactive_time"].ffill(), c='g', label='Inactive Time')

# Plot errors on the second axis
ax2.scatter(data.iloc[indices]["datetime"], 
            data.iloc[indices]["description"], c='r', label='Errors')

# Plot stem plot for pause durations (top 1%) on the third axis with blue color
ax3.stem(q99_df["datetime"] - q99_df["zero_time"], q99_df["zero_time"], 
         basefmt=" ", linefmt='b-', markerfmt='bo', label='Pause Duration (top 1%)')

# Add labels, title, and legends
ax1.set_xlabel('Datetime', fontsize=14)
ax1.set_ylabel('Good Objects / Inactive Time', fontsize=14)
ax1.set_title('System Activity and Error Analysis', fontsize=16)
ax1.legend(loc='upper left')

ax2.set_ylabel('Error Descriptions', fontsize=14)
ax2.legend(loc='upper right')

ax3.set_ylabel('Pause Duration (top 1%) (seconds)', fontsize=14)
ax3.legend(loc='upper center')

# Adjust layout to fit all elements
fig.tight_layout()
fig.subplots_adjust(right=0.85)  # Make room for the third y-axis

# Display the final plot in Streamlit
st.pyplot(fig)




