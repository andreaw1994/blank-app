#--------------
#Plotting the distribution of pause length as an interactive zoomable plot

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Part 1: Upload CSV File
st.write("## Upload your CSV file with pause lengths")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Part 2: Read the CSV into a pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Exclude rows where 'length_seconds' is 0
    df_filtered = df[df['length_seconds'] > 0]

    # Preview the first few rows of the filtered DataFrame
    st.write("## Preview of the uploaded data (Excluding pauses with length 0)")
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
    st.write(f"### The 99th percentile of pause durations is {percentile_99:.2f} seconds (Excluding 0-length pauses).")
