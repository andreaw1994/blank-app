import pandas as pd
import datetime
import matplotlib.pyplot as plt
import streamlit as st

def trim_time(data):
    start_time = data.loc[data[data["good_objects"] > 0].index[0], "datetime"]
    end_time = data.loc[data["good_objects"].idxmax(), "datetime"]
    return data[(data["datetime"] >= start_time) & (data["datetime"] <= end_time)].reset_index(drop=True)

def calculate_error_durations(data, quantile, pre_start=10, post_start=10):
    tmp_data = data.copy()
    column = f"q{quantile}"
    quantile_thresh = tmp_data[tmp_data["zero_time"] != datetime.timedelta()]["zero_time"].quantile(quantile)
    quantile_df = tmp_data[tmp_data["zero_time"].dt.total_seconds() >= quantile_thresh.total_seconds()]
    tmp_data.loc[tmp_data["zero_time"] >= quantile_thresh, column] = True
    tmp_data.fillna({column: False}, inplace=True)

    errors = []
    indices = []
    for i in (quantile_df["datetime"] - quantile_df["zero_time"]):
        error_code = tmp_data[(tmp_data["datetime"] >= (i - datetime.timedelta(seconds=pre_start))) &
                              (tmp_data["datetime"] <= (i + datetime.timedelta(seconds=post_start)))]["description"]
        errors.extend(error_code[error_code.notna()].values)
        indices.extend(error_code[error_code.notna()].index.values.astype(int))

    return tmp_data, errors, indices

def preprocess_data(data):
    data = trim_time(data)
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["derivative_good_objects"] = data[data["good_objects"].notna()]["good_objects"].diff() / \
                                      data[data["good_objects"].notna()]["datetime"].diff().dt.total_seconds()
    data["derivative_good_objects"] = data["derivative_good_objects"].fillna(0)

    data["zero_time"] = datetime.timedelta()
    prev_der = None
    first_time = data.loc[0, "datetime"]
    start_from = data.loc[0, "good_objects"]

    for i, row in enumerate(data.itertuples()):
        if prev_der is not None and row.derivative_good_objects > 0 and prev_der == 0 and row.good_objects != start_from:
            zero_time = (row.datetime - first_time)
            data.at[i, "zero_time"] = zero_time
        elif prev_der is not None and row.derivative_good_objects == 0 and prev_der > 0:
            first_time = row.datetime
        prev_der = row.derivative_good_objects

    return data

def show_complex_analysis(data, dataset_name):
    st.write(f"### Complex Analysis for {dataset_name}")

    data = preprocess_data(data)

    st.write("### Analysis Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        # Slider to select top n% longest pauses
        top_percent = st.slider("Select top % longest pauses", 1, 99, 99, 1)  # Slider from 1% to 99%
        # Convert top percent to quantile (invert logic: higher percent means higher quantile)
        quantile = 1 - (top_percent / 100.0)  # Inverted logic
    with col2:
        pre_start = st.slider("Pre-start time (seconds)", 0, 100, 10, 1)
    with col3:
        post_start = st.slider("Post-start time (seconds)", 0, 100, 10, 1)

    exclude_1705 = st.checkbox("Exclude Error Code 1705", value=False)

    if exclude_1705:
        data = data[data['error_code'] != 1705].reset_index(drop=True)

    data, errors, indices = calculate_error_durations(data=data, quantile=quantile, pre_start=pre_start, post_start=post_start)

    # Error analysis and bar plot to show only the top 10 errors
    st.write("### Error Analysis")
    error_counts = pd.Series(errors).value_counts().sort_values(ascending=False).head(10)

    # Create a bar chart using Matplotlib
    fig, ax = plt.subplots(figsize=(14, 8))
    error_counts.plot(kind='bar', ax=ax, color='#3498db', edgecolor='black')

    # Set title, labels, and ticks
    plt.title(f'Occurrences of Each Error Message in {dataset_name}', fontsize=16)
    plt.xlabel("Error Description", fontsize=14)
    plt.ylabel("Number of Occurrences", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Add grid and layout adjustments
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    st.pyplot(fig)

    # Display the top 10 errors in a table
    st.write("### Top 10 Most Frequent Errors")
    st.table(error_counts.reset_index().rename(columns={"index": "Error Description", 0: "Count"}))

def process_csv(file):
    data = pd.read_csv(file)
    return data
