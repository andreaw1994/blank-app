import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_error_code_transition_matrix(data, dataset_name):
    st.write(f"### Error Code Transition Matrix for {dataset_name}")

    # Ensure 'error_code' column exists
    if 'error_code' not in data.columns:
        st.error("Error: 'error_code' column not found in the dataset.")
        return

    # Create a dataframe for error code transitions
    df_cc = data[['error_code']].copy()
    df_cc['next_error_code'] = df_cc['error_code'].shift(-1)
    df_cc['time_delta'] = data.index.to_series().diff().shift(-1)

    # Remove rows with NaN values
    df_cc.dropna(inplace=True)

    # Create transition matrix
    transition_counts = pd.crosstab(df_cc['error_code'], df_cc['next_error_code'])

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(transition_counts, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Count'})

    plt.title(f"Error Code Transition Matrix - {dataset_name}")
    plt.xlabel("Next Error Code")
    plt.ylabel("Current Error Code")
    plt.tight_layout()

    st.pyplot(fig)

    # Calculate top transitions
    transitions = df_cc.groupby(['error_code', 'next_error_code']).size().reset_index(name='count')
    transitions = transitions.sort_values('count', ascending=False).head(10)

    # Get error descriptions
    if 'description' in data.columns:
        error_descriptions = data[['error_code', 'description']].drop_duplicates().set_index('error_code')
    else:
        error_descriptions = pd.DataFrame(index=data['error_code'].unique(), columns=['description'])
        error_descriptions['description'] = 'Description not available'

    # Prepare table data
    table_data = pd.DataFrame({
        'Current Error': transitions['error_code'],
        'Next Error': transitions['next_error_code'],
        'Count': transitions['count'],
        'Current Description': transitions['error_code'].map(lambda x: error_descriptions.loc[x, 'description'] if x in error_descriptions.index else 'N/A'),
        'Next Description': transitions['next_error_code'].map(lambda x: error_descriptions.loc[x, 'description'] if x in error_descriptions.index else 'N/A')
    })

    st.write("### Top 10 Error Transitions")
    st.table(table_data)

    # Display additional statistics
    st.write("### Error Code Statistics")
    error_counts = data['error_code'].value_counts()
    st.bar_chart(error_counts)

    st.write("### Most Common Error Codes")
    st.table(error_counts.head(10))

    # Time between errors
    if 'datetime' in data.columns:
        data['time_between_errors'] = data['datetime'].diff()
        avg_time_between_errors = data['time_between_errors'].mean()
        st.write(f"Average time between errors: {avg_time_between_errors}")

        # Plot time between errors
        fig, ax = plt.subplots(figsize=(12, 6))
        data['time_between_errors'].hist(bins=50, ax=ax)
        plt.title("Distribution of Time Between Errors")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        st.pyplot(fig)
