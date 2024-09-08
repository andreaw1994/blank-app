import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show_error_message_occurrences(data, dataset_name):
    st.write("### Error Message Occurrences")

    error_counts = data['description'].value_counts()

    num_to_plot = st.slider("Select the number of error messages to plot", min_value=1, max_value=50, value=10)

    error_counts = error_counts.head(num_to_plot)

    fig, ax = plt.subplots(figsize=(14, 8))
    error_counts.plot(kind='bar', ax=ax, color='#3498db', edgecolor='black')

    plt.title(f'Occurrences of Each Error Message in {dataset_name}', fontsize=16)
    plt.xlabel('Error Message', fontsize=14)
    plt.ylabel('Number of Occurrences', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    st.pyplot(fig)

    st.write("### Error Message Occurrences Table")
    error_table = pd.DataFrame({
        'Error Message': error_counts.index,
        'Count': error_counts.values
    })
    st.table(error_table)
