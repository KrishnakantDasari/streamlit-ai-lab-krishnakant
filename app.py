"""Simple Streamlit app for Exploratory Data Analysis on the Iris dataset.

Features:
- Load Iris from scikit-learn
- Display first rows
- Show summary statistics
- Let user select numeric columns
- Display histogram and scatter plot

Run: `streamlit run app.py`
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def load_iris_dataframe() -> pd.DataFrame:
    """Load the Iris dataset and return a DataFrame with a human-readable target."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # Add target (species) as a categorical column
    df["target"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df


def main() -> None:
    """Main Streamlit app function."""
    st.set_page_config(page_title="Iris EDA", layout="wide")
    st.title("Iris Exploratory Data Analysis")

    # Load data
    dataframe = load_iris_dataframe()

    # Show first rows
    st.header("First rows of the dataset")
    st.dataframe(dataframe.head())

    # Summary statistics for numeric columns
    st.header("Summary statistics")
    st.write(dataframe.describe())

    # Let user pick numeric columns to analyze
    numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
    st.sidebar.header("Column selection")
    selected_columns = st.sidebar.multiselect(
        "Select numeric columns to visualize",
        options=numeric_columns,
        default=numeric_columns[:2],
    )

    if not selected_columns:
        st.warning("Please select at least one numeric column from the sidebar.")
        return

    # Histogram for a single selected column
    st.subheader("Histogram")
    hist_column = st.selectbox("Choose column for histogram", options=selected_columns)
    num_bins = st.slider("Number of bins", min_value=5, max_value=50, value=10)

    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(dataframe[hist_column].dropna(), bins=num_bins, color="#4C72B0", edgecolor="black")
    ax_hist.set_xlabel(hist_column)
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"Histogram of {hist_column}")
    st.pyplot(fig_hist)

    # Scatter plot between two numeric columns
    st.subheader("Scatter plot")
    # If user selected >=2 columns, use those as options; otherwise fall back to all numeric columns
    scatter_options = selected_columns if len(selected_columns) >= 2 else numeric_columns
    x_axis = st.selectbox("X axis", options=scatter_options, index=0)
    # Ensure Y axis has a different default if possible
    y_default_index = 1 if len(scatter_options) > 1 else 0
    y_axis = st.selectbox("Y axis", options=scatter_options, index=y_default_index)

    fig_scatter, ax_scatter = plt.subplots()
    # Color points by species
    for species_name in dataframe["target"].cat.categories:
        species_subset = dataframe[dataframe["target"] == species_name]
        ax_scatter.scatter(
            species_subset[x_axis],
            species_subset[y_axis],
            label=species_name,
            alpha=0.8,
            edgecolors="w",
            s=60,
        )

    ax_scatter.set_xlabel(x_axis)
    ax_scatter.set_ylabel(y_axis)
    ax_scatter.set_title(f"Scatter: {x_axis} vs {y_axis}")
    ax_scatter.legend(title="Species")
    st.pyplot(fig_scatter)


if __name__ == "__main__":
    main()
