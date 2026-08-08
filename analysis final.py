"""
Global Inflation Analysis: Trends, Stability and Forecasting (2000-2025)

Project Overview:
This project analyzes global inflation patterns using World Bank CPI inflation data.
It includes inflation trends, country comparisons, stability analysis,
correlation analysis, and machine learning-based forecasting.

Tools:
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Author: Tooba Khalil
"""


# ============================================================
# 1. Import Libraries
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from pathlib import Path


# ============================================================
# 2. Project Paths
# ============================================================

project_folder = Path(__file__).parent

data_folder = project_folder / "data"

output_folder = project_folder / "outputs" / "graphs"

output_folder.mkdir(
    exist_ok=True
)


# ============================================================
# 3. Load Dataset
# ============================================================

csv_file = data_folder / "API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_285.csv"

df = pd.read_csv(
    csv_file,
    skiprows=4
)

print("Dataset Loaded Successfully!")

print(df.head())


# ============================================================
# 4. Data Cleaning
# ============================================================

# Remove unnecessary columns

df = df.drop(
    columns=[
        "Indicator Name",
        "Indicator Code",
        "Unnamed: 70"
    ],
    errors="ignore"
)


# Keep years 2000-2025

years = [str(year) for year in range(2000, 2026)]

df = df[
    ["Country Name", "Country Code"] + years
]


# Convert wide format to long format

df = df.melt(
    id_vars=["Country Name", "Country Code"],
    value_vars=years,
    var_name="Year",
    value_name="Inflation"
)


# Convert Year to numeric

df["Year"] = df["Year"].astype(int)


# Remove missing inflation values

df = df.dropna(
    subset=["Inflation"]
)


# Rename country column

df = df.rename(
    columns={
        "Country Name": "Country"
    }
)


# Save cleaned dataset

df.to_csv(
    data_folder / "cleaned_inflation_data.csv",
    index=False
)


print("Data Cleaning Completed!")
print(df.head())

# ============================================================
# 5. Data Inspection
# ============================================================

print("Dataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 6. Missing Value Analysis
# ============================================================

missing_values = df.isnull().sum()

print("\nMissing Values:")
print(missing_values)

print("\nTotal Missing Values:")
print(missing_values.sum())
# ============================================================
# 7. Global Inflation Trend
# ============================================================

def global_inflation_trend(df):

    global_average = (
        df.groupby("Year")["Inflation"]
        .mean()
    )

    plt.figure(figsize=(10,5))

    plt.plot(
        global_average.index,
        global_average.values,
        marker="o"
    )

    plt.title(
        "Global Average Inflation Trend (2000-2025)"
    )

    plt.xlabel("Year")
    plt.ylabel("Average Inflation (%)")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_folder / "Global_Inflation_Trend.png",
        dpi=300
    )

    plt.show()

    return global_average


global_trend = global_inflation_trend(df)

# ============================================================
# 8. Top 10 Highest Inflation Countries
# ============================================================

def highest_inflation_countries(df):

    highest10 = (
        df.groupby("Country")["Inflation"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nTop 10 Highest Inflation Countries:")
    print(highest10)


    plt.figure(figsize=(10,5))

    highest10.plot(
        kind="bar"
    )

    plt.title(
        "Top 10 Highest Average Inflation Countries (2000-2025)"
    )

    plt.xlabel("Country")
    plt.ylabel("Average Inflation (%)")

    plt.xticks(rotation=45)

    plt.tight_layout()


    plt.savefig(
        output_folder / "Highest_Inflation_Countries.png",
        dpi=300
    )

    plt.show()


    return highest10


highest_inflation = highest_inflation_countries(df)

# =============================================================
# 9. Top 10 Lowest Inflation Countries
# ============================================================

def lowest_inflation_countries(df):

    lowest10 = (
        df.groupby("Country")["Inflation"]
        .mean()
        .sort_values()
        .head(10)
    )

    print("\nTop 10 Lowest Inflation Countries:")
    print(lowest10)


    plt.figure(figsize=(10,5))

    lowest10.plot(
        kind="bar"
    )

    plt.title(
        "Top 10 Lowest Average Inflation Countries (2000-2025)"
    )

    plt.xlabel("Country")
    plt.ylabel("Average Inflation (%)")

    plt.xticks(rotation=45)

    plt.tight_layout()


    plt.savefig(
        output_folder / "Lowest_Inflation_Countries.png",
        dpi=300
    )

    plt.show()


    return lowest10


lowest_inflation = lowest_inflation_countries(df)

# ============================================================
# 10. Pakistan Inflation Analysis
# ============================================================

def pakistan_inflation_analysis(df):

    pakistan = df[
        df["Country"] == "Pakistan"
    ]


    plt.figure(figsize=(10,5))

    plt.plot(
        pakistan["Year"],
        pakistan["Inflation"],
        marker="o"
    )


    plt.title(
        "Pakistan Inflation Trend (2000-2025)"
    )

    plt.xlabel("Year")
    plt.ylabel("Inflation (%)")


    plt.grid(True)
    plt.tight_layout()


    plt.savefig(
        output_folder / "Pakistan_Inflation_Trend.png",
        dpi=300
    )

    plt.show()


    return pakistan


pakistan_data = pakistan_inflation_analysis(df)


# ============================================================
# 11. Regional Inflation Analysis
# ============================================================

def regional_inflation_analysis(df):

    south_asia = [
        "Pakistan",
        "India",
        "Bangladesh",
        "Sri Lanka",
        "Nepal"
    ]

    regional_data = df[
        df["Country"].isin(south_asia)
    ]


    regional_average = (
        regional_data.groupby("Year")["Inflation"]
        .mean()
    )


    plt.figure(figsize=(10,5))

    plt.plot(
        regional_average.index,
        regional_average.values,
        marker="o"
    )


    plt.title(
        "South Asia Average Inflation Trend (2000-2025)"
    )

    plt.xlabel("Year")
    plt.ylabel("Average Inflation (%)")


    plt.grid(True)
    plt.tight_layout()


    plt.savefig(
        output_folder / "Regional_Inflation_Trend.png",
        dpi=300
    )


    plt.show()


    return regional_average


regional_analysis = regional_inflation_analysis(df)

# ============================================================
# 12. Inflation Stability Analysis
# ============================================================

def inflation_stability_analysis(df):

    stability = (
        df.groupby("Country")["Inflation"]
        .std()
        .sort_values()
        .head(10)
    )


    print("\nMost Stable Inflation Countries:")
    print(stability)


    plt.figure(figsize=(10,5))

    stability.plot(
        kind="bar"
    )


    plt.title(
        "Top 10 Most Stable Inflation Countries"
    )

    plt.xlabel("Country")
    plt.ylabel("Inflation Volatility")


    plt.xticks(rotation=45)

    plt.tight_layout()


    plt.savefig(
        output_folder / "Inflation_Stability.png",
        dpi=300
    )


    plt.show()


    return stability


stable_countries = inflation_stability_analysis(df)

# ============================================================
# 13. Correlation Heatmap
# ============================================================

def correlation_heatmap(df):

    correlation = df[
        ["Year", "Inflation"]
    ].corr()


    plt.figure(figsize=(6,4))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm"
    )


    plt.title(
        "Inflation Correlation Heatmap"
    )

    plt.tight_layout()


    plt.savefig(
        output_folder / "Correlation_Heatmap.png",
        dpi=300
    )


    plt.show()


correlation_heatmap(df)
# ============================================================
# 14. Inflation Forecast (Machine Learning)
# ============================================================

def inflation_forecast(df):

    data = (
        df.groupby("Year")["Inflation"]
        .mean()
        .reset_index()
    )


    model = LinearRegression()

    model.fit(
        data[["Year"]],
        data["Inflation"]
    )


    future_years = pd.DataFrame(
        {"Year": range(2026, 2031)}
    )


    future_years["Forecast"] = model.predict(
        future_years[["Year"]]
    )


    plt.figure(figsize=(10,5))


    plt.plot(
        data["Year"],
        data["Inflation"],
        label="Actual"
    )


    plt.plot(
        future_years["Year"],
        future_years["Forecast"],
        marker="o",
        label="Forecast"
    )


    plt.title(
        "Global Inflation Forecast (2026-2030)"
    )

    plt.xlabel("Year")
    plt.ylabel("Inflation (%)")

    plt.legend()

    plt.tight_layout()


    plt.savefig(
        output_folder / "Inflation_Forecast.png",
        dpi=300
    )


    plt.show()


    return future_years


forecast = inflation_forecast(df)
# ============================================================
# 15. Save Outputs
# ============================================================

print("All analysis completed successfully!")
print(f"Graphs saved in: {output_folder}")