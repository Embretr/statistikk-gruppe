import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pandas.core.frame import DataFrame

# Set the style for all plots
sns.set_palette("husl")

# Read data
df = pd.read_csv("Data/lego.population.csv", sep=",", encoding="windows-1252")

# Print all distinct themes
print("\nDistinct Themes: ", df['Theme'].unique())

# Clean and prepare data
def clean_data(df):
    # Clean theme names
    df['Theme'] = df['Theme'].astype(str)
    df['Theme'] = df['Theme'].str.replace(r'[^a-zA-Z0-9\s-]', '', regex=True)

    # Clean price data
    df['Price'] = df['Price'].str.replace(r'[$]', '', regex=True)
    df['Price'] = df['Price'].astype(float)

    # Updated licensed themes list
    licensed_themes  = [
        "Architecture",
        "Batman",
        "BrickHeadz",
        "DC",
        "Disney",
        "Harry Potter",
        "Jurassic World",
        "LEGO Frozen 2",
        "LEGO Super Mario",
        "Marvel",
        "Minecraft",
        "Minions",
        "Overwatch",
        "Powerpuff Girls",
        "Speed Champions",
        "Spider-Man",
        "Star Wars",
        "Stranger Things",
        "Trolls World Tour",
        "Unikitty"
    ]

    # Create licensed flag
    df['is_licensed'] = df['Theme'].apply(lambda x: 1 if any(theme.lower() in x.lower() for theme in licensed_themes) else 0)

    return df

# Clean the data
df = clean_data(df)

# Remove missing values for model variables
model_vars = ['Price', 'is_licensed']
df_clean = df.dropna(subset=model_vars)

# Create correlation matrix
plt.figure(figsize=(8, 6))
numeric_df: DataFrame = df_clean[model_vars].select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix: Price vs Licensed Status')
plt.tight_layout()
plt.show()

# Price distribution by license status
plt.figure(figsize=(10, 6))
sns.boxplot(x='is_licensed', y='Price', data=df_clean)
plt.xlabel('Licensed (1) vs Unlicensed (0)')
plt.ylabel('Price ($)')
plt.title('Price Distribution by License Status')
plt.show()

# Basic OLS model with detailed summary
X = df_clean['is_licensed']
y = df_clean['Price']
X = sm.add_constant(X)
model_basic = sm.OLS(y, X).fit()
print("\nBasic Model Summary:")
print(model_basic.summary())

# Diagnostic plots for basic model
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Residual plot
sns.scatterplot(data=pd.DataFrame({'fitted': model_basic.fittedvalues,
                                  'resid': model_basic.resid}),
                x='fitted', y='resid', ax=axes[0])
axes[0].set_ylabel('Residual')
axes[0].set_xlabel('Predicted value')
axes[0].set_title('Residual Plot')

# Q-Q plot
sm.qqplot(model_basic.resid, line='45', fit=True, ax=axes[1])
axes[1].set_ylabel('Quantiles of Residuals')
axes[1].set_xlabel('Quantiles of Normal Distribution')
axes[1].set_title('Normal Q-Q Plot')

plt.tight_layout()
plt.show()

# Summary statistics by license status
summary_stats = df_clean.groupby('is_licensed').agg({
    'Price': ['count', 'mean', 'std', 'min', 'max']
}).round(2)

print("\nSummary Statistics by License Status:")
print(summary_stats)

# Distribution plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=df_clean[df_clean['is_licensed'] == 0], x='Price', bins=30, label='Unlicensed')
sns.histplot(data=df_clean[df_clean['is_licensed'] == 1], x='Price', bins=30, label='Licensed')
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.title('Price Distribution by License Status')
plt.legend()

plt.subplot(1, 2, 2)
sns.kdeplot(data=df_clean[df_clean['is_licensed'] == 0], x='Price', label='Unlicensed')
sns.kdeplot(data=df_clean[df_clean['is_licensed'] == 1], x='Price', label='Licensed')
plt.xlabel('Price ($)')
plt.ylabel('Density')
plt.title('Price Density by License Status')
plt.legend()

plt.tight_layout()
plt.show()
