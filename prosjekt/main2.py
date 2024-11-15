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
    df = df[["Theme", "Price", "Pieces", "Pages"]]
    df = df.dropna()

    # Clean theme names
    df['Theme'] = df['Theme'].astype(str)
    df['Theme'] = df['Theme'].str.replace(r'[^a-zA-Z0-9\s-]', '', regex=True)

    # Clean price data
    df['Price'] = df['Price'].str.replace(r'[$]', '', regex=True)
    df['Price'] = df['Price'].astype(float)

    # Updated licensed themes list
    licensed_themes = [
        'Star Wars',
        'Harry Potter',
        'Disney',
        'Marvel',
        'Batman',
        'DC',
        'Spider-Man',
        'Jurassic World',
        'Minecraft',
        'Super Mario',
        'Frozen',
        'Trolls World Tour',
        'Minions',
        'Powerpuff Girls',
        'Overwatch',
        'Stranger Things',
        'THE LEGO MOVIE',
        'Unikitty'
    ]

    # Create licensed flag
    df['Is_licensed'] = df['Theme'].apply(lambda x: 1 if any(
        theme.lower() in x.lower() for theme in licensed_themes) else 0)

    df = df[["Price", "Is_licensed", "Pieces", "Pages"]]

    return df


# Clean the data
df = clean_data(df)

# Remove missing values for model variables
model_vars = ["Price", "Is_licensed",
              "Pieces", "Pages"]

# Create correlation matrix
plt.figure(figsize=(10, 8))
numeric_df: DataFrame = df[model_vars].select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Variables')
plt.tight_layout()
plt.show()

# Basic price distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Is_licensed', y='Price', data=df)
plt.xlabel('Licensed (1) vs Unlicensed (0)')
plt.ylabel('Price ($)')
plt.title('Price Distribution by License Status')

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='Pieces', y='Price', hue='Is_licensed')
plt.xlabel('Number of Pieces')
plt.ylabel('Price ($)')
plt.title('Price vs Pieces by License Status')
plt.tight_layout()
plt.show()

# Basic OLS model with detailed summary
X = df['Is_licensed']
y = df['Price']
X = sm.add_constant(X)
model_basic_detailed = sm.OLS(y, X).fit()
print("\nDetailed Basic Model Summary:")
print(model_basic_detailed.summary())

# Enhanced model
model_enhanced = smf.ols('Price ~ Is_licensed + Pieces + Pages',
                         data=df).fit()
print("\nEnhanced Model Summary:")
print(model_enhanced.summary())

# Diagnostic plots for enhanced model
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Residual plot
sns.scatterplot(data=pd.DataFrame({'fitted': model_enhanced.fittedvalues,
                                  'resid': model_enhanced.resid}),
                x='fitted', y='resid', ax=axes[0])
axes[0].set_ylabel('Residual')
axes[0].set_xlabel('Predicted value')
axes[0].set_title('Residual Plot')

# Q-Q plot
sm.qqplot(model_enhanced.resid, line='45', fit=True, ax=axes[1])
axes[1].set_ylabel('Quantiles of Residuals')
axes[1].set_xlabel('Quantiles of Normal Distribution')
axes[1].set_title('Normal Q-Q Plot')

plt.tight_layout()
plt.show()

# Summary statistics by license status
summary_stats = df.groupby('Is_licensed').agg({
    'Price': ['count', 'mean', 'std', 'min', 'max'],
    'Pieces': 'mean',
    'Minifigures': 'mean'
}).round(2)

print("\nSummary Statistics by License Status:")
print(summary_stats)

# Calculate price per piece
df['price_per_piece'] = df['Price'] / df['Pieces']

plt.figure(figsize=(10, 6))
sns.boxplot(x='Is_licensed', y='price_per_piece', data=df)
plt.xlabel('Licensed (1) vs Unlicensed (0)')
plt.ylabel('Price per Piece ($)')
plt.title('Price per Piece Distribution by License Status')
plt.show()
