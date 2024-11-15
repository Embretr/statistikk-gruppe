import pandas as pd
import matplotlib.pyplot as plt

# Read and clean the dataset
df = pd.read_csv("Data/lego.population.csv", sep=",", encoding="windows-1252")

# Select columns including Theme this time
df2 = df[['Year', "Amazon_Price", "Price", "Set_Name", "Theme"]]

# Remove missing values
df2 = df2.dropna()

# Clean price data
df2['Price'] = df2['Price'].str.replace('\\$', '', regex=True)
df2['Amazon_Price'] = df2['Amazon_Price'].str.replace('\\$', '', regex=True)

# Convert to float
df2['Price'] = df2['Price'].astype(float)
df2['Amazon_Price'] = df2['Amazon_Price'].astype(float)

# Print unique themes
print("Unique LEGO themes:")
unique_themes = sorted(df2['Theme'].unique())
for theme in unique_themes:
    print(theme)

# List of branded themes
branded_themes = [
    "Architecture",
    "Batman™",
    "BrickHeadz",
    "DC",
    "Disney™",
    "Harry Potter™",
    "Jurassic World™",
    "LEGO® Frozen 2",
    "LEGO® Super Mario™",
    "Marvel",
    "Minecraft™",
    "Minions",
    "Overwatch®",
    "Powerpuff Girls™",
    "Speed Champions",
    "Spider-Man",
    "Star Wars™",
    "Stranger Things",
    "Trolls World Tour",
    "Unikitty!™"
]


# Categorize sets
df2['Category'] = df2['Theme'].apply(lambda x: 'Branded' if x in branded_themes else 'Generic')

# Print summary of categorization
print("\nNumber of sets in each category:")
print(df2['Category'].value_counts())

# Optional: Print example sets from each category
print("\nExample branded sets:")
print(df2[df2['Category'] == 'Branded'][['Set_Name', 'Theme']].head())

print("\nExample generic sets:")
print(df2[df2['Category'] == 'Generic'][['Set_Name', 'Theme']].head())
