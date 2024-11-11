import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("Data/lego.population.csv", sep = ",", encoding = "windows-1252")

# fjerner forklaringsvariabler vi ikke trenger
df2 = df[['Year', "Amazon_Price", "Price", "Set_Name"]]

# fjerner observasjoner med manglende datapunkter
df2 = df2.dropna()

# fjerner dollartegn og trademark-tegn fra datasettet
df2['Price'] = df2['Price'].str.replace('\\$', '', regex = True)
df2['Amazon_Price'] = df2['Amazon_Price'].str.replace('\\$', '', regex = True)

# og gjør så prisen om til float
df2['Price'] = df2['Price'].astype(float)
df2['Amazon_Price'] = df2['Amazon_Price'].astype(float)


# Only include 25th - 75th percentile
df2 = df2[(df2['Price'] > df2['Price'].quantile(0.25)) & (df2['Price'] < df2['Price'].quantile(0.75))]
df2 = df2[(df2['Amazon_Price'] > df2['Amazon_Price'].quantile(0.25)) & (df2['Amazon_Price'] < df2['Amazon_Price'].quantile(0.75))]

# Plot the scatter data
# plt.scatter(df2['Year'], df2['Amazon_Price'], color = 'red')
plt.scatter(df2['Year'], df2['Price'], color = 'blue')
plt.show()

df3 = df2.groupby('Year')['Amazon_Price'].mean().reset_index()
df4 = df2.groupby('Year')['Price'].mean().reset_index()
plt.plot(df3['Year'], df3['Amazon_Price'])
plt.plot(df4['Year'], df4['Price'])
plt.xlabel('Utgivelsesår')
plt.ylabel('Pris i dollar [$]')
plt.show()

df5 = df2[df2['Year'] == 2018]

print(df5)