# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Step 1: Read the CSV file containing shoe size and height data
# Replace 'shoesize_height.csv' with the correct path to your file if necessary
data = pd.read_csv('shoesize_height.csv')
# Display the first few rows of the data to ensure it's loaded correctly
print(data.head())

# Step 2: Extract the 'Skostørrelse' (shoe size) and 'Høyde' (height) columns
X = data['skostr'].values
Y = data['hoyde'].values

# Step 3: Perform linear regression using scipy
# This calculates the slope, intercept, and other statistics
slope, intercept, r_value, p_value, std_err = linregress(X, Y)

# Display the regression line equation and the R-squared value
print(f"Regresjonslinje: y = {intercept:.2f} + {slope:.2f}x")
print(f"R-squared: {r_value**2:.2f}")

# Step 4: Plot the data points and the regression line
plt.scatter(X, Y, color='blue', label='Data')  # Scatter plot of the data
plt.plot(X, intercept + slope * X, color='red',
         label='Regresjonslinje')  # Plot the regression line
plt.xlabel('Skostørrelse')
plt.ylabel('Høyde (cm)')
plt.title('Skostørrelse vs. Høyde')
plt.legend()

# Step 5: Save the plot as a PDF
plt.savefig('regresjonsplot.pdf')
plt.show()  # Display the plot
