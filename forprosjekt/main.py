# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

data = pd.read_csv('shoesize_height.csv')

print(data.head())


X = data['skostr'].values
Y = data['hoyde'].values

slope, intercept, r_value, p_value, std_err = linregress(X, Y)


print(f"Regresjonslinje: y = {intercept:.2f} + {slope:.2f}x")

print(f"R-squared: {r_value**2:.2f}") # type: ignore

plt.scatter(X, Y, color='blue', label='Data')  
plt.plot(X, intercept + slope * X.astype(float), color='red',
         label='Regresjonslinje') 
plt.xlabel('Skostørrelse')
plt.ylabel('Høyde (cm)')
plt.title('Skostørrelse vs. Høyde')
plt.legend()

plt.savefig('regresjonsplot.pdf')
plt.show()  
