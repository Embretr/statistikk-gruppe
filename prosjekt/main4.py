import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

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
    df['Is_licensed'] = df['Theme'].apply(lambda x: 1 if any(
        theme.lower() in x.lower() for theme in licensed_themes) else 0)

    df = df[["Price", "Is_licensed", "Pieces", "Pages"]]

    return df


# Clean the data
df = clean_data(df)


def plot_data():
    # Histogram prices
    plt.figure(figsize=(10, 6))
    sns.histplot(x=df['Price'], bins=30, kde=False)
    plt.xlabel('Pris')
    plt.ylabel('Frekvens')
    plt.title('Histogram av pris')
    plt.show()

    # Boxplot is_licensed with labels
    plt.figure(figsize=(10, 6))
    licensed_label = df['Is_licensed'].map({1: 'Licensed', 0: 'Non-Licensed'})
    sns.boxplot(x=licensed_label, y="Price", data=df)
    plt.xlabel("Lisensiert")
    plt.ylabel("Pris")
    plt.title("Boksplott: Pris vs. Lisensiert")
    plt.show()

    # Scatterplot for Pieces
    plt.figure(figsize=(10, 6))

    sns.scatterplot(x="Pieces", y="Price", hue="Is_licensed",
                    data=df, alpha=0.7)
    plt.xlabel("Brikker")
    plt.ylabel("Pris")
    plt.title("Kryssplott: Pris vs. Brikker")
    plt.legend(title="Lisensiert = 1")

    plt.show()


plot_data()


def plot_diagnostics(model, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    fitted_values = model.fittedvalues
    residuals = model.resid

    # Residual plot
    ax1.scatter(model.fittedvalues, model.resid)
    ax1.set_xlabel('Predikert verdi')
    ax1.set_ylabel('Residual')
    ax1.set_title('Residualer Plot')

    # Q-Q plot
    sm.qqplot(model.resid, line='45', fit=True, ax=ax2)
    ax2.set_title('QQ Plot')
    ax2.set_xlabel('Kvantiler i normalfordelingen')
    ax2.set_ylabel('Kvantiler i residualene')

    fig.suptitle(title)
    plt.show()


def plot_modelA():
    formula = "Price ~ Pieces"
    model = smf.ols(formula=formula, data=df).fit()
    summary = model.summary()
    print("Model A Summary:")
    print(summary)

    X = df['Pieces']
    y = df['Price']

    plt.figure(figsize=(8, 6))
    # Scatter plot for actual data
    sns.scatterplot(x=X, y=y, alpha=0.6, label='Data')
    plt.plot(X, model.fittedvalues, color='red',
             label='Regresjonslinje')  # Regression line
    plt.title('Model A: Pris vs. brikker')
    plt.xlabel('Brikker')
    plt.ylabel('Pris')
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_diagnostics(model, "Model A Diagnostikk")


plot_modelA()


def plot_modelB():
    formula = "Price ~ Pieces + Pages"
    model = smf.ols(formula=formula, data=df).fit()
    summary = model.summary()
    print("Model B Summary:")
    print(summary)

    X = df[['Pieces', 'Pages']]
    y = df['Price']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for actual data
    ax.scatter(X['Pieces'], X['Pages'], y, alpha=0.6, label='Data')

    # Create a meshgrid for the plane
    xx, yy = np.meshgrid(np.linspace(X['Pieces'].min(), X['Pieces'].max(), 100),
                         np.linspace(X['Pages'].min(), X['Pages'].max(), 100))
    zz = model.params[0] + model.params[1] * xx + model.params[2] * yy

    # Plot the plane
    ax.plot_surface(xx, yy, zz, color='red', alpha=0.3,  # type: ignore // python er dum
                    rstride=100, cstride=100)

    ax.set_title('Model B: Pris vs. Brikker og Sider')
    ax.set_xlabel('Brikker')
    ax.set_ylabel('Sider')
    ax.set_zlabel('Pris')  # type: ignore // python er dum
    ax.legend()
    plt.show()

    plot_diagnostics(model, "Model B Diagnostikk")


plot_modelB()


def plot_modelC1():
    formula = "Price ~ Pieces"

    # Separate the data by Is_licensed
    licensed_df = df[df['Is_licensed'] == 1]
    non_licensed_df = df[df['Is_licensed'] == 0]

    # Fit separate models for each group
    model_licensed = smf.ols(formula=formula, data=licensed_df).fit()
    model_non_licensed = smf.ols(formula=formula, data=non_licensed_df).fit()

    # Plot data for each group side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.scatterplot(data=licensed_df, x='Pieces', y='Price',
                    color='blue', label='Lisensiert data', ax=ax1)
    ax1.plot(licensed_df['Pieces'], model_licensed.fittedvalues,
             color='blue', label='Lisensiert regresjonslinje')
    ax1.set_title('Lisensiert data: Pris vs. Brikker')
    ax1.set_xlabel('Brikker')
    ax1.set_ylabel('Pris')
    ax1.legend()
    ax1.grid(True)

    sns.scatterplot(data=non_licensed_df, x='Pieces', y='Price',
                    color='orange', label='Ikke-lisensiert data', ax=ax2)
    ax2.plot(non_licensed_df['Pieces'], model_non_licensed.fittedvalues,
             color='orange', label='Ikke-liseniert regresjonslinje')
    ax2.set_title('Ikke-lisensiert data: Pris vs. Brikker')
    ax2.set_xlabel('Brikker')
    ax2.set_ylabel('Pris')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle('Model C1: Pris vs. Brikker (Lisensiert vs. Ikke-lisensiert)')
    plt.show()

    # Plot diagnostics for both models
    plot_diagnostics(model_licensed, "Model C1 Lisensiert diagnostikk")
    plot_diagnostics(model_non_licensed,
                     "Model C1 Ikke-lisensiert diagnostikk")


plot_modelC1()


def plot_modelC2():
    formula = "Price ~ Pieces + Is_licensed"
    model = smf.ols(formula=formula, data=df).fit()

    print("Model C2 Summary:")
    print(model.summary())

    plt.figure(figsize=(8, 6))

    # Scatter plots for licensed and non-licensed data
    sns.scatterplot(data=df[df['Is_licensed'] == 1], x='Pieces',
                    y='Price', color='blue', label='Lisensiert data', alpha=0.6)
    sns.scatterplot(data=df[df['Is_licensed'] == 0], x='Pieces',
                    y='Price', color='orange', label='Ikke-lisensiert data', alpha=0.6)

    # Define x values for plotting lines
    x_vals = np.linspace(df['Pieces'].min(), df['Pieces'].max(), 100)

    # Licensed regression line
    licensed_line = model.params['Intercept'] + \
        model.params['Pieces'] * x_vals + model.params['Is_licensed']
    plt.plot(x_vals, licensed_line, color='blue',
             label='Lisensiert regresjonslinje', zorder=2, linestyle='--')

    # Non-licensed regression line
    non_licensed_line = model.params['Intercept'] + \
        model.params['Pieces'] * x_vals
    plt.plot(x_vals, non_licensed_line, color='orange',
             label='Ikke-lisensiert regresjonslinje', zorder=1, linestyle='-.')

    plt.title('Model C2: Pris vs. Brikker + Lisensiert')
    plt.xlabel('Brikker')
    plt.ylabel('Pris')
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_diagnostics(model, "Model C2 Diagnostikk")


plot_modelC2()


def plot_modelC3():
    formula = "Price ~ Pieces * Is_licensed"
    model = smf.ols(formula=formula, data=df).fit()

    print("Model C3 Summary:")
    print(model.summary())

    plt.figure(figsize=(8, 6))

    # Scatter plots
    sns.scatterplot(data=df[df['Is_licensed'] == 1], x='Pieces',
                    y='Price', color='blue', label='Lisensiert data')
    sns.scatterplot(data=df[df['Is_licensed'] == 0], x='Pieces',
                    y='Price', color='orange', label='Ikke-lisensiert data')

    # Regression lines (Different slopes due to interaction term)
    x_vals = np.linspace(df['Pieces'].min(), df['Pieces'].max(), 100)
    licensed_line = model.params['Intercept'] + model.params['Pieces'] * x_vals + \
        model.params['Is_licensed'] + \
        model.params['Pieces:Is_licensed'] * x_vals
    non_licensed_line = model.params['Intercept'] + \
        model.params['Pieces'] * x_vals

    plt.plot(x_vals, licensed_line, color='blue',
             label='Lisensiert regresjonslinje')
    plt.plot(x_vals, non_licensed_line, color='orange',
             label='Ikke-lisensiert regresjonslinje')

    plt.title('Model C3: Pris vs. Brikker * Lisensiert')
    plt.xlabel('Brikker')
    plt.ylabel('Pris')
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_diagnostics(model, "Model C3 Diagnostikk")


plot_modelC3()
