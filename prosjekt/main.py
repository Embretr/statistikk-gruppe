# Relevante pakker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
# Rense dataene
df = pd.read_csv("prosjekt/Data/lego.population.csv",
                 sep=",", encoding="latin1")

# fjerner forklaringsvariabler vi ikke trenger
df2 = df[['Set_Name', 'Theme', 'Pieces', 'Price', 'Pages',  'Unique_Pieces']]

# fjerner observasjoner med manglende datapunkter
df2 = df2.dropna()

# gjør themes om til string og fjern alle tegn vi ikke vil ha med
df2['Theme'] = df2['Theme'].astype(str)
df2['Theme'] = df2['Theme'].str.replace(r'[^a-zA-Z0-9\s-]', '', regex=True)

# fjerner dollartegn og trademark-tegn fra datasettet
df2['Price'] = df2['Price'].str.replace('\\$', '', regex=True)

# og gjør så prisen om til float
df2['Price'] = df2['Price'].astype(float)

# Mer eller mindre relevante kodesnutter
print(df2.mean(numeric_only=True))
print(df2['Theme'].value_counts())
plt.hist(df2['Price'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Pris i dollar [$]')
plt.ylabel('')
plt.gca().set_aspect(1)
plt.show()
plt.scatter(df2['Pieces'], df2['Price'])
plt.xlabel('Antall brikker')
plt.ylabel('Pris i dollar [$]')
plt.gca().set_aspect(5)
plt.show()
# hva er det dyreste settet i datasettet mon tro?
print(df2.loc[df2['Price'].idxmax()])
# og hvilket har flest brikker?
print(df2.loc[df2['Pieces'].idxmax()])
# hvilke tema har de billigste settene?
df2.groupby('Theme')['Price'].mean().sort_values(ascending=True)[:3]
# hvilke tema har flest brikker?
df2.groupby('Theme')['Pieces'].mean().sort_values(ascending=False)[:3]
sns.pairplot(df2, vars=['Price', 'Pieces', 'Pages', 'Unique_Pieces'],
             hue='Theme',
             diag_kind='kde',
             plot_kws=dict(alpha=0.4))
plt.show()
# enkel lineær regresjon
formel = 'Price ~ Pieces'

modell = smf.ols(formel, data=df2)
resultat = modell.fit()

resultat.summary()
slope = resultat.params['Pieces']
intercept = resultat.params['Intercept']

regression_x = np.array(df2['Pieces'])

regression_y = slope * regression_x + intercept

plt.scatter(df2['Pieces'], df2['Price'], label='Data Points')
plt.plot(regression_x, regression_y, color='red', label='Regression Line')
plt.xlabel('Antall brikker')
plt.ylabel('Pris [$]')
plt.title('Kryssplott med regresjonslinje (enkel LR)')
plt.legend()
plt.grid()
plt.show()
figure, axis = plt.subplots(1, 2, figsize=(15, 5))
sns.scatterplot(x=resultat.fittedvalues, y=resultat.resid, ax=axis[0])
axis[0].set_ylabel("Residual")
axis[0].set_xlabel("Predikert verdi")

sm.qqplot(resultat.resid, line='45', fit=True, ax=axis[1])
axis[1].set_ylabel("Kvantiler i residualene")
axis[1].set_xlabel("Kvantiler i normalfordelingen")
plt.show()
mythemes = ['Star Wars', 'NINJAGO', 'Harry Potter']
subset_df = df2[df2['Theme'].isin(mythemes)]
sns.pairplot(subset_df, vars=['Price', 'Pieces', 'Pages',  'Unique_Pieces'],
             hue='Theme',
             diag_kind='kde',
             plot_kws=dict(alpha=0.4))
plt.show()
# enkel lineær regresjon, tar ikke hensyn til tema
res_sub = smf.ols('Price ~ Pieces', data=subset_df).fit()
# enkel LR for hvert tema hver for seg
resultater = []
for i, theme in enumerate(mythemes):
    modell3 = smf.ols('Price ~ Pieces',
                      data=subset_df[subset_df['Theme'].isin([theme])])
    resultater.append(modell3.fit())
# plott av dataene og regresjonslinjene
for i, theme in enumerate(mythemes):
    slope = resultater[i].params['Pieces']
    intercept = resultater[i].params['Intercept']

    regression_x = np.array(
        subset_df[subset_df['Theme'].isin([theme])]['Pieces'])
    regression_y = slope * regression_x + intercept

    # Plot scatter plot and regression line
    plt.scatter(subset_df[subset_df['Theme'].isin([theme])]['Pieces'], subset_df[subset_df['Theme'].isin(
        [theme])]['Price'], color=plt.get_cmap('tab10')(i))
    plt.plot(regression_x, regression_y,
             color=plt.get_cmap('tab10')(i), label=theme)

plt.xlabel('Antall brikker')
plt.ylabel('Pris')
plt.title('Kryssplott med regresjonslinjer')
plt.legend()
plt.grid()
plt.show()
##
# multippel lineær regresjon
modell3_mlr = smf.ols('Price ~ Pieces + Theme', data=subset_df)
modell3_mlr.fit().summary()
# multippel lineær regresjon med en annen referansekategori
modell3_mlr_alt = smf.ols(
    'Price ~ Pieces + C(Theme, Treatment("Star Wars"))', data=subset_df)
modell3_mlr_alt.fit().summary()
# plott
intercept = [modell3_mlr.fit().params['Theme[T.Star Wars]'], modell3_mlr.fit(
).params['Theme[T.NINJAGO]'], 0] + modell3_mlr.fit().params['Intercept']
slope = modell3_mlr.fit().params['Pieces']

for i, theme in enumerate(mythemes):

    regression_x = np.array(
        subset_df[subset_df['Theme'].isin([theme])]['Pieces'])
    regression_y = slope * regression_x + intercept[i]

    # Plot scatter plot and regression line
    plt.scatter(subset_df[subset_df['Theme'].isin([theme])]['Pieces'], subset_df[subset_df['Theme'].isin(
        [theme])]['Price'], color=plt.get_cmap('tab10')(i))
    plt.plot(regression_x, regression_y,
             color=plt.get_cmap('tab10')(i), label=theme)

# uten tema som forklaringsvariabel:
regression_x = np.array(subset_df['Pieces'])
regression_y = res_sub.params['Pieces'] * \
    regression_x + res_sub.params['Intercept']
plt.plot(regression_x, regression_y, color='black', label='No theme')

plt.xlabel('Antall brikker')
plt.ylabel('Pris')
plt.title('Kryssplott med regresjonslinjer')
plt.legend()
plt.grid()
plt.show()
# med interaksjonsledd mellom antall brikker og tema
modell3_mlri = smf.ols('Price ~ Pieces*Theme', data=subset_df)
modell3_mlri.fit().summary()
# plott
intercept = [modell3_mlri.fit().params['Theme[T.Star Wars]'], modell3_mlri.fit(
).params['Theme[T.NINJAGO]'], 0] + modell3_mlri.fit().params['Intercept']
slope = [modell3_mlri.fit().params['Pieces:Theme[T.Star Wars]'], modell3_mlri.fit(
).params['Pieces:Theme[T.NINJAGO]'], 0] + modell3_mlri.fit().params['Pieces']

for i, theme in enumerate(mythemes):

    regression_x = np.array(
        subset_df[subset_df['Theme'].isin([theme])]['Pieces'])
    regression_y = slope[i] * regression_x + intercept[i]

    # Plot scatter plot and regression line
    plt.scatter(subset_df[subset_df['Theme'].isin([theme])]['Pieces'], subset_df[subset_df['Theme'].isin(
        [theme])]['Price'], color=plt.get_cmap('tab10')(i))
    plt.plot(regression_x, regression_y,
             color=plt.get_cmap('tab10')(i), label=theme)

# uten tema som forklaringsvariabel:
regression_x = np.array(subset_df['Pieces'])
regression_y = res_sub.params['Pieces'] * \
    regression_x + res_sub.params['Intercept']
plt.plot(regression_x, regression_y, color='black',
         label='Theme unaccounted for')

plt.xlabel('Antall brikker')
plt.ylabel('Pris [$]')
plt.title('Kryssplott med regresjonslinjer')
plt.legend()
plt.grid()
plt.show()
# Kode for å lagre plot som (.png)
# fjern 'plt.show()' og erstatt med:
# plt.savefig('my_plot.png')
# Steg 5: Evaluere om modellen passer til dataene
# Plotte predikert verdi mot residual
figure, axis = plt.subplots(1, 2, figsize=(15, 5))
sns.scatterplot(x=modell3_mlri.fit().fittedvalues,
                y=modell3_mlri.fit().resid, ax=axis[0])
axis[0].set_ylabel("Residual")
axis[0].set_xlabel("Predikert verdi")

# Lage kvantil-kvantil-plott for residualene
sm.qqplot(modell3_mlri.fit().resid, line='45', fit=True, ax=axis[1])
axis[1].set_ylabel("Kvantiler i residualene")
axis[1].set_xlabel("Kvantiler i normalfordelingen")
plt.show()
# Gruppere temaer i nye grupper:
# (Harry Potter, NINJAGO og Star Wars havner i én gruppe, City og Friends i en annen, og alle andre i en tredje)
df2['cat'] = np.where(df2['Theme'].isin(['Harry Potter', 'NINJAGO', 'Star Wars']), 'Cat1',
                      np.where(df2['Theme'].isin(['City', 'Friends']), 'Cat2', 'Cat3'))
df2.groupby(['cat']).size().reset_index().rename(columns={0: 'Count'})
df2.groupby(['cat', 'Theme']).size().reset_index().rename(columns={0: 'Count'})
