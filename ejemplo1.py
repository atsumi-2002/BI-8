import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn.metrics import mean_squared_error as mse, r2_score as r2s

archivo = pd.read_excel('BI_Alumnos08.xlsx')
data = pd.DataFrame()

alt = archivo['Altura']
ed = archivo['Edad']
data['Altura'] = alt
data['Edad'] = ed
xy = np.array(data)
z = archivo['Peso'].values
regLM = lm.LinearRegression()
regLM.fit(xy, z)
z_pred = regLM.predict(xy)

print('Analisis de datos de BI_Alumnos08.xlsx')
print('Coeficiente de R: ', regLM.coef_)
print('Termino independiente: ', regLM.intercept_)
print('Error cuadrado medio: %.2f' % mse(z, z_pred))
print('Puntaje de varianza: %.2f' % r2s(z, z_pred))

predPeso = regLM.predict([[180, 22]])
print('Prediccion de peso de alumno de 22 años y 180cm: ', int(predPeso))
