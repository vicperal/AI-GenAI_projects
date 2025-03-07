####################################################################################################
# Victor Peral - 2025 
#
# ML model for clinical study - prediction of the drug efficacy based on the dose and age of the patient.
# Usage of a linear regression model.
#  
# WARNING: the model is trained with a randon dataset of clinical study created just for the purpose
# of validating the end-to-end process of ML model creation
####################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Crear un DataFrame de ensayo clínico
ensayo_clinico = pd.DataFrame({
    'paciente_id': range(1, 1001),
    'edad': np.random.randint(18, 80, 1000),
    'sexo': np.random.choice(['M', 'F'], 1000),
    'dosis': np.random.choice([50, 100, 150, 200], 1000),
    'eficacia': np.random.uniform(0, 1, 1000),
    'efecto_secundario': np.random.choice(['Ninguno', 'Leve', 'Moderado', 'Severo'], 1000)
})

# Exploración básica
print(ensayo_clinico.head())
print(ensayo_clinico.describe())

# Limpieza de datos
ensayo_clinico = ensayo_clinico.dropna()
ensayo_clinico['dosis'] = ensayo_clinico['dosis'].astype(float)

print (ensayo_clinico)
#########################################################
#########################################################

# visualizacion de datos
# Distribución de efectos secundarios
plt.figure(figsize=(10, 6))
sns.countplot(x='efecto_secundario', data=ensayo_clinico)
plt.title('Distribución de Efectos Secundarios')
plt.xticks(rotation=45)
plt.show()

# Correlación entre dosis y eficacia
plt.figure(figsize=(8, 6))
sns.scatterplot(x='dosis', y='eficacia', data=ensayo_clinico)
plt.title('Relación entre Dosis y Eficacia')
plt.show()

######################################################################
######################################################################

# Crear un DataFrame de inventario
inventario = pd.DataFrame({
    'id': range(1, 6),
    'medicamento': ['Aspirina', 'Ibuprofeno', 'Paracetamol', 'Amoxicilina', 'Omeprazol'],
    'stock': [1000, 800, 1200, 500, 700],
    'precio': [5.99, 7.50, 4.25, 12.00, 8.75]
})

# Crear conexión a la base de datos en memoria
conn = sqlite3.connect(':memory:')

print("inventario\n:", inventario.head())

# Guardar el DataFrame en la base de datos
inventario.to_sql('inventario', conn, index=False)

# Consulta SQL
query = "SELECT medicamento, stock FROM inventario WHERE stock < 1000"
resultado = pd.read_sql_query(query, conn)
print("resultado: \n", resultado)

conn.close()
######################################################################
# modelo de regresion lineal para predecir la eficacia de medicamentos
# eficacia es variable dependiente y dosis y edad son variables independientes
#######################################################################

# Preparar los datos
X = ensayo_clinico[['dosis', 'edad']]
y = ensayo_clinico['eficacia']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Hacer predicciones
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error cuadrático medio: {mse}")
print(f"R2 Score: {r2}")

######################################################################
# muestra el grafico de la prediccion
######################################################################
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Crear la figura y el eje 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar los datos reales
ax.scatter(X_test['dosis'], X_test['edad'], y_test, c='blue', label='Datos reales')

# Graficar las predicciones
ax.scatter(X_test['dosis'], X_test['edad'], y_pred, c='red', label='Predicciones')

# Configurar etiquetas y título
ax.set_xlabel('Dosis')
ax.set_ylabel('Edad')
ax.set_zlabel('Eficacia')
ax.set_title('Predicciones vs Datos Reales')

# Añadir leyenda
ax.legend()

# Mostrar el gráfico
plt.show()

###################################################################################################
# ejemplo: tabla de valores recomendados de dosis, dado la edad y la eficacia deseada
####################################################################################################

import numpy as np

# Suponemos que ya tenemos el modelo entrenado (modelo = LinearRegression())

# Paso 1: Obtener los coeficientes y el intercepto del modelo
coef_dosis = modelo.coef_[0]
coef_edad = modelo.coef_[1]
intercepto = modelo.intercept_

# Paso 2: Definir una función para calcular la dosis óptima
def calcular_dosis_optima(edad, eficacia_objetivo):
    # La ecuación del modelo es: eficacia = intercepto + coef_dosis * dosis + coef_edad * edad
    # Despejamos la dosis:
    dosis_optima = (eficacia_objetivo - intercepto - coef_edad * edad) / coef_dosis
    return max(0, dosis_optima)  # Aseguramos que la dosis no sea negativa

# Paso 3: Ejemplo de uso
edad_paciente = 45
eficacia_deseada = 0.8  # Supongamos que queremos una eficacia del 80%

dosis_recomendada = calcular_dosis_optima(edad_paciente, eficacia_deseada)

print(f"Para un paciente de {edad_paciente} años, la dosis recomendada para una eficacia del {eficacia_deseada*100}% es: {dosis_recomendada:.2f}")

# Paso 4: Crear una tabla de dosis recomendadas para diferentes edades
edades = range(20, 81, 10)  # Edades de 20 a 80 años, en intervalos de 10
eficacias = [0.7, 0.8, 0.9]  # Diferentes niveles de eficacia

print("\nTabla de dosis recomendadas:")
print("Edad | Eficacia 70% | Eficacia 80% | Eficacia 90%")
print("-" * 50)
for edad in edades:
    dosis = [calcular_dosis_optima(edad, ef) for ef in eficacias]
    print(f"{edad:3d} | {dosis[0]:11.2f} | {dosis[1]:11.2f} | {dosis[2]:11.2f}")
