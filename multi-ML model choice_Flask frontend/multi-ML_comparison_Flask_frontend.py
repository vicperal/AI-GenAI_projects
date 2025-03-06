####################################################################################################
# Victor Peral - 2025 
# multi-ML choice Flask application, usage of http://api.worldbank.org public data for model training
# Server and Frontend code 
# index.html is generated to plot the 3d graph with real and predicted values. It also compares the 
# predicted model performance
####################################################################################################

from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
import json
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__, template_folder='templates')

# Cargar datos del Banco Mundial
def load_world_bank_data():
    indicators = {
        'NY.GDP.PCAP.CD': 'PIB per cápita',
        'SH.XPD.CHEX.GD.ZS': 'Gasto en salud',
        'SE.XPD.TOTL.GD.ZS': 'Gasto en educación',
        'SP.DYN.LE00.IN': 'Esperanza de vida'
    }

    data = {}
    for indicator, name in indicators.items():
        url = f"http://api.worldbank.org/v2/countries/all/indicators/{indicator}?date=2015:2020&format=json&per_page=1000"
        response = requests.get(url)
        data[name] = pd.json_normalize(response.json()[1])

    df = pd.DataFrame({
        'País': data['PIB per cápita']['country.value'],
        'PIB per cápita': data['PIB per cápita']['value'].astype(float),
        'Gasto en salud': data['Gasto en salud']['value'].astype(float),
        'Gasto en educación': data['Gasto en educación']['value'].astype(float),
        'Esperanza de vida': data['Esperanza de vida']['value'].astype(float)
    }).dropna()

    return df

# Cargar datos y entrenar los modelos
df = load_world_bank_data()
X = df[['PIB per cápita', 'Gasto en salud', 'Gasto en educación']]
y = df['Esperanza de vida']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos para el modelo de Deep Learning
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelos
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Calcular métricas de rendimiento
def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

rf_mse, rf_r2 = calculate_metrics(rf_model, X_test, y_test)
gb_mse, gb_r2 = calculate_metrics(gb_model, X_test, y_test)
lr_mse, lr_r2 = calculate_metrics(lr_model, X_test, y_test)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pib = float(request.form['pib'])
        salud = float(request.form['salud'])
        educacion = float(request.form['educacion'])
    else:
        pib = X['PIB per cápita'].median()
        salud = X['Gasto en salud'].median()
        educacion = X['Gasto en educación'].median()

    # Realizar predicciones
    input_data = pd.DataFrame([[pib, salud, educacion]], columns=['PIB per cápita', 'Gasto en salud', 'Gasto en educación'])
    rf_prediction = rf_model.predict(input_data)[0]
    gb_prediction = gb_model.predict(input_data)[0]
    lr_prediction = lr_model.predict(input_data)[0]

    # Crear gráfica 3D para datos reales
    trace_real = go.Scatter3d(
        x=X['PIB per cápita'].tolist(),
        y=X['Gasto en salud'].tolist(),
        z=X['Gasto en educación'].tolist(),
        mode='markers',
        marker=dict(
            size=5,
            color=df['Esperanza de vida'].tolist(),
            colorscale='Viridis',
            cmin=df['Esperanza de vida'].min(),
            cmax=df['Esperanza de vida'].max(),
            colorbar=dict(
                title='Esperanza de vida real',
                len=0.5,
                yanchor='top',
                y=0.99,
                x=1.0,
                thickness=20,
                title_text='Esperanza de vida real',
                title_side='right'
            ),
        ),
        text=[f"País: {país}<br>PIB per cápita: {pib:.2f}<br>Gasto en salud: {salud:.2f}%<br>Gasto en educación: {edu:.2f}%<br>Esperanza de vida real: {vida:.2f}"
            for país, pib, salud, edu, vida in zip(df['País'], X['PIB per cápita'], X['Gasto en salud'], X['Gasto en educación'], df['Esperanza de vida'])],
        hoverinfo='text',
        name='Datos reales'
    )

    # Puntos para las predicciones
    prediction_points = [
        go.Scatter3d(
            x=[pib], y=[salud], z=[educacion],
            mode='markers',
            marker=dict(size=10, color=color, symbol='diamond'),
            name=f'Predicción {name}',
            text=[f"{name}<br>PIB per cápita: {pib:.2f}<br>Gasto en salud: {salud:.2f}%<br>Gasto en educación: {educacion:.2f}%<br>Esperanza de vida predicha: {pred:.2f}"],
            hoverinfo='text'
        ) for name, color, pred in [
            ('Random Forest', 'red', rf_prediction),
            ('Gradient Boosting', 'blue', gb_prediction),
            ('Regresión Lineal', 'green', lr_prediction)
        ]
    ]

    layout = go.Layout(
        scene=dict(
            xaxis_title='PIB per cápita',
            yaxis_title='Gasto en salud (% del PIB)',
            zaxis_title='Gasto en educación (% del PIB)'
        ),
        margin=dict(l=0, r=150, b=0, t=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    fig = go.Figure(data=[trace_real] + prediction_points, layout=layout)
    graphJSON = json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',
                           graphJSON=graphJSON,
                           rf_prediction=rf_prediction,
                           gb_prediction=gb_prediction,
                           lr_prediction=lr_prediction,
                           pib=pib, salud=salud, educacion=educacion,
                           rf_mse=rf_mse, rf_r2=rf_r2,
                           gb_mse=gb_mse, gb_r2=gb_r2,
                           lr_mse=lr_mse, lr_r2=lr_r2)

if __name__ == '__main__':
    app.run(debug=True)
