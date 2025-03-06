####################################################################################################
# Victor Peral - 2025 
# ML model to predict the price given 3 scenarios of competition (severe, mid,low)
# index.html is generated to plot the 3d graph with the price prediction graph real and predicted values
####################################################################################################

from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
import json
import numpy as np
import base64
import io

app = Flask(__name__, template_folder='templates')

# Generar datos de ejemplo
np.random.seed(42)
n_samples = 100
demanda = np.random.rand(n_samples) * 100
competencia_baja = np.random.rand(n_samples) * 30
competencia_moderada = np.random.rand(n_samples) * 60
competencia_alta = np.random.rand(n_samples) * 90
precio = 50 + 2 * demanda - 0.5 * competencia_moderada + np.random.randn(n_samples) * 10

# Crear y entrenar el modelo
X = pd.DataFrame({'demanda': demanda, 'competencia': competencia_moderada})
y = precio
model = LinearRegression()
model.fit(X, y)


def decode_plotly_data(fig):
    """Decodes base64 encoded data in a Plotly figure."""
    for trace in fig['data']:
        for key in ['x', 'y', 'z']:
            if isinstance(trace[key], dict) and 'bdata' in trace[key]:
                encoded_data = trace[key]['bdata']
                decoded_bytes = base64.b64decode(encoded_data)
                buffer = io.BytesIO(decoded_bytes)
                dtype = trace[key]['dtype']
                data_array = np.frombuffer(buffer.read(), dtype=dtype)
                trace[key] = data_array.tolist() # Convert NumPy array to list

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        escenario = ['escenario']
        if escenario == 'baja':
            competencia = competencia_baja
        elif escenario == 'alta':
            competencia = competencia_alta
        else:
            competencia = competencia_moderada
    else:
        escenario = 'moderada'
        competencia = competencia_moderada

    # Realizar predicciones
    X_pred = pd.DataFrame({'demanda': demanda, 'competencia': competencia})
    y_pred = model.predict(X_pred)

    # Crear gráfica 3D
    trace = go.Scatter3d(
        x=demanda,
        y=competencia,
        z=y_pred,
        mode='markers',
        marker=dict(size=5, color=y_pred, colorscale='Viridis', opacity=0.8),
    )

    layout = go.Layout(
        scene=dict(
            xaxis_title='Demanda',
            yaxis_title='Competencia',
            zaxis_title='Precio predicho'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=[trace], layout=layout)

    # Decode the base64 data BEFORE converting to JSON
    fig_dict = fig.to_dict()  # Convert the Figure object to a dictionary
    decode_plotly_data(fig_dict)
    graphJSON = json.dumps(fig_dict, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html', graphJSON=graphJSON, escenario=escenario)

if __name__ == '__main__':
    app.run(debug=True)
