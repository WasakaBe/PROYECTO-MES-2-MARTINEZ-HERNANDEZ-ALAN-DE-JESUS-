from flask import Flask, request, render_template, jsonify
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Ruta completa del archivo CSV
csv_path = 'energy_efficiency_data.csv'

# Intentar cargar datos desde el CSV
try:
    data = pd.read_csv(csv_path)
    app.logger.debug('Datos cargados correctamente.')
except Exception as e:
    app.logger.error(f'Error al cargar los datos: {str(e)}')
    data = None

# Verifica que los datos fueron cargados correctamente
if data is not None:
    # Selección de las mejores características
    selected_features = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 
                         'Glazing_Area', 'Glazing_Area_Distribution', 'Heating_Load']
    X = data[selected_features]
    y = data['Cooling_Load']
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    app.logger.debug('Modelo entrenado correctamente.')
else:
    model = None

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no disponible'}), 500

    try:
        # Obtener los datos enviados en el request
        relative_compactness = float(request.form['relative_compactness'])
        surface_area = float(request.form['surface_area'])
        wall_area = float(request.form['wall_area'])
        glazing_area = float(request.form['glazing_area'])
        glazing_area_distribution = float(request.form['glazing_area_distribution'])
        heating_load = float(request.form['heating_load'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[relative_compactness, surface_area, wall_area, 
                                 glazing_area, glazing_area_distribution, heating_load]],
                               columns=['Relative_Compactness', 'Surface_Area', 'Wall_Area', 
                                        'Glazing_Area', 'Glazing_Area_Distribution', 'Heating_Load'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'cooling_load': prediction[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
