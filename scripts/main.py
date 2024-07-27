import logging
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from libs.benchmark import Benchmark

log = logging.getLogger(__name__)


@Benchmark("main")
def the_main():
    st.title('Análisis Prediccion utilizando Regresión Polinómica')

    df = pd.read_csv(r"C:\Users\capv2\PycharmProjects\Calidad_Aire\scripts\dataset.csv", sep=";")
    st.write("Datos cargados exitosamente.")

    # Mostrar las primeras filas del DataFrame
    st.write("Primeras filas del DataFrame:")
    st.write(df.head())

    # Seleccionar las características y la variable objetivo
    st.subheader("Selección de variables")
    features = st.multiselect("Selecciona las características:", df.columns.tolist(),
                              default=['SO4', 'MP2.5', 'Tem_amb_med'])
    target = st.selectbox("Selecciona la variable objetivo:", df.columns.tolist(),
                          index=df.columns.get_loc('MP10') if 'MP10' in df.columns else 0)

    if st.button("Realizar análisis"):
        X = df[features]
        y = df[target]

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Escalar las características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Transformar las características para incluir términos polinómicos
        degree = 2  # Grado del polinomio
        poly = PolynomialFeatures(degree)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)

        # Crear y entrenar el modelo de regresión polinómica
        model = LinearRegression()

        # Realizar validación cruzada
        cv_scores = cross_val_score(model, X_train_poly, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_mse_scores = -cv_scores
        cv_rmse_scores = np.sqrt(cv_mse_scores)

        st.write(f'Scores de validación cruzada (RMSE): {cv_rmse_scores}')
        st.write(f'Promedio de RMSE en validación cruzada: {cv_rmse_scores.mean():.4f}')
        st.write(f'Desviación estándar de RMSE en validación cruzada: {cv_rmse_scores.std():.4f}')

        # Ajustar el modelo en los datos de entrenamiento completos
        model.fit(X_train_poly, y_train)

        # Predecir en el conjunto de prueba
        y_pred = model.predict(X_test_poly)

        # Calcular las métricas de evaluación
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.subheader('Evaluación en conjunto de prueba:')
        st.write(f'Error Cuadrático Medio (MSE): {mse:.4f}')
        st.write(f'Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}')
        st.write(f'Coeficiente de Determinación (R²): {r2:.4f}')

if __name__ == '__main__':
    the_main()
