from flask import Flask, render_template
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Ruta del dataset
DATASET_PATH = 'ferreteria_dataset_large.csv'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pca')
def pca_model():
    # Cargar el dataset
    df = pd.read_csv(DATASET_PATH)

    # Preprocesamiento de datos
    df['Genero'] = df['Genero'].map({'F': 0, 'M': 1})
    features = df[['Edad', 'Genero', 'Precio', 'Cantidad', 'Dias_Desde_Ultima_Compra', 'Total_Compras', 'Descuento_Aplicado']]

    # Aplicar PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features)
    
    # Crear DataFrame para los componentes principales
    pca_df = pd.DataFrame(data=principal_components, columns=['Componente_1', 'Componente_2'])
    pca_df['Compra_Futura'] = df['Compra_Futura']

    # Graficar los componentes principales
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Componente_1', y='Componente_2', hue='Compra_Futura', data=pca_df, palette='viridis')
    plt.title('PCA de Compras en Ferretería')
    plt.savefig('static/pca_plot.png')
    
    return render_template('pca.html', pca_image='static/pca_plot.png')

@app.route('/multi-armed-bandit')
def multi_armed_bandit():
    # Simulación del problema del multi-brazo (multi-armed bandit)
    np.random.seed(42)
    n_arms = 5
    n_trials = 1000
    rewards = np.zeros(n_arms)
    counts = np.zeros(n_arms)

    # Generar recompensas aleatorias para cada brazo
    true_rewards = np.random.rand(n_arms)

    def select_arm(epsilon):
        if np.random.rand() > epsilon:
            return np.argmax(rewards / (counts + 1e-5))  # Selección de brazo con mayor recompensa
        else:
            return np.random.randint(0, n_arms)  # Selección aleatoria (exploración)

    epsilon = 0.1  # Factor de exploración
    history = []

    for _ in range(n_trials):
        arm = select_arm(epsilon)
        reward = np.random.binomial(1, true_rewards[arm])
        rewards[arm] += reward
        counts[arm] += 1
        history.append((arm, reward))

    # Graficar los resultados del multi-brazo
    arms = range(n_arms)
    plt.figure(figsize=(10, 6))
    plt.bar(arms, rewards / (counts + 1e-5))
    plt.xlabel('Brazo')
    plt.ylabel('Recompensa Promedio')
    plt.title('Resultados del Multi-Armed Bandit')
    plt.savefig('static/multi_armed_bandit_plot.png')
    
    return render_template('multi_armed_bandit.html', bandit_image='static/multi_armed_bandit_plot.png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
