import pandas as pd
import numpy as np
from faker import Faker


# Configuración
np.random.seed(42)
fake = Faker('es_ES')
num_atletas = 200  # 100 velocistas y 100 fondistas


# Generar datos sintéticos
data = {
    'id': range(1, num_atletas + 1),
    'nombre': [fake.first_name() + " " + fake.last_name() for _ in range(num_atletas)],
    'genero': np.random.choice(['M', 'F'], num_atletas, p=[0.7, 0.3]),
    'edad': np.concatenate([
        np.random.normal(24, 3, 100),  # velocistas
        np.random.normal(27, 4, 100)    # fondistas
    ]).astype(int),
    'peso_kg': np.concatenate([
        np.random.normal(75, 5, 95),    # velocistas
        [np.nan]*5,                     # 5 NaN
        np.random.normal(65, 6, 95),    # fondistas
        [120, np.nan, 58, np.nan, 62]  # outliers y NaN
    ]),
    'altura_cm': np.concatenate([
        np.random.normal(180, 5, 100),  # velocistas
        np.random.normal(170, 6, 100)   # fondistas
    ]),
    'pais': [fake.country_code() for _ in range(num_atletas)],
    'tipo': ['velocista']*100 + ['fondista']*100,
    # Tiempos en segundos para velocistas
    '100m': np.concatenate([
        np.random.normal(10.5, 0.5, 98),  # velocistas normales
        [15.8, 9.1],                      # outliers
        [np.nan]*100                       # fondistas (NaN)
    ]),
    '200m': np.concatenate([
        np.random.normal(21.0, 0.7, 98),  # velocistas normales
        [30.5, 18.9],                     # outliers
        [np.nan]*100                      # fondistas (NaN)
    ]),
    '400m': np.concatenate([
        np.random.normal(47.5, 1.2, 98), # velocistas normales
        [60.0, 40.2],                     # outliers
        [np.nan]*100                      # fondistas (NaN)
    ]),
    # Tiempos en minutos para fondistas
    '5km': np.concatenate([
        [np.nan]*100,                     # velocistas (NaN)
        np.random.normal(17.5, 1.5, 98),  # fondistas normales
        [30.0, 12.5]                       # outliers
    ]),
    '10km': np.concatenate([
        [np.nan]*100,                     # velocistas (NaN)
        np.random.normal(36.0, 2.5, 98),  # fondistas normales
        [60.0, 25.0]                       # outliers
    ]),
    'maraton': np.concatenate([
        [np.nan]*100,                     # velocistas (NaN)
        np.random.normal(130, 10, 98),     # fondistas normales (min)
        [300, 100]                         # outliers
    ]),
    'vo2max': np.concatenate([
        np.random.normal(55, 5, 95),      # velocistas
        [np.nan]*5,                       # NaN
        np.random.normal(70, 8, 95),      # fondistas
        [30, np.nan, 90, np.nan, 50]     # outliers y NaN
    ]),
    'frecuencia_entrenamiento': np.random.choice(
        ['diario', '5x_semana', '3x_semana', 'ocasional'],
        num_atletas,
        p=[0.4, 0.3, 0.2, 0.1]
    )
}


# Crear DataFrame
df_atletas = pd.DataFrame(data)


# Añadir algunos NaN aleatorios adicionales
for col in ['edad', 'altura_cm', 'vo2max']:
    df_atletas.loc[df_atletas.sample(10).index, col] = np.nan


# Guardar a CSV
df_atletas.to_csv('datos_atletas.csv', index=False)
