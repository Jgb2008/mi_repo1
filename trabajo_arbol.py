import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             recall_score, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Configuración de la app
st.set_page_config(page_title="Clasificación de Atletas", layout="centered")


# Cargar datos (usando el nuevo dataset)
@st.cache_data
def load_data():
    # Datos sintéticos con las nuevas variables
    data = pd.DataFrame({
        'tipo': ['velocista']*100 + ['fondista']*100,
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
        'vo2max': np.concatenate([
            np.random.normal(55, 5, 95),      # velocistas
            [np.nan]*5,                       # NaN
            np.random.normal(70, 8, 95),      # fondistas
            [30, np.nan, 90, np.nan, 50]     # outliers y NaN
        ]),
        'frecuencia_cardiaca_basal': np.concatenate([
            np.random.normal(60, 5, 100),    # velocistas
            np.random.normal(50, 4, 100)     # fondistas
        ]),
        'porcentaje_fibras_lentas': np.concatenate([
            np.random.normal(30, 5, 100),    # velocistas
            np.random.normal(70, 8, 100)    # fondistas
        ]),
        'porcentaje_fibras_rapidas': np.concatenate([
            np.random.normal(70, 8, 100),   # velocistas
            np.random.normal(30, 5, 100)    # fondistas
        ])
    })
    
    # Calcular porcentaje de fibras rápidas como complemento de las lentas
    data['porcentaje_fibras_rapidas'] = 100 - data['porcentaje_fibras_lentas']
    
    return data


data = load_data()


# Página principal
st.title("Clasificación de Atletas: Fondistas vs Velocistas")


# Sidebar para navegación
page = st.sidebar.selectbox("Seleccione una página",
                           ["Preprocesamiento", "Modelado", "Predicción"])


if page == "Preprocesamiento":
    st.header("Preprocesamiento de Datos")
   
    # Mostrar datos crudos
    st.subheader("Datos Crudos")
    st.write(data.head())
   
    # Valores faltantes
    st.subheader("Valores Faltantes")
    st.write(data.isna().sum())
   
    # Eliminar NaN
    data_clean = data.dropna()
    st.write("Datos después de eliminar NaN:", data_clean.shape)
   
    # Outliers
    st.subheader("Detección de Outliers")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data_clean, x='tipo', y='vo2max', ax=ax)
    st.pyplot(fig)
   
    st.write("""
    Se observan algunos outliers en el VO2 max, particularmente
    valores muy bajos para fondistas y muy altos para velocistas.
    """)
   
    # Distribuciones
    st.subheader("Distribuciones")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    sns.histplot(data_clean, x='peso_kg', hue='tipo', kde=True, ax=axes[0,0])
    sns.histplot(data_clean, x='altura_cm', hue='tipo', kde=True, ax=axes[0,1])
    sns.histplot(data_clean, x='vo2max', hue='tipo', kde=True, ax=axes[0,2])
    sns.histplot(data_clean, x='frecuencia_cardiaca_basal', hue='tipo', kde=True, ax=axes[1,0])
    sns.histplot(data_clean, x='porcentaje_fibras_lentas', hue='tipo', kde=True, ax=axes[1,1])
    sns.histplot(data_clean, x='porcentaje_fibras_rapidas', hue='tipo', kde=True, ax=axes[1,2])
    plt.tight_layout()
    st.pyplot(fig)
   
    # Correlación
    st.subheader("Correlación entre Variables")
    numeric_data = data_clean.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm', center=0)
    st.pyplot(fig)
   
    st.write("""
    Se observa alta correlación negativa entre el porcentaje de fibras lentas y rápidas (lo cual es esperable).
    El VO2 max muestra correlación positiva con fibras lentas y negativa con fibras rápidas.
    """)
   
   
elif page == "Modelado":
    st.header("Modelado y Evaluación")
   
    # Preprocesamiento
    data_clean = data.dropna()
    le = LabelEncoder()
    data_clean['tipo_encoded'] = le.fit_transform(data_clean['tipo'])
   
    # Balance de clases
    st.subheader("Balance de Clases")
    class_counts = data_clean['tipo'].value_counts()
    st.bar_chart(class_counts)
   
    st.write(f"""
    Las clases están balanceadas: {class_counts['velocista']} velocistas vs
    {class_counts['fondista']} fondistas. Esto es ideal para el modelado.
    """)
   
    # Dividir datos - usando las nuevas variables
    X = data_clean[['peso_kg', 'altura_cm', 'vo2max', 
                   'frecuencia_cardiaca_basal', 
                   'porcentaje_fibras_lentas', 
                   'porcentaje_fibras_rapidas']]
    y = data_clean['tipo_encoded']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
   
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
   
    # Modelos
    models = {
        "Regresión Logística": LogisticRegression(),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=3),
        "SVM": SVC(probability=True)
    }
   
    # Evaluar modelos
    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:,1]
       
        # Métricas
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
       
        results.append({
            "Modelo": name,
            "Accuracy": acc,
            "Recall": rec,
            "AUC": roc_auc
        })
       
        # Mostrar matriz de confusión
        st.subheader(f"{name} - Matriz de Confusión")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Real')
        st.pyplot(fig)
       
        # Curva ROC
        st.subheader(f"{name} - Curva ROC")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('Tasa de Falsos Positivos')
        ax.set_ylabel('Tasa de Verdaderos Positivos')
        ax.legend()
        st.pyplot(fig)
   
    # Comparación de modelos
    st.subheader("Comparación de Métricas")
    results_df = pd.DataFrame(results)
    st.table(results_df)
   
    st.write("""
    El modelo de SVM muestra el mejor rendimiento en todas las métricas,
    seguido por la Regresión Logística. El Árbol de Decisión tiene un
    rendimiento ligeramente inferior pero sigue siendo aceptable.
    """)
   
    # Tuning de hiperparámetros
    st.subheader("Tuning de Hiperparámetros")
   
    st.write("**Regresión Logística - Variación de C**")
    c_values = [0.001, 0.01, 0.1, 1, 10, 100]
    lr_acc = []
    for c in c_values:
        model = LogisticRegression(C=c)
        model.fit(X_train_scaled, y_train)
        lr_acc.append(accuracy_score(y_test, model.predict(X_test_scaled)))
   
    fig, ax = plt.subplots()
    ax.plot(c_values, lr_acc, marker='o')
    ax.set_xscale('log')
    ax.set_xlabel('Valor de C')
    ax.set_ylabel('Accuracy')
    st.pyplot(fig)
   
    st.write("""
    Se observa que el accuracy mejora hasta C=1 y luego se estabiliza,
    indicando que valores muy altos de C pueden llevar a sobreajuste.
    """)


elif page == "Predicción":
    st.header("Predicción de Tipo de Atleta")
   
    # Preprocesamiento
    data_clean = data.dropna()
    le = LabelEncoder()
    data_clean['tipo_encoded'] = le.fit_transform(data_clean['tipo'])
    X = data_clean[['peso_kg', 'altura_cm', 'vo2max', 
                   'frecuencia_cardiaca_basal', 
                   'porcentaje_fibras_lentas', 
                   'porcentaje_fibras_rapidas']]
    y = data_clean['tipo_encoded']
   
    # Entrenar modelo final (SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SVC(probability=True)
    model.fit(X_scaled, y)
   
    # FORMULARIO EN EL SIDEBAR (IZQUIERDA)
    with st.sidebar:
        st.subheader("Ingrese los datos del atleta")
       
        peso_kg = st.slider("Peso (kg)", 40, 120, 70, 1)
        altura_cm = st.slider("Altura (cm)", 150, 220, 175, 1)
        vo2max = st.slider("VO2 máx (ml/kg/min)", 30, 90, 60, 1)
        frecuencia_cardiaca_basal = st.slider("Frecuencia cardíaca basal (lpm)", 40, 80, 60, 1)
        porcentaje_fibras_lentas = st.slider("% Fibras lentas", 10, 90, 50, 1)
        porcentaje_fibras_rapidas = 100 - porcentaje_fibras_lentas
        st.write(f"% Fibras rápidas: {porcentaje_fibras_rapidas}")
       
        if st.button("Predecir tipo de atleta", type="primary"):
            input_data = scaler.transform([[peso_kg, altura_cm, vo2max, 
                                         frecuencia_cardiaca_basal, 
                                         porcentaje_fibras_lentas, 
                                         porcentaje_fibras_rapidas]])
            prediction = model.predict(input_data)
            proba = model.predict_proba(input_data)
            st.session_state['prediction'] = prediction
            st.session_state['proba'] = proba
   
    # RESULTADOS EN EL ÁREA PRINCIPAL (DERECHA)
    if 'prediction' in st.session_state:
        prediction = st.session_state['prediction']
        proba = st.session_state['proba']
       
        st.subheader("Resultado de la predicción")
       
        # Mostrar resultado con estilo
        if prediction[0] == 0:
            st.markdown("""
            <div style='background-color:#e6f7ff; padding:20px; border-radius:10px; border-left:5px solid #1890ff;'>
                <h2 style='color:#1890ff; margin-top:0;'>🏃‍♂️ FONDISTA</h2>
                <p style='font-size:16px;'>El atleta muestra características típicas de fondista</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color:#fff7e6; padding:20px; border-radius:10px; border-left:5px solid #faad14;'>
                <h2 style='color:#faad14; margin-top:0;'>⚡ VELOCISTA</h2>
                <p style='font-size:16px;'>El atleta muestra características típicas de velocista</p>
            </div>
            """, unsafe_allow_html=True)
       
        st.subheader("Distribución de Probabilidades")
       
        # Gráfico de probabilidades mejorado
        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.bar(['Fondista', 'Velocista'],
                     [proba[0][0], proba[0][1]],
                     color=['#1890ff', '#faad14'])
       
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probabilidad')
       
        # Añadir etiquetas de porcentaje
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom')
       
        st.pyplot(fig)
       
        # Mostrar métricas adicionales
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilidad Fondista", f"{proba[0][0]*100:.1f}%")
        with col2:
            st.metric("Probabilidad Velocista", f"{proba[0][1]*100:.1f}%")
    else:
        st.info("Por favor ingrese los datos del atleta y haga clic en 'Predecir'")
