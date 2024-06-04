import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import random
import os 
import plotly.graph_objects as go
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Function to load the trained model with caching
@st.cache_resource
def load_trained_model():
    return load_model('../best_model.keras')

# Function to load validation data and predictions with caching
@st.cache_data
def load_validation_data():
    X_val = np.load('../X_val.npy')
    Y_val = np.load('../Y_val.npy')
    Y_pred = np.load('../Y_pred.npy')
    return X_val, Y_val, Y_pred

# Load the trained model
model = load_trained_model()

# Load validation data and predictions
X_val, Y_val, Y_pred = load_validation_data()

# Sidebar for navigation

st.sidebar.title("Navegación")
st.sidebar.markdown("Usa las opciones de abajo para navegar entre secciones.")
section = st.sidebar.radio("Section", ["Descripción", "Modelado de datos y categorías", "Rendimiento del modelo", "Matriz de Confusión y Reporte de métricas"])

# Project title and description
if section == "Descripción":
    st.title("Modelo de IA para Deteccion de cáncer colorrectal")
    st.write("""
    Esta aplicación utiliza un modelo de aprendizaje profundo para ayudar en la detección de cáncer de colon 
    a partir de imágenes histopatológicas. El modelo se basa en InceptionV3, pre-entrenado en el conjunto de datos
    ImageNet, y se ha afinado utilizando técnicas de aprendizaje por transferencia para adaptarse a las especificidades de la 
    clasificación del tejido del colon. El conjunto de datos utilizado para el entrenamiento comprende imágenes etiquetadas como 
    benignas o malignas, proporcionando al modelo los ejemplos necesarios para distinguir entre células sanas y cancerosas. Además,
    la aplicación ofrece información detallada sobre el rendimiento del modelo, incluyendo precisión, métricas de pérdida y matriz 
    de confusión, que ayudan a entender la eficacia y fiabilidad del modelo.""")

if section == "Modelado de datos y categorías":
    st.header("Categorías")
    st.write("""
        Hay un total de 5000 imágenes para cada categorías, y en este apartado se muestran 6 imágenes aleatorias del set de datos para la categoría correspondiente.""")
    # Path to the images
    path = '../lung_colon_image_set/colon_image_sets'
    classes = ['colon_aca', 'colon_n']

    for cat in classes:
        image_dir = f'{path}/{cat}'
        images = os.listdir(image_dir)
        
        if cat == 'colon_aca':
            st.subheader('Tumor Maligno')
            
        elif cat == 'colon_n':
            st.subheader('Tumor Benigno')

        fig, ax = plt.subplots(2, 3, figsize=(10, 7))
        fig.patch.set_alpha(0)  # Remove the figure background

        for i in range(6):
            k = random.randint(0, len(images) - 1)
            img = np.array(Image.open(f'{path}/{cat}/{images[k]}'))
            ax[i//3, i%3].imshow(img)
            ax[i//3, i%3].axis('off')
        
        # Hide the last empty subplot
        ax[1, 2].axis('off')
        
        st.pyplot(fig)
    
    st.header("División de datos")

    # Datos para las gráficas de pastel con números totales de imágenes
    data_cancer = {'Type': ['Training', 'Validation'], 'Count': [5000 - 979, 979]}
    data_benign = {'Type': ['Training', 'Validation'], 'Count': [5000 - 1021, 1021]}

    df_cancer = pd.DataFrame(data_cancer)
    df_benign = pd.DataFrame(data_benign)

    # Función para calcular los porcentajes y generar etiquetas personalizadas
    def make_labels(data):
        total = sum(data['Count'])
        labels = []
        for i, row in data.iterrows():
            percent = (row['Count'] / total) * 100
            labels.append(f"{row['Type']}<br>{row['Count']} ({percent:.1f}%)")
        return labels

    labels_cancer = make_labels(df_cancer)
    labels_benign = make_labels(df_benign)

    fig_cancer = px.pie(df_cancer, values='Count', names='Type', title='Distribución de imágenes - Cancer', 
                        color_discrete_sequence=px.colors.sequential.Reds)
    fig_cancer.update_traces(textposition='inside', textinfo='label+percent', hoverinfo='label+percent+value', 
                             text=labels_cancer)
    fig_cancer.update_layout(showlegend=True, margin=dict(t=200, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')

    fig_benign = px.pie(df_benign, values='Count', names='Type', title='Distribución de imágenes - Benigno', 
                        color_discrete_sequence=px.colors.sequential.Blues)
    fig_benign.update_traces(textposition='inside', textinfo='label+percent', hoverinfo='label+percent+value', 
                             text=labels_benign)
    fig_benign.update_layout(showlegend=True, margin=dict(t=200, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_cancer)
    with col2:
        st.plotly_chart(fig_benign)

    # Descripción de los gráficos
    st.write("""
    Los gráficos de pastel muestran la distribución de las imágenes de entrenamiento y validación utilizadas en el modelo.
    Como podemos observar, la categoría cancer tiene 979 imágenes de validación y 4021 imágenes de entrenamiento, 
    mientras que la categoría benigno tiene 1021 imágenes de validación y 3979 imágenes de entrenamiento.
    """)

# Function to load training history with caching
@st.cache_data
def load_training_history():
    return pd.read_csv('../history.csv')

# Section for model performance
if section == "Rendimiento del modelo":
    st.header("Rendimiento del modelo")

    # Load training history
    history = load_training_history()

    st.subheader("Loss")
    fig_loss = px.line(history, x=history.index, y=['loss', 'val_loss'], labels={'index': 'Epoch', 'value': 'Loss'}, 
                       title='Training and Validation Loss')
    fig_loss.update_layout(showlegend=True, margin=dict(t=50, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')

    st.plotly_chart(fig_loss)

    st.subheader("Accuracy")
    fig_accuracy = px.line(history, x=history.index, y=['accuracy', 'val_accuracy'], labels={'index': 'Epoch', 'value': 'Accuracy'}, 
                           title='Training and Validation Accuracy')
    fig_accuracy.update_layout(showlegend=True, margin=dict(t=50, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')

    st.plotly_chart(fig_accuracy)

# Section for confusion matrix and classification report
if section == "Matriz de Confusión y Reporte de métricas":
    st.header("Matriz de Confusión y Reporte de métricas")

    # Confusion matrix
    cm = confusion_matrix(Y_val, Y_pred)
    
    # Create a DataFrame for the confusion matrix
    cm_df = pd.DataFrame(cm, index=['Malignant', 'Benign'], columns=['Predicted Malignant', 'Predicted Benign'])
    
    # Create a Plotly figure for the confusion matrix
    fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale='Reds')
    fig_cm.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig_cm.update_traces(hovertemplate='True Label: %{y}<br>Predicted Label: %{x}<br>Numero de predicciones: %{z}<extra></extra>')


    st.plotly_chart(fig_cm)
    st.write("""
    Para determinar el rendimiento del modelo, nos basaremos en las siguientes métricas: exactitud, precisión, recuperación y puntuación F1. Cada métrica cumple una función única:

    1. Precisión: Refleja el total de predicciones correctas (tanto resultados positivos como negativos) en relación con todos los casos considerados. 
    Por ejemplo, 
    si un modelo identifica con precisión 90 de 100 correos electrónicos, su precisión es del 90%.        
    2. Recall: Es la proporción de verdaderos positivos de todos los casos positivos reales, es decir, la exactitud con la que acertó las predicciones de validación. Por ejemplo, si hay 50 correos electrónicos no deseados y el modelo detecta 40 de ellos, la tasa el recall sería del 80%.
    3. Puntuación F1: es una métrica que logra un equilibrio entre precisión y recuperación mediante el uso de su media armónica.
    4. Soporte: Es una medida del número de muestras de cada clase contenidas en el conjunto de datos, en este caso, el soporte son las imágenes de validación usados en cada categoría
        """)
        
    # Generar un reporte de clasificación
    report = classification_report(Y_val, Y_pred, target_names=['Malignant', 'Benign'], output_dict=True)
    report_df = pd.DataFrame(report).transpose().reset_index()

    # Limitar los decimales a 2
    report_df = report_df.round(2)

    # Reemplazar los valores que deseas dejar en blanco
    report_df.loc[report_df['index'] == 'accuracy', ['precision', 'recall']] = ''
    report_df.loc[report_df['index'] == 'accuracy', ['support']] = '2000'
    report_df.loc[report_df['index'] == 'index', ['index']] = ''

    # Crear una tabla con Plotly
    fig_table = go.Figure(data=[go.Table(
        header=dict(values=list(report_df.columns),
                    fill_color='rgba(14,17,23,255)',
                    line_color='white',
                    line_width=0.3,
                    align='center',
                    font=dict(color='white', size=16),
                    height=35),  # Reducir el tamaño de las celdas de la cabecera
        cells=dict(values=[report_df[col] for col in report_df.columns],
                   fill_color='rgba(14,17,23,255)',
                   line_color='white',
                   line_width=0.15,
                   align='center',
                   font=dict(color='white', size=16),
                   height=35))
                     # Reducir el tamaño de las celdas de los datos
    ])

    fig_table.update_layout(
        title='Classification Report',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig_table)
    st.markdown("""
        <div style="text-align: center; margin-top: 100px; padding-bottom: 10px;">
            <img src="https://seeklogo.com/images/U/uninorte-logo-5211DDF859-seeklogo.com.png" alt="Imagen" width="100">
            <p style="margin-top: 10px;">
                Juan Céspedes, Miguel Marsiglia, Juan Jiménez
            </p>
        </div>
        """, unsafe_allow_html=True)