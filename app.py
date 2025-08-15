import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

# 🔧 Configuração da página
st.set_page_config(page_title="Classificador Carro vs Moto", layout="wide")

# 🔹 Carregar modelo
@st.cache_resource
def carregar_modelo():
    return tf.keras.models.load_model("modelo/modelo_final.keras")

model = carregar_modelo()
class_names = ["carro", "moto"]

# 🔹 Sidebar de navegação
aba = st.sidebar.radio("📂 Navegação", ["Upload", "Inferência", "Treinamento", "Matriz de Confusão", "Download"])

# 🔹 Aba: Upload
if aba == "Upload":
    st.title("📤 Upload de Imagem")
    uploaded_file = st.file_uploader("Envie uma imagem para classificar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagem enviada", use_column_width=True)

        img = image.resize((224, 224))
        img_array = np.array(img) / 127.5 - 1
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        classe = class_names[np.argmax(pred)]
        confianca = np.max(pred)

        st.markdown(f"### 🧠 Previsão: **{classe.upper()}**")
        st.markdown(f"Confiabilidade: `{confianca:.2%}`")
        st.progress(float(confianca))

# 🔹 Aba: Inferência em lote (opcional)
elif aba == "Inferência":
    st.title("🔍 Inferência em lote")
    st.info("Em breve: upload de múltiplas imagens para classificação.")

# 🔹 Aba: Treinamento
elif aba == "Treinamento":
    st.title("📊 Gráficos de Treinamento")

    try:
        history = np.load("resultados/history.npy", allow_pickle=True).item()
        acc = history['accuracy']
        val_acc = history['val_accuracy']
        loss = history['loss']
        val_loss = history['val_loss']
        epochs_range = range(len(acc))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Acurácia por Época")
            fig, ax = plt.subplots()
            ax.plot(epochs_range, acc, label="Treino")
            ax.plot(epochs_range, val_acc, label="Validação")
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("Perda por Época")
            fig, ax = plt.subplots()
            ax.plot(epochs_range, loss, label="Treino")
            ax.plot(epochs_range, val_loss, label="Validação")
            ax.legend()
            st.pyplot(fig)

    except FileNotFoundError:
        st.warning("Arquivo de histórico não encontrado. Salve como 'resultados/history.npy'.")

# 🔹 Aba: Matriz de Confusão
elif aba == "Matriz de Confusão":
    st.title("📉 Matriz de Confusão")

    try:
        y_true = np.load("resultados/y_true.npy")
        y_pred = np.load("resultados/y_pred.npy")

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        st.pyplot(fig)

        st.subheader("📋 Métricas por Classe")
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        df_report = pd.DataFrame(report).iloc[:-1, :].T
        st.dataframe(df_report.style.highlight_max(axis=0))

    except FileNotFoundError:
        st.warning("Arquivos 'y_true.npy' e 'y_pred.npy' não encontrados.")

# 🔹 Aba: Download
elif aba == "Download":
    st.title("💾 Download do Modelo")
    with open("modelo/modelo_final.keras", "rb") as f:
        st.download_button("📥 Baixar modelo (.keras)", f, file_name="modelo_final.keras")

    try:
        with open("modelo_mobile.tflite", "rb") as f:
            st.download_button("📱 Baixar modelo (.tflite)", f, file_name="modelo_mobile.tflite")
    except FileNotFoundError:
        st.info("Modelo .tflite ainda não gerado.")
