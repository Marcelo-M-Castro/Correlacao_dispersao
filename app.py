import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from PIL import Image

# Configuração da página
st.set_page_config(page_title="Análise de Performance", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Análise de Performance dos Agentes")

# Exibe imagem de orientação antes do upload
image = Image.open("orientacao.png")  # imagem na raiz do projeto
st.markdown(
    """
    <div style="text-align: center;">
    """,
    unsafe_allow_html=True
)
st.image(image, caption="Imagem de orientação", use_column_width=True)
st.markdown(
    """
    </div>
    """,
    unsafe_allow_html=True
)

# Upload do arquivo
uploaded_file = st.file_uploader("📂 Faça upload da planilha Excel (.xlsx)", type="xlsx")

if uploaded_file:
    # Leitura do Excel
    df = pd.read_excel(uploaded_file)
    st.subheader("Prévia dos dados originais")
    st.dataframe(df.head())

    # Remove colunas
    columns_to_drop = ['Mês', 'Líder', 'Agente Email','TMA', 'TMR','Nivel de Serviço (%)','RESOLUTIVIDADE (%)']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    st.success(f"Colunas removidas: {columns_to_drop}")

    # Correlação
    st.subheader("🔗 Mapa de Calor - Correlação")
    corr = df.corr(numeric_only=True)
    fig_corr, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig_corr)

    # Limpeza adicional
    df_original = pd.read_excel(uploaded_file)
    columns_to_clean = ['NS', '% Resolvidos', '% CSAT ', '% TMPR', 'Qualidade']
    for col in columns_to_clean:
        if col in df_original.columns:
            df_original[col] = df_original[col].astype(str).str.replace(',', '.')
            df_original[col] = pd.to_numeric(df_original[col], errors='coerce')

    for col in ['Entrantes', 'Atendidos']:
        if col in df_original.columns:
            df_original[col] = df_original[col].astype(str).str.replace(',', '')
            df_original[col] = pd.to_numeric(df_original[col], errors='coerce').astype('Int64')

    df_cleaned = df_original.dropna(subset=['ATENDIDOS', 'Qualidade (%)']).copy()

    # K-Means
    st.subheader("🔵 Agrupamento (K-Means)")
    if 'ATENDIDOS' in df_cleaned.columns and 'Qualidade (%)' in df_cleaned.columns:
        X_cluster = df_cleaned[['ATENDIDOS', 'Qualidade (%)']]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_cleaned['cluster'] = kmeans.fit_predict(X_cluster)
        st.dataframe(df_cleaned[['ATENDIDOS', 'Qualidade (%)', 'cluster']].head())

    # Função para gráficos com regressão
    def regressao_plot(x_col, y_col, hover_cols, titulo):
        if x_col in df_cleaned.columns and y_col in df_cleaned.columns:
            x = df_cleaned[x_col]
            y = df_cleaned[y_col]
            slope, intercept = np.polyfit(x, y, 1)
            linha_x = np.array([x.min(), x.max()])
            linha_y = slope * linha_x + intercept

            fig = px.scatter(
                df_cleaned,
                x=x_col,
                y=y_col,
                color='Qualidade (%)' if 'Qualidade (%)' in df_cleaned.columns else None,
                color_continuous_scale='RdYlGn',
                hover_data=hover_cols,
                title=titulo
            )
            fig.add_trace(go.Scatter(
                x=linha_x,
                y=linha_y,
                mode='lines',
                name='Linha de Tendência',
                line=dict(color='white', dash='dash')
            ))
            fig.update_layout(
                plot_bgcolor='#2f2f2f',
                paper_bgcolor='#2f2f2f',
                font_color='white',
                xaxis=dict(gridcolor='gray', zerolinecolor='gray'),
                yaxis=dict(gridcolor='gray', zerolinecolor='gray')
            )
            st.plotly_chart(fig, use_container_width=True)

    # Exibe gráficos
    regressao_plot('media_dia', 'Qualidade (%)', ['Agente Email', 'Líder'], "Dispersão Média Diária vs Qualidade")
    regressao_plot('ATENDIDOS', 'Qualidade (%)', ['Agente Email', 'Líder'], "Dispersão Atendidos vs Qualidade")
    regressao_plot('ATENDIDOS', 'CSAT (%)', ['Agente Email', 'ATENDIDOS', 'TMA_seg'], "Dispersão Atendidos vs CSAT (%)")
    regressao_plot('TMA_seg', 'Qualidade (%)', ['Agente Email', 'Líder'], "Dispersão TMA (segundos) vs Qualidade")
else:
    st.info("⬆️ Por favor, envie um arquivo Excel para iniciar a análise.")
