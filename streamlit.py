import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import time
import numpy as np

#formulas

#preco justo acao 6% dividendo
def preco_acao(preco,dy, p = 0.06):
    dy_t = float(str(dy).replace(',','.')) / 100
    preco = float(str(preco).replace(',','.'))

    ultimo_dividendo = dy_t * preco

    preco_justo = ultimo_dividendo / p

    return preco_justo


#tratar erro de texto 
def numInc(num):
    try:
        return float(num.replace(',','.'))
    except:
        return float(num.replace(',',''))


#custo graham
def calculo_graham(d_ft):
    ls_rent = []
    ls_custo = []

    for index , i in d_ft.iterrows():
        lpa = numInc(i.LPA)
        vpa = numInc(i.VPA)
        preco_m = numInc(i.PRECO)
        preco_r = np.sqrt(22.5 * lpa * vpa)
        custo_acao = preco_m/preco_r

        ls_custo.append(custo_acao)
        ls_rent.append(preco_r)
        
    d_ft['Previsao Graham'] = ls_rent
    d_ft['Custo_Graham'] = ls_custo        
    
    return d_ft


df = pd.read_excel('preco_investido.xlsx')

# Título do app
st.title("Gráfico de Barras - Preço Investido por Título")

# Filtro de seleção dos títulos
titulos_selecionados = st.multiselect(
    "Selecione os títulos para exibir no gráfico:",
    options=df["Titulo"].tolist(),
    default=df["Titulo"].tolist()  # por padrão mostra todos
)

# Filtrar o DataFrame
df_filtrado = df[df["Titulo"].isin(titulos_selecionados)]

# Ordenar por valor investido (opcional)
df_sorted = df_filtrado.sort_values(by="Preço Investido", ascending=False)

# Criar gráfico de barras
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(df_sorted["Titulo"], df_sorted["Preço Investido"], color='skyblue')
ax.set_xlabel("Preço Investido")
ax.set_ylabel("Título")
ax.set_title("Preço Investido por Título")

# Adiciona os valores nas barras
for bar in bars:
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
            f'{width:.2f}', va='center', fontsize=9)

plt.tight_layout()

# Mostrar no Streamlit
st.pyplot(fig)


#acao
# Carregar os dados do CSV de múltiplos
multiplos_df = pd.read_csv('statusinvest-busca-avancada (4).csv', sep=';')
multiplos_df.columns = multiplos_df.columns.str.strip()  # Remover espaços extras nos nomes das colunas

tickers_ls = ['ABEV3', 'BBDC3', 'EGIE3',  'ITSA4', 'KLBN11',  'TRPL4', 'USIM3',  'WEGE3', 'SAPR11', 'VALE3']

# Filtrar pelo ticker selecionado
multiplos_df_filtrado = multiplos_df[multiplos_df['TICKER'].isin(tickers_ls)]

stocks_multiplos = multiplos_df_filtrado

stocks_multiplos = stocks_multiplos.reset_index(drop=True) 
stocks_multiplos = calculo_graham(stocks_multiplos)
stocks_multiplos['PRECO_JUSTO_6%'] = stocks_multiplos.apply(lambda row: preco_acao(row['PRECO'],row['DY']),axis=1)

# Mostrar os dados no Streamlit
if not stocks_multiplos.empty:
    st.subheader("Múltiplos dos Títulos Selecionados")
    st.dataframe(stocks_multiplos)
else:
    st.warning("Nenhum dado encontrado para os títulos selecionados.")

cols_para_corrigir = ['PRECO', 'Previsao Graham', 'PRECO_JUSTO_6%']

for col in cols_para_corrigir:
    stocks_multiplos[col] = (
        stocks_multiplos[col]
        .astype(str)
        .str.replace(',', '.', regex=False)
        .astype(float)
        .fillna(0)      # Preenche NaN com 0
        .replace([float('inf'), -float('inf')], 0)  # Substitui infinitos por 0
        .round()
        .astype(int)
    )



# Verifica se as colunas necessárias existem
if not stocks_multiplos.empty and all(col in stocks_multiplos.columns for col in ['TICKER', 'PRECO', 'Previsao Graham', 'PRECO_JUSTO_6%']):
    st.subheader("Comparativo: Preço Atual x Preço Justo (Graham) x Preço Justo com DY 6%")

    # Configurar dados
    tickers = stocks_multiplos['TICKER']
    preco_atual = stocks_multiplos['PRECO']
    preco_graham = stocks_multiplos['Previsao Graham']
    preco_justo_6 = stocks_multiplos['PRECO_JUSTO_6%']

    x = np.arange(len(tickers))  # localização das barras
    largura = 0.25  # largura de cada barra

    # Criar gráfico de barras
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - largura, preco_atual, width=largura, label='Preço Atual', color='skyblue')
    bars2 = ax.bar(x, preco_graham, width=largura, label='Graham', color='lightgreen')
    bars3 = ax.bar(x + largura, preco_justo_6, width=largura, label='Preço Justo 6%', color='salmon')

    ax.set_xlabel('Ticker')
    ax.set_ylabel('Valor (R$)')
    ax.set_title('Comparação de Preços por Ticker')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adicionar valores nas barras
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # deslocamento vertical
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()

    # Mostrar no Streamlit
    st.pyplot(fig)
else:
    st.warning("Não foi possível criar o gráfico. Verifique se as colunas 'PRECO', 'Previsao Graham' e 'PRECO_JUSTO_6%' estão presentes.")


#acao
# Carregar os dados do CSV de múltiplos
multiplos_fundo = pd.read_csv('statusinvest-busca-avancada (5).csv', sep=';')
multiplos_fundo.columns = multiplos_fundo.columns.str.strip()  # Remover espaços extras nos nomes das colunas

tickers_ls = ['XPLG11','KNRI11','JSRE11','HGLG11',]

# Filtrar pelo ticker selecionado
multiplos_fundo = multiplos_fundo[multiplos_fundo['TICKER'].isin(tickers_ls)]

stocks_multiplos = multiplos_fundo

stocks_multiplos = stocks_multiplos.reset_index(drop=True) 
#stocks_multiplos = calculo_graham(stocks_multiplos)
#stocks_multiplos['PRECO_JUSTO_6%'] = stocks_multiplos.apply(lambda row: preco_acao(row['PRECO'],row['DY']),axis=1)

# Mostrar os dados no Streamlit
if not stocks_multiplos.empty:
    st.subheader("Múltiplos dos FUNDOS Selecionados")
    st.dataframe(stocks_multiplos)
else:
    st.warning("Nenhum dado encontrado para os títulos selecionados.")