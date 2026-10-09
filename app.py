"""Calculadora Streamlit do relógio Garma & Quintela-Fandiño (2024)."""
from pathlib import Path

import pandas as pd
import streamlit as st

from clock import calculate, load_beta_matrix, load_models_from_csv, load_models_from_supplement, load_reference_means

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / 'dados_referencia' / 'coeficientes_originais_S5_S8.csv'
DEFAULT_MEANS = ROOT / 'dados_referencia' / 'medias_originais_S2.csv'

st.set_page_config(page_title='Calculadora de Idade Epigenética', page_icon='🧬', layout='wide')
st.title('🧬 Calculadora de Idade Epigenética')
st.caption('Garma & Quintela-Fandiño (2024) · Modelo compatível com Illumina 450K / EPIC v1 / EPIC v2')
st.markdown('**Pesos reais do estudo incluídos:** Additional file 5, Tabelas S5–S8. **Médias reais incluídas:** Additional file 2, Tabela S2. [Artigo publicado](https://doi.org/10.1186/s13073-024-01387-4).')

with st.sidebar:
    st.header('1 · Coeficientes publicados')
    st.success('S5–S8 incluídos no pacote (4 modelos).')
    weights_upload = st.file_uploader('Substituir por outra planilha S5–S8 ou CSV (opcional)', type=['xlsx', 'csv'])
    mode = st.radio('Modelo de previsão', ['combined', 'general'],
                    format_func=lambda x: 'Combinado (geral + faixa etária)' if x == 'combined' else 'Somente geral')
    st.header('2 · Valores faltantes')
    impute = st.checkbox('Imputar CpGs ausentes com médias publicadas (S2)', value=True)
    means_upload = st.file_uploader('Substituir médias S2 por outro arquivo (opcional)', type=['xlsx', 'csv'], disabled=not impute)
    st.caption('A cobertura sempre informa a proporção de CpGs realmente medidos, não imputados.')

st.subheader('3 · Importar β-values')
beta_file = st.file_uploader('Arquivo com uma ou várias amostras', type=['csv', 'tsv', 'txt', 'xlsx'])
layout = st.selectbox('Formato da matriz', ['cpg_rows', 'samples_rows', 'long'],
    format_func=lambda x: {'cpg_rows':'CpGs nas linhas; amostras nas colunas',
                           'samples_rows':'Amostras nas linhas; CpGs nas colunas',
                           'long':'Uma linha por CpG e amostra (probe,sample,beta)'}[x])
st.code('CpG,amostra_A,amostra_B\ncg16867657_TC11,0.32,0.35\ncg11071401_TC11,0.80,0.75', language='csv')
st.caption('Entrada: β-values previamente normalizados e submetidos a controle de qualidade; valores entre 0 e 1. IDs EPIC v2 com sufixos são reconhecidos. Esta ferramenta não lê IDAT.')

if st.button('Calcular idade epigenética', type='primary', disabled=beta_file is None):
    try:
        if weights_upload is None:
            models = load_models_from_csv(DEFAULT_WEIGHTS)
        elif weights_upload.name.lower().endswith('.csv'):
            models = load_models_from_csv(weights_upload)
        else:
            models = load_models_from_supplement(weights_upload)
        means = (load_reference_means(means_upload or DEFAULT_MEANS) if impute else {})
        matrix, raw_rows = load_beta_matrix(beta_file, layout=layout)
        reports = calculate(matrix, models, means=means, mode=mode, raw_rows=raw_rows)
        summary = pd.DataFrame([r.summary() for r in reports])
    except Exception as exc:
        st.error(f'Não foi possível realizar o cálculo: {exc}')
        st.info('Verifique o formato dos dados de entrada e o README. O projeto já contém os coeficientes originais.')
    else:
        st.session_state['reports'] = reports
        st.session_state['summary'] = summary
        st.success(f'Processadas {len(reports)} amostra(s).')

if 'summary' in st.session_state:
    summary = st.session_state['summary']
    reports = st.session_state['reports']
    st.subheader('4 · Relatório por amostra')
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.download_button('Baixar resumo CSV', summary.to_csv(index=False).encode('utf-8-sig'),
                       file_name='relatorio_idade_epigenetica.csv', mime='text/csv')
    absent = pd.DataFrame(
        [{'amostra':r.sample, 'cpg':c, 'imputado':c in r.probes_imputed} for r in reports for c in r.probes_missing],
        columns=['amostra', 'cpg', 'imputado'])
    st.download_button('Baixar lista de CpGs ausentes CSV', absent.to_csv(index=False).encode('utf-8-sig'),
                       file_name='cpgs_ausentes.csv', mime='text/csv')
    chosen = st.selectbox('Detalhamento da amostra', [r.sample for r in reports])
    result = next(r for r in reports if r.sample == chosen)
    a, b, c, d = st.columns(4)
    a.metric('Idade estimada', f'{result.age:.2f} anos' if result.age is not None else 'Não calculada')
    b.metric('CpGs recebidos', f'{result.probes_received:,}')
    c.metric('CpGs utilizados', f'{result.probes_used:,} / {result.required_probes:,}')
    d.metric('Cobertura medida', f'{result.coverage_percent:.2f}%')
    st.write(f'**Modelo aplicado:** {result.final_model} · **Condição:** {result.status} · **CpGs imputados:** {len(result.probes_imputed)}')
    if result.coverage_percent < 95:
        st.warning('Cobertura medida abaixo de 95%. Este é um alerta conservador de qualidade desta ferramenta, não um ponto de corte validado no artigo. A previsão pode depender fortemente de imputação e não deve ser interpretada sem avaliação cuidadosa.')
    if result.probes_missing:
        st.dataframe(pd.DataFrame({'CpG': result.probes_missing,
                        'Imputado pela média S2':[c in result.probes_imputed for c in result.probes_missing]}),
                     hide_index=True, use_container_width=True)
    else:
        st.success('Todos os CpGs necessários estavam presentes nesta amostra.')

st.divider()
st.caption('Uso em pesquisa, não diagnóstico clínico. Ainda é necessário validar concordância com previsões publicadas em amostras de referência e garantir pré-processamento adequado de metilação.')
