"""Processamento em lote do relógio original Garma & Quintela-Fandiño (2024)."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from clock import calculate, load_beta_matrix, load_models_from_csv, load_models_from_supplement, load_reference_means

ROOT = Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description='Idade epigenética usando os coeficientes originais de Garma (2024).')
    p.add_argument('--betas', required=True, type=Path, help='Matriz de β-values CSV/TSV/XLSX')
    p.add_argument('--weights', type=Path, default=ROOT/'dados_referencia'/'coeficientes_originais_S5_S8.csv', help='Substitui os pesos originais S5--S8 incluídos')
    p.add_argument('--means', type=Path, default=ROOT/'dados_referencia'/'medias_originais_S2.csv', help='Médias S2 para imputação, incluídas')
    p.add_argument('--sem-imputacao', action='store_true', help='Não substituir CpGs ausentes por médias de S2')
    p.add_argument('--mode', choices=['general','combined'], default='combined')
    p.add_argument('--layout', choices=['cpg_rows','samples_rows','long'], default='cpg_rows')
    p.add_argument('--out', default='relatorio')
    args=p.parse_args()

    if args.weights.suffix.lower()=='.csv':
        models=load_models_from_csv(args.weights)
    else:
        with args.weights.open('rb') as fh:
            models=load_models_from_supplement(fh)
    means={} if args.sem_imputacao else load_reference_means(args.means)
    mat,n=load_beta_matrix(args.betas,args.layout)
    reports=calculate(mat,models,means,args.mode,n)
    pd.DataFrame([r.summary() for r in reports]).to_csv(f'{args.out}_resumo.csv',index=False)
    pd.DataFrame([{'amostra':r.sample,'cpg':x,'imputado':x in r.probes_imputed} for r in reports for x in r.probes_missing],
                 columns=['amostra','cpg','imputado']).to_csv(f'{args.out}_ausentes.csv',index=False)
    for r in reports:
        age=f'{r.age:.2f}' if r.age is not None else 'não calculada'
        warn=' [ATENÇÃO: cobertura <95%, critério interno não publicado]' if r.coverage_percent < 95 else ''
        print(f'{r.sample}: idade={age} anos, cobertura={r.coverage_percent:.2f}%, modelo={r.final_model}, {r.status}{warn}')
    print(f'Arquivos gerados: {args.out}_resumo.csv e {args.out}_ausentes.csv')

if __name__ == '__main__':
    main()
