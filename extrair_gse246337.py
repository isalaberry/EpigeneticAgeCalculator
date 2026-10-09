#!/usr/bin/env python3
"""Extrai 5 participantes do GSE246337 para a calculadora Garma (2024).

Requer apenas a biblioteca padrao do Python. Le .csv.gz diretamente, linha a
linha, sem descompactar o arquivo inteiro ou carrega-lo na memoria.

Uso (na pasta calculadora_garma_original):
    python3 extrair_gse246337.py

Saidas:
    amostras_epicv2_5_pessoas.csv
    idades_cronologicas_5_pessoas.csv

Os IDs das amostras, GSMs e idades abaixo foram conferidos nos registros
publicos individuais do NCBI GEO, serie GSE246337.
"""
from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
import re

AMOSTRAS = [
    # pessoa, GSM, ID tecnico Illumina (GEO Description), idade (anos)
    ("BoA462", "GSM7867425", "207705770050_R07C01", 18.44),
    ("BoA45", "GSM7867008", "207700470024_R03C01", 26.96),
    ("BoA339", "GSM7867302", "207700460084_R03C01", 41.46),
    ("BoA11", "GSM7866974", "207700470041_R08C01", 57.14),
    ("BoA377", "GSM7867340", "207705770089_R08C01", 73.77),
]

RE_CPG = re.compile(r"^cg\d{8}$", re.IGNORECASE)


def ler_cpgs(pesos: Path) -> set[str]:
    if not pesos.is_file():
        raise FileNotFoundError(
            f"Nao encontrei os coeficientes: {pesos}\n"
            "Use o pacote da calculadora com S5-S8 originais."
        )
    with pesos.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"feature", "model", "coefficient", "term"}.issubset(reader.fieldnames or []):
            raise ValueError("O CSV de coeficientes deve conter model,feature,coefficient,term.")
        cpgs = {r["feature"].strip().lower() for r in reader
                if r["term"].strip().lower() != "intercept"
                and RE_CPG.fullmatch(r["feature"].strip())}
    if not cpgs:
        raise ValueError("Nenhum CpG foi encontrado no arquivo de coeficientes.")
    return cpgs


def extrair(entrada: Path, pesos: Path, saida: Path) -> None:
    if not entrada.is_file():
        raise FileNotFoundError(f"Arquivo de metilacao nao encontrado: {entrada}")
    cpgs = ler_cpgs(pesos)
    saida.mkdir(parents=True, exist_ok=True)
    betas_csv = saida / "amostras_epicv2_5_pessoas.csv"
    idades_csv = saida / "idades_cronologicas_5_pessoas.csv"

    with gzip.open(entrada, "rt", encoding="utf-8-sig", newline="") as f_in:
        header = next(csv.reader(f_in), None)
        if not header:
            raise ValueError("O arquivo GSE246337 nao contem cabecalho.")
        if len(header) != len(set(header)):
            raise ValueError("O cabecalho possui nomes de colunas repetidos.")
        indices = []
        for pessoa, gsm, nome_tecnico, idade in AMOSTRAS:
            if nome_tecnico not in header:
                raise ValueError(
                    f"A coluna {nome_tecnico} ({pessoa}, {gsm}) nao aparece "
                    "no arquivo. Confira se e o GSE246337_betas.csv.gz."
                )
            indices.append(header.index(nome_tecnico))
        print(f"Colunas totais no GSE246337: {len(header)-1}")
        print(f"CpGs distintos exigidos pelos quatro modelos: {len(cpgs)}")
        print("Amostras encontradas:")
        for pessoa, gsm, tecnico, idade in AMOSTRAS:
            print(f"  {pessoa} ({gsm}) = {tecnico}; idade {idade:.2f}")

        contagem_linhas = 0
        cpgs_encontrados = set()
        com_valor = {nome: 0 for nome, *_ in AMOSTRAS}
        # A matriz original contem multiplos ensaios EPICv2 por CpG.
        # Preservamos cada linha de ensaio e deixamos a calculadora resolver
        # as replicatas como documentado em seu README.
        with betas_csv.open("w", newline="", encoding="utf-8-sig") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["CpG"] + [nome for nome, *_ in AMOSTRAS])
            for i, linha in enumerate(f_in, start=2):
                # Checagem rapida do CpG antes de interpretar as 500 colunas;
                # reduz tempo/memoria durante a leitura de arquivos gigantes.
                primeiro_campo = linha.partition(",")[0].strip().strip('"')
                cpg_base = primeiro_campo.split("_", 1)[0].lower()
                if cpg_base not in cpgs:
                    continue
                campos = next(csv.reader([linha]))
                if len(campos) != len(header):
                    raise ValueError(
                        f"Linha {i}: numero de colunas inesperado "
                        f"({len(campos)} em vez de {len(header)})."
                    )
                selecionados = [campos[j] for j in indices]
                writer.writerow([campos[0]] + selecionados)
                contagem_linhas += 1
                cpgs_encontrados.add(cpg_base)
                for (nome, *_), beta in zip(AMOSTRAS, selecionados):
                    if beta.strip().lower() not in {"", "na", "nan", "null", "n/a"}:
                        com_valor[nome] += 1
                if contagem_linhas % 2500 == 0:
                    print(f"  {contagem_linhas} linhas de CpGs relevantes extraidas...")

    with idades_csv.open("w", newline="", encoding="utf-8-sig") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["amostra", "gsm", "id_tecnico", "idade_cronologica", "fonte_geo"])
        for nome, gsm, id_tecnico, idade in AMOSTRAS:
            writer.writerow([nome, gsm, id_tecnico, idade,
                             f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm}"])

    print(f"\nExtracao concluida: {contagem_linhas} linhas de probes em {betas_csv.name}.")
    print(f"CpGs distintos detectados: {len(cpgs_encontrados)} de {len(cpgs)}.")
    print(f"CpGs sem nenhuma linha na matriz: {len(cpgs - cpgs_encontrados)}.")
    print("Linhas com beta nao vazio por amostra:")
    for nome, *_ in AMOSTRAS:
        print(f"  {nome}: {com_valor[nome]}")
    print(f"Idades e referencias: {idades_csv}")
    print(f"Betas para a calculadora: {betas_csv}")
    if not contagem_linhas:
        print("AVISO: nenhum CpG foi encontrado; confira os nomes dos probes.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path,
                        default=Path("GSE246337_betas.csv.gz"))
    parser.add_argument("--pesos", type=Path,
                        default=Path("dados_referencia/coeficientes_originais_S5_S8.csv"))
    parser.add_argument("--saida", type=Path, default=Path("."))
    args = parser.parse_args()
    extrair(args.entrada, args.pesos, args.saida)

if __name__ == "__main__":
    main()
