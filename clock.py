"""Cálculo transparente do relógio de Garma & Quintela-Fandiño (2024).

Os coeficientes NÃO são inventados nem treinados aqui: devem ser carregados
integralmente do suplemento S5–S8 ou de um CSV normalizado a partir dele.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re
import math
from typing import BinaryIO

import numpy as np
import pandas as pd

CPG = re.compile(r"^(cg\d{8})(?:_([TB][CO][12]\d+))?$", re.IGNORECASE)
FEATURE = re.compile(r"(cg\d{8})(?:_[TB][CO][12]\d+)?", re.IGNORECASE)
MODEL_NAMES = ("general", "young", "middle", "old")
INTERCEPT = re.compile(r"^(?:\(?intercept\)?|intercepto|constant|const|bias|\(intercept\))$", re.I)
SQUARED = re.compile(r"\^\s*2|\*\*\s*2|²|_sq(?:uared)?\b|\bsq(?:uared)?\b|\bquadratic\b|\bquadrad[oa]\b", re.I)


@dataclass(frozen=True)
class LinearClock:
    name: str
    intercept: float
    terms: tuple[tuple[str, str, float], ...]  # CpG, linear|square, coef

    @property
    def probes(self) -> set[str]:
        return {p for p, _, w in self.terms if w != 0}


@dataclass
class Report:
    sample: str
    age: float | None
    preliminary_age: float | None
    final_model: str
    probes_received: int  # unique CpGs in imported file, including missing beta
    probes_valid_beta: int  # unique CpGs with measured beta in this sample
    raw_probe_rows: int
    required_probes: int
    probes_used: int  # number measured, never including imputation
    probes_missing: list[str]
    probes_imputed: list[str]
    coverage_percent: float
    status: str

    def summary(self):
        return {
            "amostra": self.sample,
            "idade_epigenetica": self.age,
            "idade_preliminar": self.preliminary_age,
            "modelo_aplicado": self.final_model,
            "probes_recebidos": self.probes_received,
            "probes_com_beta_valido": self.probes_valid_beta,
            "linhas_de_probes_recebidas": self.raw_probe_rows,
            "probes_necessarios": self.required_probes,
            "probes_utilizados_medidos": self.probes_used,
            "probes_ausentes": len(self.probes_missing),
            "probes_imputados": len(self.probes_imputed),
            "cobertura_percentual": round(self.coverage_percent, 4),
            "situacao": self.status,
        }


def normalize_probe(value: object) -> str | None:
    """EPIC v2 IlmnID -> cgXXXXXXXX; não modifica sondas inválidas."""
    if value is None or pd.isna(value):
        return None
    m = CPG.fullmatch(str(value).strip())
    return m.group(1).lower() if m else None


def normalize_model(value: object) -> str:
    s = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    if s in {"general", "geral", "s5", "tables5", "modelgeneral", "generalmodel"}:
        return "general"
    if s in {"young", "jovem", "youth", "s6", "tables6", "modelyoung", "youngmodel"}:
        return "young"
    if s in {"middle", "middleaged", "intermediario", "intermediate", "s7", "tables7", "modelmiddle", "middlemodel"}:
        return "middle"
    if s in {"old", "older", "elderly", "idoso", "s8", "tables8", "modelold", "oldmodel"}:
        return "old"
    raise ValueError(f"Modelo desconhecido: {value!r}. Use general, young, middle ou old.")


def _parse_feature(value: object, term_hint: object = None) -> tuple[str | None, str]:
    s = str(value).strip()
    if INTERCEPT.fullmatch(s):
        return None, "intercept"
    m = FEATURE.search(s)
    if m is None:
        raise ValueError(f"Identificador de feature não reconhecido: {value!r}")
    site = m.group(1).lower()
    rest = s[:m.start()] + s[m.end():]
    # Evitar aceitar fragmentos ilegítimos como IDs de outros campos.
    if re.sub(r"[\s\(\)\[\]]", "", rest) and not SQUARED.search(rest):
        raise ValueError(f"Feature ambígua: {value!r}")
    hint = "" if term_hint is None or pd.isna(term_hint) else str(term_hint).strip().lower()
    square = bool(SQUARED.search(rest) or SQUARED.search(hint) or hint in {"square", "squared", "quadratic", "quadratico", "quadrático"})
    if hint in {"intercept", "intercepto"}:
        return None, "intercept"
    if hint and hint not in {"linear", "beta", "1", "quadratic", "square", "squared", "2", "quadratico", "quadrático"} and not SQUARED.search(hint):
        raise ValueError(f"Transformação desconhecida: {term_hint!r}")
    return site, "square" if square or hint == "2" else "linear"


def load_models_from_csv(source) -> dict[str, LinearClock]:
    """CSV padronizado: model, feature, coefficient, [term].

    'feature' aceita cgXXXXXXXX, cgXXXXXXXX^2, (Intercept).
    O campo opcional term=linear|square explicita as potências.
    """
    df = pd.read_csv(source, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"modelo": "model", "probe": "feature", "cpg": "feature",
               "coeficiente": "coefficient", "weight": "coefficient", "peso": "coefficient",
               "transformacao": "term", "transformação": "term"}
    df = df.rename(columns=aliases)
    for c in ("model", "feature", "coefficient"):
        if c not in df:
            raise ValueError(f"CSV de coeficientes necessita coluna '{c}'.")
    records = []
    for _, r in df.iterrows():
        if pd.isna(r["feature"]):
            continue
        model = normalize_model(r["model"])
        feature, term = _parse_feature(r["feature"], r.get("term", ""))
        coef = float(r["coefficient"])
        if not math.isfinite(coef):
            raise ValueError("Coeficiente não finito no modelo.")
        records.append((model, feature, term, coef))
    return _make_models(records)


def _make_models(records: list[tuple]) -> dict[str, LinearClock]:
    models = {}
    for name in MODEL_NAMES:
        subset = [(p, t, w) for n, p, t, w in records if n == name]
        if not subset:
            continue
        ints = [w for p, t, w in subset if t == "intercept"]
        if len(ints) != 1:
            raise ValueError(f"Modelo '{name}' precisa de exatamente um intercepto (encontrados {len(ints)}).")
        seen = set()
        terms = []
        for p, t, w in subset:
            if t == "intercept":
                continue
            if (p, t) in seen:
                raise ValueError(f"Feature repetida em {name}: {p} ({t}). Não somamos pesos duplicados.")
            seen.add((p, t))
            if w != 0:
                terms.append((p, t, w))
        models[name] = LinearClock(name, ints[0], tuple(terms))
    if any(len(m.terms) == 0 for m in models.values()):
        raise ValueError("Modelo sem coeficientes de CpG não é válido.")
    if "general" not in models:
        raise ValueError("Modelo general não encontrado.")
    if any(x in models for x in ("young", "middle", "old")) and not all(x in models for x in MODEL_NAMES):
        raise ValueError("Modelo combinado requer os quatro modelos (general, young, middle, old).")
    return models


def _candidate_tables(df: pd.DataFrame):
    """Encontra pares nome-de-feature / peso sem supor layout fixo do Excel."""
    values = df.values.tolist()
    scores = []
    for col in range(df.shape[1]):
        found = 0
        for row in values[:min(len(values), 200)]:
            s = str(row[col]).strip()
            if INTERCEPT.fullmatch(s) or (FEATURE.search(s) and (s.lower().startswith("cg") or SQUARED.search(s))):
                found += 1
        scores.append(found)
    if not scores or max(scores) < 1:
        raise ValueError("Não foi possível localizar a coluna de CpGs no suplemento. Use CSV padronizado.")
    fcol = int(np.argmax(scores))
    opts = []
    for c in range(df.shape[1]):
        if c == fcol:
            continue
        matches = sum(_is_finite_number(row[c]) for row in values[:min(200, len(values))]
                      if FEATURE.search(str(row[fcol])) or INTERCEPT.fullmatch(str(row[fcol]).strip()))
        if matches:
            opts.append((matches, c))
    if not opts:
        raise ValueError("Não foi possível localizar coeficientes numéricos. Use CSV padronizado.")
    opts.sort(reverse=True)
    if len(opts) > 1 and opts[0][0] == opts[1][0]:
        raise ValueError("Mais de uma coluna de coeficientes possível. Use CSV padronizado.")
    wcol = opts[0][1]
    return fcol, wcol, values


def _is_finite_number(x):
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _read_excel_bytes(source):
    return source.getvalue() if hasattr(source, "getvalue") else source.read() if hasattr(source, "read") else source


def load_models_from_supplement(source) -> dict[str, LinearClock]:
    """Lê as tabelas originais S5--S8, incluindo a coluna `order`.

    Layout REAL de Additional file 5 (2024):
      A: CpG / Intercept; B: Coefficient; C: order (0/1/2); D: Model.
    Coeficiente do intercepto usa vírgula decimal (p.ex. 19,06433).
    O campo order, e NÃO a ocorrência repetida de CpG, define a transformação.
    """
    excel = pd.read_excel(BytesIO(_read_excel_bytes(source)), sheet_name=None, header=None, dtype=object)
    sheets = list(excel.items())
    if len(sheets) != 4:
        raise ValueError(f"Esperadas quatro abas S5--S8; encontradas {len(sheets)}.")
    records = []
    for sheet_idx, (sheet_name, frame) in enumerate(sheets):
        expected = MODEL_NAMES[sheet_idx]
        if re.fullmatch(r"(?i)table\s*s[5-8]",str(sheet_name).strip()):
            tab_id = int(re.search(r"[5-8]$",str(sheet_name)).group())
            if tab_id != sheet_idx + 5:
                raise ValueError("A ordem das abas S5, S6, S7, S8 está incorreta.")
        if (frame.shape[1] >= 4 and
            str(frame.iloc[0, 1]).strip().lower() == "coefficient" and
            str(frame.iloc[0, 2]).strip().lower() == "order"):
            expected_label = {"general": "general", "young": "age-young",
                              "middle": "age-middle", "old": "age-old"}[expected]
            for rowno, row in enumerate(frame.itertuples(index=False, name=None), start=1):
                if rowno == 1:
                    continue
                f, raw_coef, raw_order, raw_label = row[:4]
                if pd.isna(f) and pd.isna(raw_coef):
                    continue
                if pd.isna(f) or pd.isna(raw_coef) or pd.isna(raw_order):
                    raise ValueError(f"Linha incompleta em {sheet_name}, linha {rowno}.")
                label = str(raw_label).strip().lower()
                if label != expected_label:
                    raise ValueError(f"Modelo incoerente em {sheet_name} linha {rowno}: {label!r}.")
                try:
                    order = int(raw_order)
                    if float(raw_order) != order or order not in (0, 1, 2):
                        raise ValueError()
                except (ValueError, TypeError):
                    raise ValueError(f"Termo `order` inválido na linha {rowno}: {raw_order!r}.")
                try:
                    coef = float(str(raw_coef).strip().replace(",", "."))
                except ValueError:
                    raise ValueError(f"Coeficiente inválido em {sheet_name}, linha {rowno}: {raw_coef!r}.")
                if not math.isfinite(coef):
                    raise ValueError(f"Coeficiente não finito em {sheet_name}, linha {rowno}.")
                feature, transform = _parse_feature(f, str(order))
                if (order == 0) != (transform == "intercept"):
                    raise ValueError(f"Inconsistência entre feature e order na linha {rowno}.")
                if order == 1 and transform != "linear" or order == 2 and transform != "square":
                    raise ValueError(f"Transformação inesperada na linha {rowno}.")
                records.append((expected, feature, transform, coef))
        else:
            # Formato alternativo usado nos testes: CpG e peso;
            # termos quadráticos só são aceitos quando explicitados por `^2` etc.
            fcol, wcol, rows = _candidate_tables(frame)
            for row in rows:
                f = row[fcol]
                if not isinstance(f, str):
                    continue
                if not (FEATURE.search(f) or INTERCEPT.fullmatch(f.strip())):
                    continue
                if not _is_finite_number(row[wcol]):
                    continue
                feature, transform = _parse_feature(f)
                records.append((expected, feature, transform, float(row[wcol])))
    return _make_models(records)


def load_reference_means(source) -> dict[str, float]:
    """Importa as médias de S2 em XLSX ou CSV, validando β entre 0 e 1."""
    filename = str(getattr(source, "name", source)).lower()
    if filename.endswith((".csv", ".tsv", ".txt")) or isinstance(source, __import__("io").StringIO):
        frames = {"csv": pd.read_csv(source, sep=None, engine="python", header=None)}
    else:
        if isinstance(source, (str, __import__("pathlib").Path)):
            frames = pd.read_excel(source, sheet_name=None, header=None)
        else:
            frames = pd.read_excel(BytesIO(_read_excel_bytes(source)), sheet_name=None, header=None)
    out = {}
    for frame in frames.values():
        fcol, bcol, rows = _candidate_tables(frame)
        for row in rows:
            site = normalize_probe(row[fcol])
            if site and _is_finite_number(row[bcol]):
                v = float(row[bcol])
                if not 0 <= v <= 1:
                    raise ValueError(f"Média de metilação fora de [0,1]: {site}={v}")
                if site in out and not np.isclose(out[site], v):
                    raise ValueError(f"Média duplicada e divergente para {site}.")
                out[site] = v
    if not out:
        raise ValueError("Nenhum valor médio válido foi encontrado em S2.")
    return out


def load_beta_matrix(source, layout="cpg_rows") -> tuple[pd.DataFrame, int]:
    """Retorna tabela com índice CpG normalizado e colunas amostras."""
    fname = str(getattr(source, "name", source)).lower()
    if fname.endswith((".xls", ".xlsx")):
        df = pd.read_excel(source)
    else:
        df = pd.read_csv(source, sep=None, engine="python", encoding="utf-8-sig")
    if df.empty or df.shape[1] < 2:
        raise ValueError("Arquivo vazio ou sem colunas de amostras.")
    if layout == "long":
        lowered = {str(c).strip().lower(): c for c in df.columns}
        p = next((lowered[x] for x in ("probe", "cpg", "ilmnid", "id") if x in lowered), None)
        s = next((lowered[x] for x in ("sample", "amostra", "sample_id") if x in lowered), None)
        b = next((lowered[x] for x in ("beta", "beta_value", "beta-value", "value") if x in lowered), None)
        if None in (p, s, b):
            raise ValueError("Formato longo requer colunas probe/cpg, sample/amostra e beta.")
        raw = df[[p, s, b]].copy()
        raw.columns = ["cpg", "sample", "beta"]
        raw["cpg"] = raw["cpg"].map(normalize_probe)
        raw = raw.dropna(subset=["cpg"])
        raw["beta"] = _beta_values(raw["beta"])
        n_raw = len(raw)
        matrix = raw.pivot_table(index="cpg", columns="sample", values="beta", aggfunc="mean", dropna=False)
        matrix.columns.name = None
        return matrix, n_raw
    if layout == "samples_rows":
        df = df.set_index(df.columns[0]).T.reset_index()
    names = df.iloc[:, 0].map(normalize_probe)
    usable = names.notna()
    df = df.loc[usable].copy()
    df.iloc[:, 0] = names.loc[usable]
    n_raw = len(df)
    if n_raw == 0:
        raise ValueError("Nenhum ID cgXXXXXXXX (com ou sem sufixo EPICv2) encontrado.")
    if df.columns[1:].duplicated().any():
        raise ValueError("Há identificadores de amostra repetidos nas colunas.")
    for column in df.columns[1:]:
        df[column] = _beta_values(df[column])
    df = df.set_index(df.columns[0])
    # Replicatas físicas de um mesmo CpG são agregadas de modo explícito.
    result = df.groupby(level=0, sort=False).mean()
    return result, n_raw


def _beta_values(values):
    s = values.map(lambda x: x.replace(",", ".") if isinstance(x, str) else x)
    missing = s.isna() | s.astype(str).str.strip().str.lower().isin(["", "na", "nan", "null", "n/a", "-"])
    out = pd.to_numeric(s.where(~missing, other=np.nan), errors="coerce")
    if (out.isna() & ~missing).any():
        raise ValueError(f"β-value não numérico: {s[out.isna() & ~missing].iloc[0]!r}.")
    bad = out.notna() & (~np.isfinite(out) | (out < 0) | (out > 1))
    if bad.any():
        raise ValueError(f"β-value fora do intervalo [0,1]: {out[bad].iloc[0]}.")
    return out.astype(float)


def _score(model: LinearClock, observed: dict[str, float], means: dict[str, float]):
    required = model.probes
    missing = sorted(required - observed.keys())
    unable = [p for p in missing if p not in means]
    if unable:
        return None
    x = {**{p: means[p] for p in missing}, **observed}
    return model.intercept + math.fsum(w * (x[p] if t == "linear" else x[p] ** 2)
                                        for p, t, w in model.terms)


def calculate(matrix: pd.DataFrame, models: dict[str, LinearClock],
              means: dict[str, float] | None = None,
              mode: str = "combined", raw_rows: int | None = None) -> list[Report]:
    """Sem S2, faltas bloqueiam cálculo em vez de serem tratadas como zero."""
    means = means or {}
    if mode not in {"general", "combined"}:
        raise ValueError("Escolha general ou combined.")
    if mode == "combined" and not all(m in models for m in MODEL_NAMES):
        raise ValueError("Modo combinado requer os quatro modelos da publicação.")
    outputs = []
    for sample in matrix.columns:
        col = matrix[sample].dropna()
        observed = {str(p): float(v) for p, v in col.items()}
        initial = _score(models["general"], observed, means)
        if mode == "general":
            branch = "general"
            age = initial
            needed = models["general"].probes
        elif initial is None:
            branch = "indeterminado"
            age = None
            needed = set.union(*(models[m].probes for m in MODEL_NAMES))
        else:
            branch = "young" if initial <= 36 else "middle" if initial < 59 else "old"
            age = _score(models[branch], observed, means)
            needed = models["general"].probes | models[branch].probes
        missing = sorted(needed - observed.keys())
        imputed = sorted(set(missing) & means.keys())
        uncovered = len(set(missing) - means.keys())
        measured = len(needed & observed.keys())
        coverage = (100 * measured / len(needed)) if needed else 100.0
        outputs.append(Report(
            sample=str(sample), age=age, preliminary_age=initial, final_model=branch,
            probes_received=len(matrix), probes_valid_beta=len(observed), raw_probe_rows=raw_rows if raw_rows is not None else len(matrix),
            required_probes=len(needed), probes_used=measured, probes_missing=missing,
            probes_imputed=imputed, coverage_percent=coverage,
            status=("OK" if not missing else "OK (com imputação)" if uncovered == 0 and age is not None
                    else "NÃO CALCULADA: CpGs sem valor de referência"),
        ))
    return outputs
