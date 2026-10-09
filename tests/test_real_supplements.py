"""Verificação de importação do modelo REAL fornecido nos suplementos.

Não é validação externa de coorte (faltam betas individuais de referência).
"""
from pathlib import Path
import sys
import unittest
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clock import load_models_from_csv, load_models_from_supplement, load_reference_means, calculate

ROOT=Path(__file__).resolve().parents[1]

class RealSupplementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_csv=load_models_from_csv(ROOT/'dados_referencia'/'coeficientes_originais_S5_S8.csv')
        cls.means_csv=load_reference_means(ROOT/'dados_referencia'/'medias_originais_S2.csv')

    def test_real_xlsx_matches_converted_csv_exactly(self):
        with (ROOT/'suplementos_originais'/'S5_a_S8_coeficientes.xlsx').open('rb') as f:
            models_xlsx=load_models_from_supplement(f)
        self.assertEqual(models_xlsx,self.models_csv)

    def test_number_of_real_terms(self):
        expected={'general':(4962,4885),'young':(3906,3838),'middle':(4619,4539),'old':(4735,4645)}
        self.assertEqual({key:(len(m.terms),len(m.probes)) for key,m in self.models_csv.items()},expected)
        self.assertAlmostEqual(self.models_csv['general'].intercept,19.064334444910056)

    def test_reference_means_cover_required_cpgs(self):
        needed=set.union(*(m.probes for m in self.models_csv.values()))
        self.assertEqual(len(needed),7869)
        self.assertEqual(needed - self.means_csv.keys(),set())
        with (ROOT/'suplementos_originais'/'S2_medias_beta.xlsx').open('rb') as f:
            self.assertEqual(load_reference_means(f),self.means_csv)

    def test_s2_mean_vector_numerical_consistency_not_biological_sample(self):
        # All 7869 CpGs use the published mean values; synthetic vector,
        # not a biologically observed subject and not external validation.
        needed=set.union(*(m.probes for m in self.models_csv.values()))
        mat=pd.DataFrame({'vetor_medias_S2':pd.Series({p:self.means_csv[p] for p in needed})})
        result=calculate(mat,self.models_csv,self.means_csv,'combined')[0]
        self.assertEqual(result.final_model,'middle')
        self.assertAlmostEqual(result.preliminary_age,46.139779817341875,places=8)
        self.assertAlmostEqual(result.age,45.67790610138286,places=8)
        self.assertEqual(result.coverage_percent,100.0)
        self.assertEqual(result.probes_missing,[])
