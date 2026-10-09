"""Testes com coeficientes ARTIFICIAIS, não representam o artigo."""
import io
import unittest
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clock import (calculate, load_beta_matrix, load_models_from_csv,
                   load_reference_means, load_models_from_supplement, normalize_probe)


def fake_models():
    return load_models_from_csv(io.StringIO("""model,feature,coefficient,term
    general,(Intercept),10,intercept
    general,cg00000029,40,linear
    general,cg00000109,10,square
    young,(Intercept),8,intercept
    young,cg00000029,35,linear
    middle,(Intercept),28,intercept
    middle,cg00000029,35,linear
    old,(Intercept),48,intercept
    old,cg00000029,35,linear
    """))


class ClockTests(unittest.TestCase):
    def test_epicv2_name(self):
        self.assertEqual(normalize_probe("cg00000029_TC11"), "cg00000029")
        self.assertIsNone(normalize_probe("random_probe"))

    def test_general_quadratic_multiple_samples(self):
        data = io.StringIO("CpG,A,B\ncg00000029_TC11,0.5,0.8\ncg00000109_TC21,0.2,0.4\n")
        mat, n = load_beta_matrix(data)
        r = calculate(mat, fake_models(), mode="general", raw_rows=n)
        self.assertAlmostEqual(r[0].age, 30.4)
        self.assertAlmostEqual(r[1].age, 43.6)
        self.assertEqual(r[0].coverage_percent, 100)
        self.assertEqual(r[0].probes_used, 2)

    def test_combined_branches(self):
        raw = io.StringIO("CpG,young,middle,old\ncg00000029_TC11,0.3,0.8,1\ncg00000109_TC11,0,0,1\n")
        mat, _ = load_beta_matrix(raw)
        r = calculate(mat, fake_models())
        self.assertEqual([x.final_model for x in r], ["young", "middle", "old"])
        self.assertAlmostEqual(r[0].age, 18.5)
        self.assertAlmostEqual(r[1].age, 56)
        self.assertAlmostEqual(r[2].age, 83)

    def test_missing_blocks_and_reports(self):
        mat, n = load_beta_matrix(io.StringIO("CpG,A\ncg00000029,0.5\n"))
        r = calculate(mat, fake_models(), mode="general", raw_rows=n)[0]
        self.assertIsNone(r.age)
        self.assertEqual(r.probes_missing, ["cg00000109"])
        self.assertEqual(r.probes_used, 1)
        self.assertEqual(r.coverage_percent, 50)

    def test_imputation_s2_preserves_coverage(self):
        mat, _ = load_beta_matrix(io.StringIO("CpG,A\ncg00000029,0.5\n"))
        means = load_reference_means(io.StringIO("CpG,Mean\ncg00000109,0.2\n"))
        r = calculate(mat, fake_models(), means, "general")[0]
        self.assertAlmostEqual(r.age, 30.4)
        self.assertEqual(r.coverage_percent, 50)
        self.assertEqual(r.probes_imputed, ["cg00000109"])

    def test_aggregates_replicates(self):
        mat, n = load_beta_matrix(io.StringIO("CpG,A\ncg00000029_TC11,0.2\ncg00000029_TC12,0.6\n"))
        self.assertEqual(n, 2)
        self.assertEqual(len(mat), 1)
        self.assertAlmostEqual(mat.loc["cg00000029", "A"], 0.4)

    def test_invalid_beta_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            load_beta_matrix(io.StringIO("CpG,A\ncg00000029,1.2\n"))

    def test_csv_double_cpg_square_terms(self):
        m = fake_models()["general"]
        self.assertEqual({(p, t) for p, t, _ in m.terms},
                         {("cg00000029", "linear"), ("cg00000109", "square")})

    def test_long_format(self):
        mat, n = load_beta_matrix(io.StringIO("probe,sample,beta\ncg00000029_TC11,A,0.3\ncg00000029_TC12,A,0.5\ncg00000029_TC11,B,0.6\n"), "long")
        self.assertEqual(n, 3)
        self.assertAlmostEqual(mat.loc["cg00000029", "A"], 0.4)
        self.assertAlmostEqual(mat.loc["cg00000029", "B"], 0.6)

    def test_without_general_stage_combined_no_prediction(self):
        mat, _ = load_beta_matrix(io.StringIO("CpG,A\ncg00000029,0.8\n"))
        r = calculate(mat, fake_models())[0]
        self.assertIsNone(r.age)
        self.assertEqual(r.final_model, "indeterminado")

    def test_boundary_36_59(self):
        data = io.StringIO("CpG,A,B\ncg00000029,0.65,0.975\ncg00000109,0,1\n")
        mat, _ = load_beta_matrix(data)
        result = calculate(mat, fake_models())
        self.assertAlmostEqual(result[0].preliminary_age, 36)
        self.assertEqual(result[0].final_model, "young")
        self.assertAlmostEqual(result[1].preliminary_age, 59)
        self.assertEqual(result[1].final_model, "old")

    def test_four_sheet_xlsx_supplement_import(self):
        x = io.BytesIO()
        rows = {
            "S5": [("(Intercept)", 10), ("cg00000029", 40), ("cg00000109^2", 10)],
            "S6": [("(Intercept)", 8), ("cg00000029", 35)],
            "S7": [("(Intercept)", 28), ("cg00000029", 35)],
            "S8": [("(Intercept)", 48), ("cg00000029", 35)],
        }
        with pd.ExcelWriter(x, engine="openpyxl") as writer:
            for name, data in rows.items():
                pd.DataFrame(data, columns=["CpG", "Weight"]).to_excel(writer, sheet_name=name, index=False)
        models = load_models_from_supplement(x)
        self.assertEqual(set(models), {"general", "young", "middle", "old"})
        self.assertIn(("cg00000109", "square", 10.0), models["general"].terms)

    def test_xlsx_s2(self):
        x = io.BytesIO()
        with pd.ExcelWriter(x, engine="openpyxl") as writer:
            pd.DataFrame([("cg00000109", 0.42)], columns=["CpG", "Mean beta"]).to_excel(writer, index=False)
        out = load_reference_means(x)
        self.assertAlmostEqual(out["cg00000109"], 0.42)

    def test_samples_rows(self):
        raw = io.StringIO("Sample,cg00000029_TC11,cg00000109_TC21\nA,0.5,0.2\nB,0.8,0.4\n")
        mat, n = load_beta_matrix(raw, "samples_rows")
        self.assertEqual(list(mat.columns), ["A", "B"])
        self.assertEqual(n, 2)


if __name__ == "__main__":
    unittest.main()
