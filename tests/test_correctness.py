import unittest
import torch
from compare_outputs import compare_payloads
from models.esm3_adapter import ca_rmsd
from common.reporting import stats
from main import load_samples, REPO


def payload():
    ids = torch.tensor([[0, 4, 5, 2, 1]])
    logits = torch.tensor([[[0.,0.], [3.,1.], [1.,3.], [0.,0.], [0.,0.]]])
    return {"sequence_ids": ids, "valid_mask": torch.tensor([[False, True, True, False, False]]),
            "logits": logits, "embeddings": logits.clone()}


class ComparisonTests(unittest.TestCase):
    def test_equal_strict(self):
        a = payload()
        self.assertEqual(compare_payloads(a, a, atol=0, rtol=0, keys=("logits", "embeddings"))["status"], "pass")

    def test_candidate_argmax_is_used(self):
        a, b = payload(), payload()
        b["logits"][0, 1] = torch.tensor([1., 3.])
        result = compare_payloads(a, b)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["metrics"]["logits"]["top1_agreement"], 0.5)

    def test_padding_is_excluded(self):
        a, b = payload(), payload()
        b["logits"][0, 4] = 999
        self.assertEqual(compare_payloads(a, b, atol=0, rtol=0)["status"], "pass")

    def test_nan_is_rejected(self):
        a, b = payload(), payload()
        b["logits"][0, 1, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN"):
            compare_payloads(a, b)

    def test_shape_is_rejected(self):
        a, b = payload(), payload()
        b["logits"] = b["logits"][:, :-1]
        with self.assertRaisesRegex(ValueError, "Shape"):
            compare_payloads(a, b)

    def test_input_mismatch_is_rejected(self):
        a, b = payload(), payload()
        b["sequence_ids"][0, 1] = 7
        with self.assertRaisesRegex(ValueError, "Token inputs"):
            compare_payloads(a, b)

    def test_missing_and_empty_mask_rejected(self):
        a = payload()
        b = {"logits": a["logits"]}
        with self.assertRaises(ValueError):
            compare_payloads(a, b)
        a["valid_mask"].fill_(False)
        with self.assertRaisesRegex(ValueError, "No valid"):
            compare_payloads(a, a)

    def test_zero_vectors(self):
        a = payload()
        a["logits"].zero_()
        self.assertEqual(compare_payloads(a, a)["metrics"]["logits"]["cosine"], 1)

    def test_legacy_baseline_schema(self):
        a = payload()
        old = {"sequence_ids": a["sequence_ids"][0], "sequence_logits": a["logits"]}
        self.assertEqual(compare_payloads(old, a, atol=0, rtol=0)["status"], "pass")

    def test_embeddings_are_independent(self):
        a, b = payload(), payload()
        b["embeddings"][0, 1] += 2
        self.assertEqual(compare_payloads(a, b, keys=("logits",))["status"], "pass")
        self.assertEqual(compare_payloads(a, b, keys=("logits", "embeddings"))["status"], "fail")


class MeasurementTests(unittest.TestCase):
    def test_kabsch_translation_rotation(self):
        torch.manual_seed(0)
        points = torch.randn(20, 3, dtype=torch.float64)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
        rmsd, count = ca_rmsd(points, points @ rotation + 4)
        self.assertLess(rmsd, 1e-10)
        self.assertEqual(count, 20)

    def test_kabsch_reflection_is_not_rotation(self):
        points = torch.tensor([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
        flipped = points.clone(); flipped[:, 0] *= -1
        self.assertGreater(ca_rmsd(points, flipped)[0], 0.1)

    def test_percentile_and_units(self):
        result = stats([1.,2.,3.,4.,5.])
        self.assertEqual(result["median"], 3)
        self.assertAlmostEqual(result["p90"], 4.6)

    def test_dataset_ids_and_batch(self):
        result = load_samples(REPO / "inputs/test_sequences.jsonl", ["random_L64"], 4)
        self.assertEqual(len(result), 4)
        self.assertTrue(all(len(s["sequence"]) == 64 for s in result))


if __name__ == "__main__":
    unittest.main()
