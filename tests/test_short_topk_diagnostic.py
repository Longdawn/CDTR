import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShortTopkDiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.tool = load_module(ROOT / "tools" / "diagnose_short_topk_candidates.py")

    def test_classify_error_type_len1(self):
        self.assertEqual(
            self.tool.classify_error_type({"gt_len": 1, "pred_len": 0, "cer": 1.0, "ins": 0, "del": 1, "sub": 0}),
            "empty",
        )
        self.assertEqual(
            self.tool.classify_error_type({"gt_len": 1, "pred_len": 1, "cer": 1.0, "ins": 0, "del": 0, "sub": 1}),
            "wrong_char",
        )
        self.assertEqual(
            self.tool.classify_error_type({"gt_len": 1, "pred_len": 2, "cer": 1.0, "ins": 1, "del": 0, "sub": 0}),
            "over_decode",
        )

    def test_classify_error_type_len2(self):
        self.assertEqual(
            self.tool.classify_error_type({"gt_len": 2, "pred_len": 0, "cer": 1.0, "ins": 0, "del": 2, "sub": 0}),
            "empty",
        )
        self.assertEqual(
            self.tool.classify_error_type({"gt_len": 2, "pred_len": 1, "cer": 0.5, "ins": 0, "del": 1, "sub": 0}),
            "short_output",
        )
        self.assertEqual(
            self.tool.classify_error_type({"gt_len": 2, "pred_len": 2, "cer": 0.5, "ins": 0, "del": 0, "sub": 1}),
            "one_correct_one_wrong",
        )
        self.assertEqual(
            self.tool.classify_error_type({"gt_len": 2, "pred_len": 2, "cer": 1.0, "ins": 0, "del": 0, "sub": 2}),
            "both_wrong",
        )
        self.assertEqual(
            self.tool.classify_error_type({"gt_len": 2, "pred_len": 4, "cer": 1.0, "ins": 2, "del": 0, "sub": 0}),
            "over_decode",
        )

    def test_unique_top_labels_removes_duplicates_and_limits_topk(self):
        top_queries = [
            {"best_label": 10},
            {"best_label": 10},
            {"best_label": 20},
            {"best_label": 0},
            {"best_label": 30},
        ]
        self.assertEqual(self.tool.unique_top_labels(top_queries, topk=2), [10, 20])
        self.assertEqual(self.tool.unique_top_labels(top_queries, topk=4), [10, 20, 0, 30])

    def test_compute_topk_flags_for_len2_tracks_any_both_and_missing(self):
        diag_case = {
            "gt_best": [{"label": 100}, {"label": 200}],
            "top_queries": [
                {"best_label": 300},
                {"best_label": 100},
                {"best_label": 200},
                {"best_label": 400},
            ],
        }
        flags = self.tool.compute_topk_flags(diag_case, topks=(1, 2, 4))
        self.assertFalse(flags["top1"]["any_gt_in_topk"])
        self.assertFalse(flags["top1"]["both_gt_in_topk"])
        self.assertEqual(flags["top1"]["missing_gt_count"], 2)
        self.assertTrue(flags["top2"]["any_gt_in_topk"])
        self.assertFalse(flags["top2"]["both_gt_in_topk"])
        self.assertEqual(flags["top2"]["missing_gt_count"], 1)
        self.assertTrue(flags["top4"]["any_gt_in_topk"])
        self.assertTrue(flags["top4"]["both_gt_in_topk"])
        self.assertEqual(flags["top4"]["missing_gt_count"], 0)

    def test_assign_rescue_bucket_marks_len1_empty_as_fill_only(self):
        case = {"gt_len": 1, "pred": "", "gt": "佛"}
        error_type = "empty"
        topk_flags = {
            "top1": {"any_gt_in_topk": True, "both_gt_in_topk": True, "missing_gt_count": 0},
            "top2": {"any_gt_in_topk": True, "both_gt_in_topk": True, "missing_gt_count": 0},
            "top4": {"any_gt_in_topk": True, "both_gt_in_topk": True, "missing_gt_count": 0},
        }
        rescue = self.tool.assign_rescue_bucket(case, error_type, topk_flags)
        self.assertEqual(rescue["bucket"], "fill_only_rescue")
        self.assertTrue(rescue["eligible_empty_top1_rescue"])

    def test_assign_rescue_bucket_marks_len2_short_output_as_append_when_one_gt_already_kept(self):
        case = {"gt_len": 2, "pred": "十", "gt": "十靴"}
        error_type = "short_output"
        topk_flags = {
            "top1": {"any_gt_in_topk": True, "both_gt_in_topk": False, "missing_gt_count": 1},
            "top2": {"any_gt_in_topk": True, "both_gt_in_topk": True, "missing_gt_count": 0},
            "top4": {"any_gt_in_topk": True, "both_gt_in_topk": True, "missing_gt_count": 0},
        }
        rescue = self.tool.assign_rescue_bucket(case, error_type, topk_flags)
        self.assertEqual(rescue["bucket"], "append_rescue")
        self.assertTrue(rescue["pred_char_hits_gt"])
        self.assertTrue(rescue["missing_gt_in_top2"])

    def test_assign_rescue_bucket_marks_len2_substitution_as_reranking(self):
        case = {"gt_len": 2, "pred": "究醯", "gt": "兜醯"}
        error_type = "one_correct_one_wrong"
        topk_flags = {
            "top1": {"any_gt_in_topk": True, "both_gt_in_topk": False, "missing_gt_count": 1},
            "top2": {"any_gt_in_topk": True, "both_gt_in_topk": False, "missing_gt_count": 1},
            "top4": {"any_gt_in_topk": True, "both_gt_in_topk": False, "missing_gt_count": 1},
        }
        rescue = self.tool.assign_rescue_bucket(case, error_type, topk_flags)
        self.assertEqual(rescue["bucket"], "reranking_candidate")

    def test_assign_rescue_bucket_marks_missing_gt_as_not_decode_rescuable(self):
        case = {"gt_len": 2, "pred": "剛杭", "gt": "酬抗"}
        error_type = "both_wrong"
        topk_flags = {
            "top1": {"any_gt_in_topk": False, "both_gt_in_topk": False, "missing_gt_count": 2},
            "top2": {"any_gt_in_topk": False, "both_gt_in_topk": False, "missing_gt_count": 2},
            "top4": {"any_gt_in_topk": False, "both_gt_in_topk": False, "missing_gt_count": 2},
        }
        rescue = self.tool.assign_rescue_bucket(case, error_type, topk_flags)
        self.assertEqual(rescue["bucket"], "not_decode_rescuable")

    def test_align_cases_raises_on_unmatched_idx(self):
        fixed_cases = [{"idx": 1, "gt_len": 1, "pred_len": 0, "cer": 1.0, "ins": 0, "del": 1, "sub": 0, "gt": "佛", "pred": ""}]
        diag_cases = []
        with self.assertRaisesRegex(ValueError, "unmatched|idx"):
            self.tool.align_cases(fixed_cases, diag_cases, strict=True)

    def test_build_report_markdown_mentions_append_limit(self):
        summary = {
            "headline": {
                "len2_short_output_append_rescue_upper_bound_top2": 0.2353,
                "recommendation": "continue_short_rescue",
            },
            "counts": {"len1_errors": 228, "len2_errors": 278},
        }
        report = self.tool.build_markdown_report(summary)
        self.assertIn("append", report)
        self.assertIn("235", report)
        self.assertIn("continue_short_rescue", report)


if __name__ == "__main__":
    unittest.main()
