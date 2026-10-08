import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import report  # noqa: E402

TRUTH_CFG = {
    'classes': {
        'aluminum_can': {'bin': 'recycle', 'sources': []},
        'aerosol_can': {'bin': 'hhw', 'sources': []},
        'cardboard_box': {'bin': 'recycle', 'sources': []},
        'paper': {'bin': 'recycle', 'sources': []},
        'not_waste': {'bin': 'none', 'sources': []},
    }
}
BINS = report.bins(TRUTH_CFG)


def pred(truth, *tops, file='x.jpg'):
    return {
        'file': file,
        'truth': truth,
        'top': [{'label': l, 'confidence': c} for l, c in tops],
    }


class ScoreTests(unittest.TestCase):
    def test_bins(self):
        self.assertEqual(BINS['aerosol_can'], 'hhw')
        self.assertEqual(BINS['not_waste'], 'none')

    def test_perfect(self):
        preds = [
            pred('aluminum_can', ('aluminum_can', 0.9)),
            pred('paper', ('paper', 0.8)),
        ]
        s = report.score(preds, BINS, BINS)
        self.assertEqual(s['n'], 2)
        self.assertEqual(s['item_top1'], 1.0)
        self.assertEqual(s['bin_acc'], 1.0)

    def test_item_wrong_bin_right(self):
        preds = [pred('cardboard_box', ('paper', 0.9))]
        s = report.score(preds, BINS, BINS)
        self.assertEqual(s['item_top1'], 0.0)
        self.assertEqual(s['bin_acc'], 1.0)
        self.assertEqual(s['confusions'], [['cardboard_box', 'paper', 1]])

    def test_top3(self):
        preds = [pred('paper', ('cardboard_box', 0.5), ('aluminum_can', 0.3), ('paper', 0.1))]
        s = report.score(preds, BINS, BINS)
        self.assertEqual(s['item_top1'], 0.0)
        self.assertEqual(s['item_top3'], 1.0)

    def test_not_waste_bin_counts_correct(self):
        preds = [pred('not_waste', ('not_waste', 0.9))]
        s = report.score(preds, BINS, BINS)
        self.assertEqual(s['bin_acc'], 1.0)

    def test_hazard_misses(self):
        preds = [pred('aerosol_can', ('aluminum_can', 0.9))]
        s = report.score(preds, BINS, BINS)
        self.assertEqual(s['hazard_misses'], 1)

    def test_hazard_not_missed_when_bin_right(self):
        preds = [pred('aerosol_can', ('aerosol_can', 0.9))]
        self.assertEqual(report.score(preds, BINS, BINS)['hazard_misses'], 0)

    def test_per_class_and_worst(self):
        preds = [
            pred('paper', ('paper', 0.9)),
            pred('aluminum_can', ('paper', 0.9)),
        ]
        s = report.score(preds, BINS, BINS)
        self.assertEqual(s['per_class']['paper'], {'n': 1, 'acc': 1.0})
        self.assertEqual(s['per_class']['aluminum_can'], {'n': 1, 'acc': 0.0})
        self.assertEqual(s['worst_classes'][0], 'aluminum_can')

    def test_baseline_material_labels(self):
        pred_bins = {'plastic': 'recycle', 'metal': 'recycle'}
        preds = [pred('aluminum_can', ('metal', 0.9))]
        s = report.score(preds, BINS, pred_bins)
        self.assertIsNone(s['item_top1'])
        self.assertIsNone(s['item_top3'])
        self.assertEqual(s['bin_acc'], 1.0)


class SweepTests(unittest.TestCase):
    def rows(self):
        preds = [
            pred('paper', ('paper', 0.9)),
            pred('aerosol_can', ('aluminum_can', 0.5)),
        ]
        return report.sweep(preds, BINS, BINS)

    def test_thresholds_grid(self):
        self.assertEqual(report.THRESHOLDS[0], 0.3)
        self.assertEqual(report.THRESHOLDS[-1], 0.95)
        self.assertEqual(len(report.THRESHOLDS), 14)

    def test_sweep_and_recommend(self):
        rows = {r['t']: r for r in self.rows()}
        self.assertEqual(rows[0.4]['coverage'], 1.0)
        self.assertEqual(rows[0.4]['wrong_bin_rate'], 0.5)
        self.assertEqual(rows[0.6]['coverage'], 0.5)
        self.assertEqual(rows[0.6]['wrong_bin_rate'], 0.0)
        self.assertEqual(report.recommend_threshold(self.rows(), 0.02), 0.55)

    def test_recommend_none_when_nothing_accepted_safely(self):
        preds = [pred('aerosol_can', ('aluminum_can', 0.99))]
        rows = report.sweep(preds, BINS, BINS)
        self.assertIsNone(report.recommend_threshold(rows, 0.02))

    def test_recommend_requires_an_accepted_photo(self):
        # Nothing is accepted above 0.5, so those rows have 0 wrong but 0 accepted.
        preds = [pred('paper', ('paper', 0.5))]
        rows = report.sweep(preds, BINS, BINS)
        self.assertEqual(report.recommend_threshold(rows, 0.02), 0.3)

    def test_not_waste_never_accepted(self):
        preds = [pred('not_waste', ('not_waste', 0.99))]
        for r in report.sweep(preds, BINS, BINS):
            self.assertEqual(r['coverage'], 0.0)
            self.assertEqual(r['wrong_bin_rate'], 0.0)


class TargetTests(unittest.TestCase):
    def test_check_targets(self):
        self.assertEqual(report.check_targets({'item_top1': 0.5}, {'item': 0.8}), ['item'])
        self.assertEqual(report.check_targets({'item_top1': 0.9}, {'item': 0.8}), [])

    def test_item_none_is_missed(self):
        self.assertEqual(report.check_targets({'item_top1': None}, {'item': 0.8}), ['item'])

    def test_bin_and_coverage(self):
        rep = {'bin_acc': 0.95, 'coverage_at_effective': 0.5}
        self.assertEqual(
            report.check_targets(rep, {'bin': 0.92, 'coverage': 0.65}), ['coverage'])


class CliTests(unittest.TestCase):
    def run_cli(self, preds, extra):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'p.json').write_text(json.dumps(preds))
            (d / 'c.json').write_text(json.dumps(TRUTH_CFG))
            argv = ['--predictions', str(d / 'p.json'), '--truth-config', str(d / 'c.json'),
                    '--out', str(d / 'out.json')] + extra
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = report.main(argv)
            out = json.loads((d / 'out.json').read_text())
            return code, out, buf.getvalue()

    # An unsafe prediction at the top of the grid: no threshold meets 2 % wrong-bin.
    NO_THRESHOLD = [
        pred('aerosol_can', ('aluminum_can', 0.99)),
        pred('paper', ('paper', 0.9)),
    ]

    def test_exit_3_when_coverage_required_and_no_threshold(self):
        code, out, _ = self.run_cli(self.NO_THRESHOLD, ['--require', 'coverage=0.5'])
        self.assertEqual(code, 3)
        self.assertIsNone(out['recommended'])
        self.assertIsNone(out['effective_accept'])

    def test_item_only_required_without_threshold_uses_item(self):
        code, out, _ = self.run_cli(self.NO_THRESHOLD, ['--require', 'item=0.4'])
        self.assertEqual(code, 0)
        self.assertIsNone(out['recommended'])
        self.assertIsNone(out['effective_accept'])
        code, _, _ = self.run_cli(self.NO_THRESHOLD, ['--require', 'item=0.9'])
        self.assertEqual(code, 1)

    def test_effective_accept_and_coverage_gate(self):
        preds = [
            pred('paper', ('paper', 0.9)),
            pred('aerosol_can', ('aluminum_can', 0.5)),
        ]
        code, out, text = self.run_cli(preds, ['--require', 'coverage=0.5'])
        self.assertEqual(out['recommended'], 0.55)
        self.assertEqual(out['effective_accept'], 0.6)
        self.assertEqual(out['coverage_at_effective'], 0.5)
        self.assertEqual(code, 0)
        self.assertIn('0.6', text)
        code, _, _ = self.run_cli(preds, ['--require', 'coverage=0.6'])
        self.assertEqual(code, 1)

    def test_effective_accept_off_grid(self):
        preds = [pred('paper', ('paper', 0.9))]
        _, out, _ = self.run_cli(preds, ['--confirm-below', '0.62'])
        self.assertEqual(out['effective_accept'], 0.62)
        self.assertEqual(out['coverage_at_effective'], 1.0)

    def test_pred_config_defaults_and_all_met(self):
        preds = [pred('paper', ('paper', 0.9))]
        code, out, _ = self.run_cli(preds, ['--require', 'item=0.8,bin=0.9,coverage=0.65'])
        self.assertEqual(code, 0)
        self.assertEqual(out['missed'], [])


if __name__ == '__main__':
    unittest.main()
