import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import report  # noqa: E402

TRUTH_CFG = {
    'classes': {
        'aluminum_can': {'bin': 'recycle', 'sources': []},
        'food_can': {'bin': 'recycle', 'sources': []},
        'aerosol_can': {'bin': 'hhw', 'sources': []},
        'styrofoam_food': {'bin': 'center', 'sources': []},
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


class OpenSetScoreTests(unittest.TestCase):
    def test_unlisted_truth_bin_needs_no_config(self):
        self.assertEqual(report.truth_bins_for(['unlisted__hhw', 'paper'], BINS)['unlisted__hhw'],
                         'hhw')

    def test_unknown_truth_label_is_config_error(self):
        with self.assertRaises(ValueError) as cm:
            report.truth_bins_for(['paper', 'pint_can'], BINS)
        self.assertIn('pint_can', str(cm.exception))

    def test_unlisted_with_unknown_bin_is_config_error(self):
        with self.assertRaises(ValueError):
            report.truth_bins_for(['unlisted__hazard'], BINS)

    def test_unlisted_excluded_from_item_metrics_but_in_bin_and_hazard(self):
        tb = report.truth_bins_for(['paper', 'unlisted__hhw', 'unlisted__recycle'], BINS)
        preds = [
            pred('paper', ('paper', 0.9)),
            pred('unlisted__hhw', ('food_can', 0.9)),
            pred('unlisted__recycle', ('aluminum_can', 0.9)),
        ]
        s = report.score(preds, tb, BINS)
        self.assertEqual(s['n'], 3)
        self.assertEqual(s['item_top1'], 1.0)
        self.assertEqual(s['item_top3'], 1.0)
        self.assertAlmostEqual(s['bin_acc'], 2 / 3)
        self.assertEqual(s['hazard_misses'], 1)
        self.assertNotIn('unlisted__hhw', s['worst_classes'])

    def test_unlisted_out_of_coverage_but_in_wrong_bin(self):
        tb = report.truth_bins_for(['paper', 'unlisted__hhw'], BINS)
        preds = [pred('paper', ('paper', 0.9)), pred('unlisted__hhw', ('food_can', 0.9))]
        row = report.sweep(preds, tb, BINS)[0]
        self.assertEqual(row['coverage'], 1.0)
        self.assertEqual(row['accepted'], 2)
        self.assertEqual(row['wrong_bin_rate'], 0.5)
        self.assertEqual(row['item_acc'], 1.0)

    def test_only_unlisted_item_top1_is_none(self):
        tb = report.truth_bins_for(['unlisted__hhw'], BINS)
        s = report.score([pred('unlisted__hhw', ('food_can', 0.9))], tb, BINS)
        self.assertIsNone(s['item_top1'])
        self.assertEqual(s['bin_acc'], 0.0)


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

    def test_recommend_non_monotone_wrong_bin_rate(self):
        # Wrong-bin photo at 0.35, correct photos over 0.40-0.59, wrong-bin at 0.9.
        # Rate at t=0.30 is 2/102 <= 2 %, so the old first-fit rule said 0.30, but
        # at t=0.60 and above the only accepted photo is the 0.9 miss (rate 1.0).
        preds = [pred('aerosol_can', ('aluminum_can', 0.35))]
        preds += [pred('paper', ('paper', 0.40 + 0.0019 * i)) for i in range(100)]
        preds += [pred('aerosol_can', ('aluminum_can', 0.9))]
        rows = report.sweep(preds, BINS, BINS)
        by_t = {r['t']: r for r in rows}
        self.assertLessEqual(by_t[0.3]['wrong_bin_rate'], 0.02)
        self.assertEqual(by_t[0.9]['wrong_bin_rate'], 1.0)
        self.assertIsNone(report.recommend_threshold(rows, 0.02))

    def test_recommend_skips_unsafe_middle_threshold(self):
        # 100 correct at 0.30-0.49, one miss at 0.50, one correct at 0.9.
        # t=0.30 looks fine (1/102) but t=0.50 is 1/2; first t safe from there up is 0.55.
        preds = [pred('paper', ('paper', 0.30 + 0.0019 * i)) for i in range(100)]
        preds += [pred('aerosol_can', ('aluminum_can', 0.50))]
        preds += [pred('paper', ('paper', 0.9))]
        rows = report.sweep(preds, BINS, BINS)
        by_t = {r['t']: r for r in rows}
        self.assertLessEqual(by_t[0.3]['wrong_bin_rate'], 0.02)
        self.assertEqual(by_t[0.5]['wrong_bin_rate'], 0.5)
        self.assertEqual(report.recommend_threshold(rows, 0.02), 0.55)

    def test_not_waste_truth_not_in_coverage_numerator(self):
        # not_waste photo predicted paper at 0.9 is accepted and wrong, but is not
        # real waste, so it adds nothing to coverage.
        preds = [
            pred('not_waste', ('paper', 0.9)),
            pred('paper', ('paper', 0.9)),
        ]
        row = report.sweep(preds, BINS, BINS)[0]
        self.assertEqual(row['t'], 0.3)
        self.assertEqual(row['coverage'], 1.0)
        self.assertEqual(row['wrong_bin_rate'], 0.5)
        self.assertEqual(row['accepted'], 2)

    def test_all_not_waste_truth_no_division_by_zero(self):
        preds = [
            pred('not_waste', ('paper', 0.9)),
            pred('not_waste', ('not_waste', 0.9)),
        ]
        rows = report.sweep(preds, BINS, BINS)
        for r in rows:
            self.assertEqual(r['coverage'], 0.0)
        self.assertEqual(rows[0]['wrong_bin_rate'], 1.0)
        self.assertEqual(rows[0]['accepted'], 1)

    def test_empty_predictions(self):
        s = report.score([], BINS, BINS)
        self.assertEqual(s['n'], 0)
        self.assertIsNone(s['item_top1'])
        self.assertIsNone(s['item_top3'])
        self.assertEqual(s['bin_acc'], 0.0)
        self.assertEqual(s['per_class'], {})
        self.assertEqual(s['worst_classes'], [])
        self.assertEqual(s['confusions'], [])
        self.assertEqual(s['hazard_misses'], 0)
        rows = report.sweep([], BINS, BINS)
        self.assertEqual(len(rows), 14)
        for r in rows:
            self.assertEqual(r['coverage'], 0.0)
            self.assertEqual(r['wrong_bin_rate'], 0.0)
            self.assertEqual(r['item_acc'], 0.0)
            self.assertEqual(r['accepted'], 0)
        self.assertIsNone(report.recommend_threshold(rows, 0.02))

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
        self.assertEqual(out['wrong_bin_at_effective'], 0.0)
        self.assertIn('wrong-bin at effective accept', text)
        self.assertEqual(code, 0)
        self.assertIn('0.6', text)
        code, _, _ = self.run_cli(preds, ['--require', 'coverage=0.6'])
        self.assertEqual(code, 1)

    def test_exit_3_when_wrong_bin_exceeds_cap_at_effective_accept(self):
        # Safe on the grid (t=0.60 has 1/102, t=0.65 has 0/1) but --confirm-below 0.62
        # lands off-grid where the 0.63 miss is 1 of 2 accepted photos.
        preds = [pred('paper', ('paper', 0.605)) for _ in range(100)]
        preds += [pred('aerosol_can', ('aluminum_can', 0.63)), pred('paper', ('paper', 0.9))]
        code, out, _ = self.run_cli(preds, ['--confirm-below', '0.62', '--require', 'coverage=0.1'])
        self.assertEqual(out['recommended'], 0.3)
        self.assertEqual(out['effective_accept'], 0.62)
        self.assertEqual(out['wrong_bin_at_effective'], 0.5)
        self.assertEqual(code, 3)
        code, _, _ = self.run_cli(preds, ['--confirm-below', '0.62'])
        self.assertEqual(code, 0)

    def test_effective_accept_off_grid(self):
        preds = [pred('paper', ('paper', 0.9))]
        _, out, _ = self.run_cli(preds, ['--confirm-below', '0.62'])
        self.assertEqual(out['effective_accept'], 0.62)
        self.assertEqual(out['coverage_at_effective'], 1.0)

    def test_unknown_truth_label_exits_1_with_message(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'p.json').write_text(json.dumps([pred('pint_can', ('paper', 0.9))]))
            (d / 'c.json').write_text(json.dumps(TRUTH_CFG))
            err = io.StringIO()
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = report.main(['--predictions', str(d / 'p.json'),
                                    '--truth-config', str(d / 'c.json'),
                                    '--out', str(d / 'out.json')])
            self.assertEqual(code, 1)
            self.assertIn('pint_can', err.getvalue())
            self.assertFalse((d / 'out.json').exists())

    def test_reviewer_repro_hazard_miss_at_effective_exits_3(self):
        # 2/110 wrong-bin is under the 2 % cap, so the old gate passed (exit 0),
        # but both misses send a hazardous aerosol can to the recycling cart.
        preds = [pred('paper', ('paper', 0.95)) for _ in range(100)]
        preds += [pred('aerosol_can', ('aerosol_can', 0.95)) for _ in range(8)]
        preds += [pred('aerosol_can', ('aluminum_can', 0.95)) for _ in range(2)]
        code, out, text = self.run_cli(preds, ['--require', 'item=0.80,bin=0.92,coverage=0.65'])
        self.assertLessEqual(out['wrong_bin_at_effective'], 0.02)
        self.assertEqual(out['missed'], [])
        self.assertEqual(out['hazard_misses_at_effective'], 2)
        self.assertEqual(code, 3)
        self.assertIn('hazardous misses at effective accept', text)
        code, _, _ = self.run_cli(preds, ['--require', 'item=0.80,bin=0.92'])
        self.assertEqual(code, 0)

    def test_unlisted_hhw_counts_as_hazard_miss_at_effective(self):
        preds = [pred('paper', ('paper', 0.95)) for _ in range(100)]
        preds += [pred('unlisted__hhw', ('food_can', 0.95))]
        code, out, _ = self.run_cli(preds, ['--require', 'coverage=0.65'])
        self.assertEqual(out['hazard_misses_at_effective'], 1)
        self.assertEqual(out['item_top1'], 1.0)
        self.assertEqual(out['open_set'],
                         {'n': 1, 'accepted_at_effective': 1, 'wrong_bin_at_effective': 1})
        self.assertEqual(code, 3)

    def test_hazard_miss_below_effective_not_counted(self):
        preds = [pred('paper', ('paper', 0.95)) for _ in range(100)]
        preds += [pred('aerosol_can', ('aluminum_can', 0.5))]
        code, out, _ = self.run_cli(preds, ['--require', 'coverage=0.65'])
        self.assertEqual(out['effective_accept'], 0.6)
        self.assertEqual(out['hazard_misses'], 1)
        self.assertEqual(out['hazard_misses_at_effective'], 0)
        self.assertEqual(code, 0)

    def test_center_misses_at_effective_is_informational(self):
        preds = [pred('paper', ('paper', 0.95)) for _ in range(100)]
        preds += [pred('styrofoam_food', ('paper', 0.95))]
        code, out, text = self.run_cli(preds, ['--require', 'coverage=0.65'])
        self.assertEqual(out['center_misses_at_effective'], 1)
        self.assertEqual(out['hazard_misses_at_effective'], 0)
        self.assertIn('drop-off misses at effective accept', text)
        self.assertEqual(code, 0)

    def test_open_set_without_unlisted_photos(self):
        _, out, _ = self.run_cli([pred('paper', ('paper', 0.9))], [])
        self.assertEqual(out['open_set'],
                         {'n': 0, 'accepted_at_effective': 0, 'wrong_bin_at_effective': 0})

    def test_at_effective_metrics_none_without_threshold(self):
        _, out, _ = self.run_cli(self.NO_THRESHOLD, [])
        self.assertIsNone(out['hazard_misses_at_effective'])
        self.assertIsNone(out['center_misses_at_effective'])
        self.assertIsNone(out['open_set']['accepted_at_effective'])

    def test_pred_config_defaults_and_all_met(self):
        preds = [pred('paper', ('paper', 0.9))]
        code, out, _ = self.run_cli(preds, ['--require', 'item=0.8,bin=0.9,coverage=0.65'])
        self.assertEqual(code, 0)
        self.assertEqual(out['missed'], [])


if __name__ == '__main__':
    unittest.main()
