import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

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
        self.assertEqual(report.recommend_threshold(self.rows(), 0.02, 0.30), 0.55)
        self.assertEqual(report.recommend_threshold(self.rows(), 0.02, 0.60), 0.6)

    def test_recommend_none_when_nothing_accepted_safely(self):
        preds = [pred('aerosol_can', ('aluminum_can', 0.99))]
        rows = report.sweep(preds, BINS, BINS)
        self.assertIsNone(report.recommend_threshold(rows, 0.02, 0.30))

    def test_recommend_requires_an_accepted_photo(self):
        # Nothing is accepted above 0.5, so those rows have 0 wrong but 0 accepted.
        preds = [pred('paper', ('paper', 0.5))]
        rows = report.sweep(preds, BINS, BINS)
        self.assertEqual(report.recommend_threshold(rows, 0.02, 0.30), 0.3)
        # Nothing at or above confirm_below is accepted.
        self.assertIsNone(report.recommend_threshold(rows, 0.02, 0.60))

    def test_recommend_non_monotone_wrong_bin_rate(self):
        # Wrong-bin photo at 0.35, correct photos over 0.40-0.59, wrong-bin at 0.9.
        # The rule is first fit at or above confirm_below: what ships is the rate at
        # the chosen t, not at higher ones. From 0.30 that is 0.30 (2/102 <= 2 %).
        # From 0.60 up the only accepted photo is the 0.9 miss (rate 1.0): none.
        preds = [pred('aerosol_can', ('aluminum_can', 0.35))]
        preds += [pred('paper', ('paper', 0.40 + 0.0019 * i)) for i in range(100)]
        preds += [pred('aerosol_can', ('aluminum_can', 0.9))]
        rows = report.sweep(preds, BINS, BINS)
        by_t = {r['t']: r for r in rows}
        self.assertLessEqual(by_t[0.3]['wrong_bin_rate'], 0.02)
        self.assertEqual(by_t[0.9]['wrong_bin_rate'], 1.0)
        self.assertEqual(report.recommend_threshold(rows, 0.02, 0.30), 0.3)
        self.assertIsNone(report.recommend_threshold(rows, 0.02, 0.60))

    def test_recommend_ignores_a_bad_rate_above_the_chosen_t(self):
        # Reviewer repro: 1.2 % wrong-bin at 0.60-0.90 and a single mistake at 0.95
        # (rate 1.0 there). The old "safe at every higher t" rule gave None.
        preds = [pred('paper', ('paper', 0.92)) for _ in range(82)]
        preds += [pred('not_waste', ('paper', 0.96))]
        rows = report.sweep(preds, BINS, BINS)
        by_t = {r['t']: r for r in rows}
        for t in (0.6, 0.7, 0.8, 0.9):
            self.assertAlmostEqual(by_t[t]['wrong_bin_rate'], 1 / 83)
        self.assertEqual(by_t[0.95]['wrong_bin_rate'], 1.0)
        self.assertEqual(report.recommend_threshold(rows, 0.02, 0.60), 0.6)

    def test_recommend_never_below_confirm_below(self):
        preds = [pred('paper', ('paper', 0.9))]
        rows = report.sweep(preds, BINS, BINS)
        self.assertEqual(report.recommend_threshold(rows, 0.02, 0.30), 0.3)
        self.assertEqual(report.recommend_threshold(rows, 0.02, 0.60), 0.6)
        self.assertEqual(report.recommend_threshold(rows, 0.02, 0.62), 0.65)

    def test_recommend_first_fit_at_or_above_confirm_below(self):
        # 100 correct at 0.30-0.49, one miss at 0.50, one correct at 0.9.
        # t=0.30 is 1/102 and t=0.50 is 1/2. First fit from 0.30 is 0.30; from
        # 0.45 it skips the 0.45 and 0.50 rows (rate 1/4 and 1/2) and picks 0.55.
        preds = [pred('paper', ('paper', 0.30 + 0.0019 * i)) for i in range(100)]
        preds += [pred('aerosol_can', ('aluminum_can', 0.50))]
        preds += [pred('paper', ('paper', 0.9))]
        rows = report.sweep(preds, BINS, BINS)
        by_t = {r['t']: r for r in rows}
        self.assertLessEqual(by_t[0.3]['wrong_bin_rate'], 0.02)
        self.assertEqual(by_t[0.5]['wrong_bin_rate'], 0.5)
        self.assertEqual(report.recommend_threshold(rows, 0.02, 0.30), 0.3)
        self.assertEqual(report.recommend_threshold(rows, 0.02, 0.45), 0.55)

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
        self.assertIsNone(report.recommend_threshold(rows, 0.02, 0.30))

    def test_not_waste_never_accepted(self):
        preds = [pred('not_waste', ('not_waste', 0.99))]
        for r in report.sweep(preds, BINS, BINS):
            self.assertEqual(r['coverage'], 0.0)
            self.assertEqual(r['wrong_bin_rate'], 0.0)


class HazardRunnerUpTests(unittest.TestCase):
    def test_constant(self):
        self.assertEqual(report.HAZARD_RUNNER_UP, 0.10)

    def test_hhw_runner_up_at_012_is_never_accepted(self):
        preds = [pred('aluminum_can', ('aluminum_can', 0.85), ('aerosol_can', 0.12))]
        for r in report.sweep(preds, BINS, BINS):
            self.assertEqual(r['accepted'], 0, r['t'])
            self.assertEqual(r['coverage'], 0.0, r['t'])

    def test_hhw_runner_up_below_threshold_is_accepted(self):
        preds = [pred('aluminum_can', ('aluminum_can', 0.85), ('aerosol_can', 0.08))]
        by_t = {r['t']: r for r in report.sweep(preds, BINS, BINS)}
        self.assertEqual(by_t[0.85]['accepted'], 1)
        self.assertEqual(by_t[0.9]['accepted'], 0)

    def test_rank_3_counts_rank_4_does_not(self):
        rank3 = [pred('aluminum_can', ('aluminum_can', 0.70), ('food_can', 0.15),
                      ('aerosol_can', 0.12))]
        rank4 = [pred('aluminum_can', ('aluminum_can', 0.62), ('food_can', 0.13),
                      ('paper', 0.12), ('aerosol_can', 0.11))]
        self.assertEqual(report.sweep(rank3, BINS, BINS)[0]['accepted'], 0)
        self.assertEqual(report.sweep(rank4, BINS, BINS)[0]['accepted'], 1)

    def test_hhw_top_is_not_downgraded(self):
        preds = [pred('aerosol_can', ('aerosol_can', 0.80), ('aluminum_can', 0.15))]
        self.assertEqual(report.sweep(preds, BINS, BINS)[0]['accepted'], 1)

    def test_threshold_is_a_parameter(self):
        preds = [pred('aluminum_can', ('aluminum_can', 0.85), ('aerosol_can', 0.08))]
        self.assertEqual(report.sweep(preds, BINS, BINS, 0.05)[0]['accepted'], 0)

    def test_uses_pred_config_bins(self):
        # Baseline material labels: 'battery' is hhw in the pred config only.
        pred_bins = {'metal': 'recycle', 'battery': 'hhw'}
        preds = [pred('aluminum_can', ('metal', 0.85), ('battery', 0.12))]
        self.assertEqual(report.sweep(preds, BINS, pred_bins)[0]['accepted'], 0)

    def test_hazard_ask_is_not_a_hazard_miss(self):
        # The app asks, so a hazardous photo that is top-1 wrong is not answered.
        preds = [pred('aerosol_can', ('aluminum_can', 0.85), ('aerosol_can', 0.12))]
        self.assertEqual(report.misses_at(preds, BINS, BINS, 0.6, 'hhw'), 0)
        self.assertEqual(report.score(preds, BINS, BINS)['hazard_misses'], 1)


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
            buf, err = io.StringIO(), io.StringIO()
            with redirect_stdout(buf), redirect_stderr(err):
                code = report.main(argv)
            self.stderr = err.getvalue()
            out = json.loads((d / 'out.json').read_text())
            return code, out, buf.getvalue()

    # Five open-set hazardous look-alikes the model sends to hhw: they satisfy the
    # open-set minimum without changing any other number the tests look at.
    OPEN_SET = [pred('unlisted__hhw', ('aerosol_can', 0.95), file=f'u{i}.jpg') for i in range(5)]

    # An unsafe prediction at the top of the grid: no threshold meets 2 % wrong-bin.
    NO_THRESHOLD = [
        pred('aerosol_can', ('aluminum_can', 0.99)),
        pred('paper', ('paper', 0.9)),
    ]

    def test_exit_3_when_coverage_required_and_no_threshold(self):
        code, out, _ = self.run_cli(self.NO_THRESHOLD + self.OPEN_SET, ['--require', 'coverage=0.5'])
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
        ] + self.OPEN_SET
        code, out, text = self.run_cli(preds, ['--require', 'coverage=0.5'])
        self.assertEqual(out['recommended'], 0.6)
        self.assertEqual(out['effective_accept'], 0.6)
        self.assertEqual(out['coverage_at_effective'], 0.5)
        self.assertEqual(out['wrong_bin_at_effective'], 0.0)
        self.assertIn('wrong-bin at effective accept', text)
        self.assertEqual(code, 0)
        self.assertIn('0.6', text)
        code, _, _ = self.run_cli(preds, ['--require', 'coverage=0.6'])
        self.assertEqual(code, 1)

    def test_recommended_is_on_grid_at_or_above_off_grid_confirm_below(self):
        # 0.63 miss is 1 of 102 at t=0.60, but t must be >= 0.62, so 0.65 (0/1).
        preds = [pred('paper', ('paper', 0.605)) for _ in range(100)]
        preds += [pred('aerosol_can', ('aluminum_can', 0.63)), pred('paper', ('paper', 0.9))]
        preds += self.OPEN_SET
        code, out, _ = self.run_cli(preds, ['--confirm-below', '0.62', '--require', 'coverage=0.005'])
        self.assertEqual(out['recommended'], 0.65)
        self.assertEqual(out['effective_accept'], 0.65)
        self.assertEqual(out['wrong_bin_at_effective'], 0.0)
        self.assertEqual(out['hazard_misses_at_effective'], 0)
        self.assertEqual(code, 0)

    def test_exit_3_when_wrong_bin_exceeds_cap_at_effective_accept(self):
        # recommend_threshold now never returns a t below confirm_below, so this
        # gate is defensive: force a recommendation under it to prove it still bites.
        preds = [pred('paper', ('paper', 0.605)) for _ in range(100)]
        preds += [pred('not_waste', ('paper', 0.63)), pred('paper', ('paper', 0.9))]
        preds += self.OPEN_SET
        with mock.patch.object(report, 'recommend_threshold', return_value=0.3):
            code, out, _ = self.run_cli(preds, ['--confirm-below', '0.62',
                                                '--require', 'coverage=0.005'])
        self.assertEqual(out['effective_accept'], 0.62)
        # 0.63 miss, 0.9 paper and the 5 open-set photos are accepted: 1/7 wrong.
        self.assertAlmostEqual(out['wrong_bin_at_effective'], 1 / 7)
        self.assertEqual(out['hazard_misses_at_effective'], 0)
        self.assertEqual(code, 3)

    def test_effective_accept_with_off_grid_confirm_below(self):
        # The recommendation is the first grid t at or above confirm-below.
        preds = [pred('paper', ('paper', 0.9))]
        _, out, _ = self.run_cli(preds, ['--confirm-below', '0.62'])
        self.assertEqual(out['recommended'], 0.65)
        self.assertEqual(out['effective_accept'], 0.65)
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
        preds += self.OPEN_SET
        code, out, text = self.run_cli(preds, ['--require', 'item=0.80,bin=0.92,coverage=0.65'])
        self.assertLessEqual(out['wrong_bin_at_effective'], 0.02)
        self.assertEqual(out['missed'], [])
        self.assertEqual(out['hazard_misses_at_effective'], 2)
        self.assertEqual(code, 3)
        self.assertIn('hazardous misses at effective accept', text)

    def test_hazard_gate_runs_without_coverage_required(self):
        preds = [pred('paper', ('paper', 0.95)) for _ in range(100)]
        preds += [pred('aerosol_can', ('aluminum_can', 0.95)) for _ in range(2)]
        code, out, _ = self.run_cli(preds, ['--require', 'item=0.80,bin=0.92'])
        self.assertEqual(out['hazard_misses_at_effective'], 2)
        self.assertEqual(code, 3)
        code, out, _ = self.run_cli(preds, [])
        self.assertEqual(code, 3)

    def test_allow_hazard_misses_overrides_with_a_loud_warning(self):
        preds = [pred('paper', ('paper', 0.95)) for _ in range(100)]
        preds += [pred('aerosol_can', ('aluminum_can', 0.95)) for _ in range(2)]
        code, _, _ = self.run_cli(preds, ['--require', 'item=0.80,bin=0.92',
                                          '--allow-hazard-misses'])
        self.assertEqual(code, 0)
        self.assertIn('WARNING', self.stderr)
        self.assertIn('2 hazardous', self.stderr)

    def test_hazard_gate_skipped_without_effective_accept(self):
        code, out, _ = self.run_cli(self.NO_THRESHOLD, ['--require', 'item=0.4'])
        self.assertIsNone(out['hazard_misses_at_effective'])
        self.assertEqual(code, 0)

    def test_unlisted_hhw_counts_as_hazard_miss_at_effective(self):
        preds = [pred('paper', ('paper', 0.95)) for _ in range(100)]
        preds += [pred('unlisted__hhw', ('food_can', 0.95))]
        code, out, _ = self.run_cli(preds, ['--require', 'coverage=0.65', '--allow-no-open-set'])
        self.assertEqual(out['hazard_misses_at_effective'], 1)
        self.assertEqual(out['item_top1'], 1.0)
        self.assertEqual(out['open_set'],
                         {'n': 1, 'hhw_n': 1, 'accepted_at_effective': 1, 'wrong_bin_at_effective': 1})
        self.assertEqual(code, 3)

    def test_hazard_miss_below_effective_not_counted(self):
        preds = [pred('paper', ('paper', 0.95)) for _ in range(100)]
        preds += [pred('aerosol_can', ('aluminum_can', 0.5))] + self.OPEN_SET
        code, out, _ = self.run_cli(preds, ['--require', 'coverage=0.65'])
        self.assertEqual(out['effective_accept'], 0.6)
        self.assertEqual(out['hazard_misses'], 1)
        self.assertEqual(out['hazard_misses_at_effective'], 0)
        self.assertEqual(code, 0)

    def test_center_misses_at_effective_is_informational(self):
        preds = [pred('paper', ('paper', 0.95)) for _ in range(100)]
        preds += [pred('styrofoam_food', ('paper', 0.95))] + self.OPEN_SET
        code, out, text = self.run_cli(preds, ['--require', 'coverage=0.65'])
        self.assertEqual(out['center_misses_at_effective'], 1)
        self.assertEqual(out['hazard_misses_at_effective'], 0)
        self.assertIn('drop-off misses at effective accept', text)
        self.assertEqual(code, 0)

    def test_open_set_without_unlisted_photos(self):
        _, out, _ = self.run_cli([pred('paper', ('paper', 0.9))], [])
        self.assertEqual(out['open_set'],
                         {'n': 0, 'hhw_n': 0, 'accepted_at_effective': 0, 'wrong_bin_at_effective': 0})

    def test_open_set_minimum_when_coverage_required(self):
        preds = [pred('paper', ('paper', 0.95)) for _ in range(10)]
        preds += self.OPEN_SET[:4]
        code, out, _ = self.run_cli(preds, ['--require', 'coverage=0.65'])
        self.assertEqual(code, 1)
        self.assertEqual(out['open_set']['hhw_n'], 4)
        self.assertIn('4', self.stderr)
        self.assertIn('at least 5', self.stderr)
        self.assertIn('unlisted__hhw', self.stderr)
        code, _, _ = self.run_cli(preds, ['--require', 'coverage=0.65', '--allow-no-open-set'])
        self.assertEqual(code, 0)
        code, _, _ = self.run_cli(preds + self.OPEN_SET[4:], ['--require', 'coverage=0.65'])
        self.assertEqual(code, 0)

    def test_open_set_minimum_counts_only_hhw(self):
        preds = [pred('paper', ('paper', 0.95)) for _ in range(10)]
        preds += [pred('unlisted__trash', ('not_waste', 0.9), file=f't{i}.jpg') for i in range(5)]
        code, out, _ = self.run_cli(preds, ['--require', 'coverage=0.65'])
        self.assertEqual(out['open_set']['n'], 5)
        self.assertEqual(out['open_set']['hhw_n'], 0)
        self.assertEqual(code, 1)

    def test_open_set_minimum_not_needed_without_coverage(self):
        code, _, _ = self.run_cli([pred('paper', ('paper', 0.9))], ['--require', 'item=0.8'])
        self.assertEqual(code, 0)

    def test_at_effective_metrics_none_without_threshold(self):
        _, out, _ = self.run_cli(self.NO_THRESHOLD, [])
        self.assertIsNone(out['hazard_misses_at_effective'])
        self.assertIsNone(out['center_misses_at_effective'])
        self.assertIsNone(out['open_set']['accepted_at_effective'])

    def test_pred_config_defaults_and_all_met(self):
        preds = [pred('paper', ('paper', 0.9))] + self.OPEN_SET
        code, out, _ = self.run_cli(preds, ['--require', 'item=0.8,bin=0.9,coverage=0.65'])
        self.assertEqual(code, 0)
        self.assertEqual(out['missed'], [])


if __name__ == '__main__':
    unittest.main()
