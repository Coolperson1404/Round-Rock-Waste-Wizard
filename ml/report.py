#!/usr/bin/env python3
"""Score raw model predictions against a test set and pick an accept threshold.

Reads predictions.json (written by ml/predict.swift) and the class configs,
reports item/bin accuracy and hazardous misses, then sweeps confidence
thresholds to find the lowest one, at or above --confirm-below, whose wrong-bin
rate is at or under the cap.

A photo counts as answered ("accepted") at t when the top confidence is >= t
and the top label is not not_waste. With --hazard-net (item-model runs only) it
must also not trip the app's item-model hazard safety net: the top label's bin
is not hhw but a rank-2 or rank-3 label's bin (in the pred config) is hhw with
confidence >= --hazard-runner-up (default HAZARD_RUNNER_UP, kept equal to the
app's WasteModels.HAZARD_RUNNER_UP); the app then asks instead of answering.
The net is off by default because the app's material fallback has no such net:
scoring material or baseline predictions with it would hide real hazardous
misses. Forgetting the flag only makes the gate stricter.

Truth labels come from the test folder names. `unlisted__<bin>` (photos of items
outside the taxonomy, see ml/PHOTO_GUIDE.md) has bin <bin> without a config entry:
it counts toward bin accuracy, wrong-bin rate and hazardous misses, but not toward
item accuracy or coverage, and is summarised separately as `open_set`. Any other
truth label missing from the truth config is a config error.

When coverage is required, the open set must hold at least OPEN_SET_HHW_MIN
unlisted__hhw photos (--allow-no-open-set overrides).

Exit codes: 0 all targets met; 1 a target missed, a config error, or (coverage
required) too few unlisted__hhw photos; 3 coverage required but no threshold
meets the max wrong-bin rate or the rate at the effective accept threshold
exceeds it, or, whenever an effective accept threshold exists, a hazardous (hhw)
photo is answered into another bin at it (--allow-hazard-misses overrides that,
with a loud warning). 3 wins over 1.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)]
NOT_WASTE = 'not_waste'
TARGET_KEYS = {'item': 'item_top1', 'bin': 'bin_acc', 'coverage': 'coverage_at_effective'}
UNLISTED_PREFIX = 'unlisted__'
# CATS ids in www/index.html.
CATS = ('recycle', 'trash', 'center', 'hhw', 'brush', 'bulk', 'other')
# Must equal WasteModels.HAZARD_RUNNER_UP in www/waste-models.js (a test checks it).
HAZARD_RUNNER_UP = 0.10
# Fewest unlisted__hhw photos a coverage-gated run accepts.
OPEN_SET_HHW_MIN = 5


def bins(config):
    return {label: spec['bin'] for label, spec in config['classes'].items()}


def is_unlisted(label):
    return isinstance(label, str) and label.startswith(UNLISTED_PREFIX)


def truth_bins_for(truth_labels, config_bins):
    """config_bins plus a bin for every unlisted__<bin> truth label. Raises
    ValueError naming any other truth label the config does not know."""
    out = dict(config_bins)
    unknown = []
    for label in sorted(set(truth_labels)):
        if label in config_bins:
            continue
        if is_unlisted(label) and label[len(UNLISTED_PREFIX):] in CATS:
            out[label] = label[len(UNLISTED_PREFIX):]
            continue
        unknown.append(label)
    if unknown:
        raise ValueError(f'truth labels not in the truth config: {unknown}. Fix the config, '
                         f'or name open-set folders {UNLISTED_PREFIX}<bin> with <bin> in {list(CATS)}')
    return out


def _top(row):
    top = row.get('top') or []
    return top[0] if top else {'label': None, 'confidence': 0.0}


def _item_comparable(truth_bins, pred_bins):
    return set(pred_bins) <= set(truth_bins)


def score(preds, truth_bins, pred_bins):
    n = len(preds)
    comparable = _item_comparable(truth_bins, pred_bins)
    known = item1 = item3 = bin_ok = hazard = 0
    per = {}
    confusions = Counter()
    for row in preds:
        truth = row['truth']
        unlisted = is_unlisted(truth)
        labels = [t['label'] for t in row.get('top') or []]
        top = labels[0] if labels else None
        truth_bin = truth_bins.get(truth)
        pred_bin = pred_bins.get(top)
        is_item = top == truth
        is_bin = truth_bin is not None and truth_bin == pred_bin
        if not unlisted:
            # An unlisted item can never be item-correct; it is scored by bin only.
            known += 1
            item1 += is_item
            item3 += truth in labels[:3]
        bin_ok += is_bin
        if truth_bin == 'hhw' and pred_bin != 'hhw':
            hazard += 1
        cls = per.setdefault(truth, {'n': 0, 'ok': 0})
        cls['n'] += 1
        cls['ok'] += is_item if comparable and not unlisted else is_bin
        if not is_item:
            confusions[(truth, top)] += 1
    per_class = {k: {'n': v['n'], 'acc': v['ok'] / v['n']} for k, v in per.items()}
    trainable = [k for k in per_class if not is_unlisted(k)]
    worst = sorted(trainable, key=lambda k: (per_class[k]['acc'], k))[:5]
    top_conf = sorted(confusions.items(), key=lambda kv: (-kv[1], kv[0][0], str(kv[0][1])))[:10]
    return {
        'n': n,
        'item_top1': item1 / known if comparable and known else None,
        'item_top3': item3 / known if comparable and known else None,
        'bin_acc': bin_ok / n if n else 0.0,
        'per_class': per_class,
        'worst_classes': worst,
        'confusions': [[t, p, c] for (t, p), c in top_conf],
        'hazard_misses': hazard,
    }


def hazard_fires(row, pred_bins, hazard_runner_up=None):
    """The app's item-model hazard safety net: the top label is not hhw, but a
    rank-2 or rank-3 label is hhw with confidence >= hazard_runner_up, so the app
    asks. hazard_runner_up None means the net is off (never fires)."""
    tops = row.get('top') or []
    if hazard_runner_up is None or not tops or pred_bins.get(tops[0]['label']) == 'hhw':
        return False
    return any(pred_bins.get(r['label']) == 'hhw' and r['confidence'] >= hazard_runner_up
               for r in tops[1:3])


def _accepted(row, t, pred_bins, hazard_runner_up=None):
    """True when the app would answer this photo at threshold t."""
    top = _top(row)
    return (top['label'] is not None and top['label'] != NOT_WASTE and top['confidence'] >= t
            and not hazard_fires(row, pred_bins, hazard_runner_up))


def _is_known_waste(truth):
    return truth != NOT_WASTE and not is_unlisted(truth)


def _row_at(preds, truth_bins, pred_bins, t, hazard_runner_up=None):
    comparable = _item_comparable(truth_bins, pred_bins)
    # Coverage is about the known items: not_waste and unlisted photos are out of it.
    real = sum(1 for r in preds if _is_known_waste(r['truth']))
    accepted = covered = wrong = item_ok = item_n = 0
    for r in preds:
        if not _accepted(r, t, pred_bins, hazard_runner_up):
            continue
        top = _top(r)
        accepted += 1
        if _is_known_waste(r['truth']):
            covered += 1
        truth_bin = truth_bins.get(r['truth'])
        if truth_bin is None or truth_bin != pred_bins.get(top['label']):
            wrong += 1
        if not is_unlisted(r['truth']):
            item_n += 1
            item_ok += top['label'] == r['truth']
    return {
        't': t,
        'coverage': covered / max(1, real),
        'wrong_bin_rate': wrong / max(1, accepted),
        'item_acc': item_ok / max(1, item_n) if comparable else None,
        'accepted': accepted,
    }


def misses_at(preds, truth_bins, pred_bins, t, bin_id, hazard_runner_up=None):
    """Photos accepted at t whose truth bin is bin_id but whose answer is another bin."""
    return sum(1 for r in preds
               if _accepted(r, t, pred_bins, hazard_runner_up)
               and truth_bins.get(r['truth']) == bin_id
               and pred_bins.get(_top(r)['label']) != bin_id)


def open_set_at(preds, truth_bins, pred_bins, t, hazard_runner_up=None):
    """Unlisted look-alikes: how many there are (hhw_n of them hazardous), how many
    are answered at t, and how many of those answers are in the wrong bin (counts,
    not rates)."""
    rows = [r for r in preds if is_unlisted(r['truth'])]
    hhw_n = sum(1 for r in rows if r['truth'] == UNLISTED_PREFIX + 'hhw')
    if t is None:
        return {'n': len(rows), 'hhw_n': hhw_n,
                'accepted_at_effective': None, 'wrong_bin_at_effective': None}
    accepted = [r for r in rows if _accepted(r, t, pred_bins, hazard_runner_up)]
    wrong = sum(1 for r in accepted
                if truth_bins.get(r['truth']) != pred_bins.get(_top(r)['label']))
    return {'n': len(rows), 'hhw_n': hhw_n,
            'accepted_at_effective': len(accepted), 'wrong_bin_at_effective': wrong}


def sweep(preds, truth_bins, pred_bins, hazard_runner_up=None):
    return [_row_at(preds, truth_bins, pred_bins, t, hazard_runner_up) for t in THRESHOLDS]


def recommend_threshold(rows, max_wrong, confirm_below):
    """Smallest grid t >= confirm_below with at least 1 accepted photo and a
    wrong-bin rate at or under the cap. The app ships exactly this t (it is never
    below confirm_below), so the rate at t is what users see; a worse rate at a
    higher, unshipped t does not matter."""
    for row in rows:
        if (row['t'] >= confirm_below - 1e-9 and row['accepted'] >= 1
                and row['wrong_bin_rate'] <= max_wrong):
            return row['t']
    return None


def check_targets(report, require):
    missed = []
    for name, minimum in require.items():
        value = report.get(TARGET_KEYS[name])
        if value is None or value < minimum:
            missed.append(name)
    return missed


def _parse_require(text):
    out = {}
    for part in filter(None, (p.strip() for p in text.split(','))):
        key, _, val = part.partition('=')
        if key not in TARGET_KEYS or not val:
            raise argparse.ArgumentTypeError(f'bad --require entry {part!r}')
        out[key] = float(val)
    return out


def _fmt(v):
    return 'n/a' if v is None else f'{v:.3f}'


def _count(v):
    return 'n/a' if v is None else str(v)


def _markdown(rep):
    lines = [
        '# Evaluation report',
        '',
        f"- photos: {rep['n']}",
        f"- item top-1: {_fmt(rep['item_top1'])}   top-3: {_fmt(rep['item_top3'])}",
        f"- bin accuracy: {_fmt(rep['bin_acc'])}",
        f"- hazardous misses (hhw): {rep['hazard_misses']}",
        f"- hazardous misses at effective accept (hhw, gated): {_count(rep['hazard_misses_at_effective'])}",
        f"- drop-off misses at effective accept (center, informational): {_count(rep['center_misses_at_effective'])}",
        f"- open set (unlisted look-alikes): {rep['open_set']['n']} photos "
        f"({rep['open_set']['hhw_n']} hazardous), "
        f"{_count(rep['open_set']['accepted_at_effective'])} answered at effective accept, "
        f"{_count(rep['open_set']['wrong_bin_at_effective'])} of those in the wrong bin",
        f"- recommended threshold (wrong-bin <= {rep['max_wrong_bin']}, "
        f"at or above confirm-below): {rep['recommended']}",
        f"- hazard runner-up net (item model; ask when an hhw rank 2-3 guess scores >=): "
        f"{rep['hazard_runner_up'] if rep['hazard_net'] else 'off'}",
        f"- effective accept (max with confirm-below {rep['confirm_below']}): {rep['effective_accept']}",
        f"- coverage at effective accept: {_fmt(rep['coverage_at_effective'])}",
        f"- wrong-bin at effective accept: {_fmt(rep['wrong_bin_at_effective'])}",
        f"- targets: {rep['require'] or 'none'}; missed: {rep['missed'] or 'none'}",
        '',
        '| t | coverage | wrong-bin | item acc |',
        '|---|---|---|---|',
    ]
    for r in rep['sweep']:
        lines.append(f"| {r['t']} | {_fmt(r['coverage'])} | {_fmt(r['wrong_bin_rate'])} | {_fmt(r['item_acc'])} |")
    lines += ['', 'Worst classes: ' + (', '.join(rep['worst_classes']) or 'none')]
    if rep['confusions']:
        lines += ['', 'Top confusions (truth -> predicted: count):']
        lines += [f'- {t} -> {p}: {c}' for t, p, c in rep['confusions']]
    return '\n'.join(lines)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--truth-config', required=True)
    ap.add_argument('--pred-config')
    ap.add_argument('--max-wrong-bin', type=float, default=0.02)
    ap.add_argument('--confirm-below', type=float, default=0.60)
    ap.add_argument('--hazard-net', action='store_true',
                    help="apply the app's item-model hazard runner-up net (item-model "
                         'predictions only; never for material or baseline runs)')
    ap.add_argument('--hazard-runner-up', type=float, default=HAZARD_RUNNER_UP,
                    help='with --hazard-net, the app asks when an hhw rank-2/3 guess '
                         'scores at least this')
    ap.add_argument('--require', type=_parse_require, default={})
    ap.add_argument('--allow-hazard-misses', action='store_true',
                    help='do not exit 3 on hazardous misses at the effective accept (warns)')
    ap.add_argument('--allow-no-open-set', action='store_true',
                    help=f'with coverage required, accept fewer than {OPEN_SET_HHW_MIN} '
                         'unlisted__hhw photos')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    hr = args.hazard_runner_up if args.hazard_net else None

    preds = json.loads(Path(args.predictions).read_text())
    try:
        truth_bins = truth_bins_for((r['truth'] for r in preds),
                                    bins(json.loads(Path(args.truth_config).read_text())))
    except ValueError as e:
        print(f'config error: {e}', file=sys.stderr)
        return 1
    pred_cfg = args.pred_config or args.truth_config
    pred_bins = bins(json.loads(Path(pred_cfg).read_text()))

    rep = score(preds, truth_bins, pred_bins)
    rows = sweep(preds, truth_bins, pred_bins, hr)
    recommended = recommend_threshold(rows, args.max_wrong_bin, args.confirm_below)
    effective = None if recommended is None else max(recommended, args.confirm_below)
    coverage = wrong_at_effective = hazard_at_effective = center_at_effective = None
    if effective is not None:
        row = next((r for r in rows if r['t'] == effective), None)
        row = row or _row_at(preds, truth_bins, pred_bins, effective, hr)
        coverage = row['coverage']
        wrong_at_effective = row['wrong_bin_rate']
        hazard_at_effective = misses_at(preds, truth_bins, pred_bins, effective, 'hhw', hr)
        center_at_effective = misses_at(preds, truth_bins, pred_bins, effective, 'center', hr)
    rep.update({
        'sweep': rows,
        'max_wrong_bin': args.max_wrong_bin,
        'confirm_below': args.confirm_below,
        'hazard_net': args.hazard_net,
        'hazard_runner_up': hr,
        'recommended': recommended,
        'effective_accept': effective,
        'coverage_at_effective': coverage,
        'wrong_bin_at_effective': wrong_at_effective,
        'hazard_misses_at_effective': hazard_at_effective,
        'center_misses_at_effective': center_at_effective,
        'open_set': open_set_at(preds, truth_bins, pred_bins, effective, hr),
        'require': args.require,
    })
    rep['missed'] = check_targets(rep, args.require)

    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(_markdown(rep))

    unsafe = 'coverage' in args.require and (
        recommended is None or wrong_at_effective > args.max_wrong_bin)
    if hazard_at_effective:
        if args.allow_hazard_misses:
            print(f'WARNING: --allow-hazard-misses: {hazard_at_effective} hazardous (hhw) '
                  f'photo(s) are answered into another bin at effective accept {effective}. '
                  'This model sends hazardous items to the wrong place; do NOT ship it.',
                  file=sys.stderr)
        else:
            unsafe = True

    open_set_short = False
    hhw_n = rep['open_set']['hhw_n']
    if 'coverage' in args.require and hhw_n < OPEN_SET_HHW_MIN:
        if args.allow_no_open_set:
            print(f'warning: --allow-no-open-set: only {hhw_n} unlisted__hhw photo(s); '
                  'the open-set hazard check is not meaningful.', file=sys.stderr)
        else:
            open_set_short = True
            print(f'open set too small: {hhw_n} unlisted__hhw test photo(s), at least '
                  f'{OPEN_SET_HHW_MIN} are needed to check hazardous look-alikes '
                  '(ml/data/own-test-unlisted/hhw/, see ml/PHOTO_GUIDE.md). '
                  'Pass --allow-no-open-set to skip this.', file=sys.stderr)

    if unsafe:
        return 3
    return 1 if rep['missed'] or open_set_short else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
