#!/usr/bin/env python3
"""Score raw model predictions against a test set and pick an accept threshold.

Reads predictions.json (written by ml/predict.swift) and the class configs,
reports item/bin accuracy and hazardous misses, then sweeps confidence
thresholds to find the lowest one whose wrong-bin rate is at or under the cap.

Exit codes: 0 all targets met, 1 a target missed, 3 coverage required but no
threshold meets the max wrong-bin rate, or the rate at the effective accept
threshold exceeds it.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)]
NOT_WASTE = 'not_waste'
TARGET_KEYS = {'item': 'item_top1', 'bin': 'bin_acc', 'coverage': 'coverage_at_effective'}


def bins(config):
    return {label: spec['bin'] for label, spec in config['classes'].items()}


def _top(row):
    top = row.get('top') or []
    return top[0] if top else {'label': None, 'confidence': 0.0}


def _item_comparable(truth_bins, pred_bins):
    return set(pred_bins) <= set(truth_bins)


def score(preds, truth_bins, pred_bins):
    n = len(preds)
    comparable = _item_comparable(truth_bins, pred_bins)
    item1 = item3 = bin_ok = hazard = 0
    per = {}
    confusions = Counter()
    for row in preds:
        truth = row['truth']
        labels = [t['label'] for t in row.get('top') or []]
        top = labels[0] if labels else None
        truth_bin = truth_bins.get(truth)
        pred_bin = pred_bins.get(top)
        is_item = top == truth
        is_bin = truth_bin is not None and truth_bin == pred_bin
        item1 += is_item
        item3 += truth in labels[:3]
        bin_ok += is_bin
        if truth_bin == 'hhw' and pred_bin != 'hhw':
            hazard += 1
        cls = per.setdefault(truth, {'n': 0, 'ok': 0})
        cls['n'] += 1
        cls['ok'] += is_item if comparable else is_bin
        if not is_item:
            confusions[(truth, top)] += 1
    per_class = {k: {'n': v['n'], 'acc': v['ok'] / v['n']} for k, v in per.items()}
    worst = sorted(per_class, key=lambda k: (per_class[k]['acc'], k))[:5]
    top_conf = sorted(confusions.items(), key=lambda kv: (-kv[1], kv[0][0], str(kv[0][1])))[:10]
    return {
        'n': n,
        'item_top1': item1 / n if comparable and n else None,
        'item_top3': item3 / n if comparable and n else None,
        'bin_acc': bin_ok / n if n else 0.0,
        'per_class': per_class,
        'worst_classes': worst,
        'confusions': [[t, p, c] for (t, p), c in top_conf],
        'hazard_misses': hazard,
    }


def _row_at(preds, truth_bins, pred_bins, t):
    comparable = _item_comparable(truth_bins, pred_bins)
    real = sum(1 for r in preds if r['truth'] != NOT_WASTE)
    accepted = covered = wrong = item_ok = 0
    for r in preds:
        top = _top(r)
        if top['label'] is None or top['label'] == NOT_WASTE or top['confidence'] < t:
            continue
        accepted += 1
        if r['truth'] != NOT_WASTE:
            covered += 1
        truth_bin = truth_bins.get(r['truth'])
        if truth_bin is None or truth_bin != pred_bins.get(top['label']):
            wrong += 1
        item_ok += top['label'] == r['truth']
    return {
        't': t,
        'coverage': covered / max(1, real),
        'wrong_bin_rate': wrong / max(1, accepted),
        'item_acc': item_ok / max(1, accepted) if comparable else None,
        'accepted': accepted,
    }


def sweep(preds, truth_bins, pred_bins):
    return [_row_at(preds, truth_bins, pred_bins, t) for t in THRESHOLDS]


def recommend_threshold(rows, max_wrong):
    """Smallest grid t that is safe, where safe means the wrong-bin rate is at or
    under the cap at t (with at least 1 accepted photo) and at every higher grid t
    that accepts anything. The rate is not monotone in t, so first-fit would miss
    a bad rate at a threshold above the one it picked."""
    best = None
    for row in reversed(rows):
        if row['accepted'] >= 1 and row['wrong_bin_rate'] > max_wrong:
            break
        if row['accepted'] >= 1:
            best = row['t']
    return best


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


def _markdown(rep):
    lines = [
        '# Evaluation report',
        '',
        f"- photos: {rep['n']}",
        f"- item top-1: {_fmt(rep['item_top1'])}   top-3: {_fmt(rep['item_top3'])}",
        f"- bin accuracy: {_fmt(rep['bin_acc'])}",
        f"- hazardous misses (hhw): {rep['hazard_misses']}",
        f"- recommended threshold (wrong-bin <= {rep['max_wrong_bin']}): {rep['recommended']}",
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
    ap.add_argument('--require', type=_parse_require, default={})
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    preds = json.loads(Path(args.predictions).read_text())
    truth_bins = bins(json.loads(Path(args.truth_config).read_text()))
    pred_cfg = args.pred_config or args.truth_config
    pred_bins = bins(json.loads(Path(pred_cfg).read_text()))

    rep = score(preds, truth_bins, pred_bins)
    rows = sweep(preds, truth_bins, pred_bins)
    recommended = recommend_threshold(rows, args.max_wrong_bin)
    effective = None if recommended is None else max(recommended, args.confirm_below)
    coverage = wrong_at_effective = None
    if effective is not None:
        row = next((r for r in rows if r['t'] == effective), None)
        row = row or _row_at(preds, truth_bins, pred_bins, effective)
        coverage = row['coverage']
        wrong_at_effective = row['wrong_bin_rate']
    rep.update({
        'sweep': rows,
        'max_wrong_bin': args.max_wrong_bin,
        'confirm_below': args.confirm_below,
        'recommended': recommended,
        'effective_accept': effective,
        'coverage_at_effective': coverage,
        'wrong_bin_at_effective': wrong_at_effective,
        'require': args.require,
    })
    rep['missed'] = check_targets(rep, args.require)

    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(_markdown(rep))

    if 'coverage' in args.require and (
            recommended is None or wrong_at_effective > args.max_wrong_bin):
        return 3
    return 1 if rep['missed'] else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
