#!/usr/bin/env python3
"""Turn a JSON config of image sources into Create ML training folders.

    python3 ml/prepare_data.py --config ml/sources.json --out ml/build/items
        [--drop-underfilled] [--check] [--root <repo root>]

Output: <out>/{train,test}/<label>/<tag>_<stem>[_<annId>].jpg and <out>/summary.json.
Exit codes: 0 ok, 1 config error, 2 train/test leakage (identical file bytes).
Dependencies: stdlib + Pillow. HEIC is converted with macOS `sips`.
"""
import argparse
import collections
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

MAX_SIDE = 384
MIN_CROP = 64
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.heic', '.webp', '.bmp'}
SPLITS = {'train', 'test', 'hash'}

# ann_id is an optional extra so COCO crops from one image get distinct filenames.
Sample = collections.namedtuple('Sample', 'label split src bbox tag ann_id')
Sample.__new__.__defaults__ = (None,)


def hash_split(rel_path, test_percent):
    digest = hashlib.sha1(rel_path.encode('utf-8')).hexdigest()
    return 'test' if int(digest, 16) % 100 < test_percent else 'train'


def crop_box(img, bbox, pad=0.10):
    x, y, w, h = bbox
    px, py = w * pad, h * pad
    left = max(0, int(round(x - px)))
    top = max(0, int(round(y - py)))
    right = min(img.width, int(round(x + w + px)))
    bottom = min(img.height, int(round(y + h + py)))
    if right - left < MIN_CROP or bottom - top < MIN_CROP:
        return None
    return img.crop((left, top, right, bottom))


def shrink(img, max_side=MAX_SIDE):
    longest = max(img.size)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                      Image.LANCZOS)


def load_image(path, max_side=MAX_SIDE):
    """Open, apply EXIF rotation, convert to RGB, downsize (never upscale).
    max_side=None keeps full resolution (COCO boxes are in full-res coordinates)."""
    path = Path(path)
    if path.suffix.lower() == '.heic':
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'converted.jpg'
            r = subprocess.run(['sips', '-s', 'format', 'jpeg', str(path), '--out', str(out)],
                               capture_output=True)
            if r.returncode != 0 or not out.exists():
                raise OSError(f'sips could not convert {path}')
            with Image.open(out) as im:
                img = ImageOps.exif_transpose(im).convert('RGB')
    else:
        with Image.open(path) as im:
            img = ImageOps.exif_transpose(im).convert('RGB')
    return shrink(img, max_side) if max_side else img


def _rel(path, root):
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _tag(text):
    return re.sub(r'[^A-Za-z0-9-]+', '-', text).strip('-') or 'src'


def _images_in(folder):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob('*')
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                  and not any(part.startswith('.') for part in p.relative_to(folder).parts))


def _resolve_split(split, rel, test_percent):
    return hash_split(rel, test_percent) if split == 'hash' else split


def _validate(config):
    if not isinstance(config.get('classes'), dict) or not config['classes']:
        raise ValueError('config needs a non-empty "classes" object')
    for key in ('testPercent', 'minTrain', 'minTest'):
        if not isinstance(config.get(key), int):
            raise ValueError(f'config "{key}" must be an integer')
    for label, spec in config['classes'].items():
        for src in spec.get('sources', []):
            kind = src.get('type')
            if kind not in ('folder', 'coco'):
                raise ValueError(f'{label}: unknown source type {kind!r}')
            if src.get('split') not in SPLITS:
                raise ValueError(f'{label}: split must be one of {sorted(SPLITS)}')
            need = ('path',) if kind == 'folder' else ('annotations', 'images', 'categories')
            for key in need:
                if key not in src:
                    raise ValueError(f'{label}: {kind} source is missing "{key}"')


def _coco_samples(label, src, root, test_percent, cache):
    ann_path = root / src['annotations']
    if not ann_path.is_file():
        raise ValueError(f'{label}: COCO annotations not found: {ann_path}')
    if ann_path not in cache:
        with open(ann_path, encoding='utf-8') as fh:
            data = json.load(fh)
        cats = {c['id']: c['name'] for c in data.get('categories', [])}
        files = {i['id']: i['file_name'] for i in data.get('images', [])}
        cache[ann_path] = (cats, files, data.get('annotations', []))
    cats, files, anns = cache[ann_path]
    available = sorted(set(cats.values()))
    wanted = set(src['categories'])
    unknown = sorted(wanted - set(available))
    if unknown:
        raise ValueError(f'{label}: unknown COCO categories {unknown} in {ann_path}. '
                         f'Available: {available}')
    wanted_ids = {cid for cid, name in cats.items() if name in wanted}
    images_dir = root / src['images']
    out = []
    for ann in anns:
        if ann['category_id'] not in wanted_ids or ann['image_id'] not in files:
            continue
        name = files[ann['image_id']]
        split = _resolve_split(src['split'], name, test_percent)
        out.append(Sample(label, split, images_dir / name, list(ann['bbox']), 'coco', ann['id']))
    return out


def collect(config, root):
    root = Path(root)
    _validate(config)
    tp = config['testPercent']
    cache = {}
    samples = []
    for label, spec in config['classes'].items():
        for src in spec.get('sources', []):
            if src['type'] == 'coco':
                samples += _coco_samples(label, src, root, tp, cache)
                continue
            folder = root / src['path']
            tag = _tag(Path(src['path']).name)
            for p in _images_in(folder):
                split = _resolve_split(src['split'], _rel(p, root), tp)
                samples.append(Sample(label, split, p, None, tag))
        if config.get('ownPhotos'):
            for rel_dir, split in (('ml/data/own', 'train'), ('ml/data/own-test', 'test')):
                for p in _images_in(root / rel_dir / label):
                    samples.append(Sample(label, split, p, None, 'own'))
    return samples


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def find_leaks(samples):
    """Pairs (train_src, test_src) whose files have identical bytes."""
    by_split = {'train': {}, 'test': {}}
    digests = {}
    for s in samples:
        if s.split not in by_split or not s.src.is_file():
            continue
        if s.src not in digests:
            digests[s.src] = _sha256(s.src)
        by_split[s.split].setdefault(digests[s.src], s.src)
    return sorted((by_split['train'][d], by_split['test'][d])
                  for d in by_split['train'].keys() & by_split['test'].keys())


def summarize(counts, min_train, min_test):
    under = [label for label, c in counts.items()
             if c.get('train', 0) < min_train or c.get('test', 0) < min_test]
    return {'counts': counts, 'underfilled': sorted(under)}


def _out_name(sample, dest_dir):
    stem = f'{sample.tag}_{sample.src.stem}'
    if sample.ann_id is not None:
        stem += f'_{sample.ann_id}'
    name, n = f'{stem}.jpg', 1
    while (dest_dir / name).exists():
        n += 1
        name = f'{stem}-{n}.jpg'
    return name


def _write_samples(samples, out, labels):
    counts = {label: {'train': 0, 'test': 0} for label in labels}
    unreadable = skipped = 0
    for s in samples:
        try:
            if s.bbox is not None:
                cropped = crop_box(load_image(s.src, max_side=None), s.bbox)
                if cropped is None:
                    skipped += 1
                    continue
                img = shrink(cropped)
            else:
                img = load_image(s.src)
        except (OSError, ValueError, SyntaxError):
            unreadable += 1
            continue
        dest = out / s.split / s.label
        dest.mkdir(parents=True, exist_ok=True)
        img.save(dest / _out_name(s, dest), 'JPEG', quality=90)
        counts[s.label][s.split] += 1
    return counts, unreadable, skipped


def _source_dirs(config, root):
    """Every directory the config reads from, including implicit own-photo dirs."""
    dirs = []
    if config.get('ownPhotos'):
        dirs += [root / 'ml/data/own', root / 'ml/data/own-test']
    for spec in config['classes'].values():
        for src in spec.get('sources', []):
            dirs.append(root / (src['path'] if src['type'] == 'folder' else src['images']))
    return [d.resolve() for d in dirs]


def _is_within(child, parent):
    return child == parent or parent in child.parents


def unsafe_out_reason(out_arg, root, config):
    """Why --out must be refused (it is wiped and rewritten), or None if safe."""
    if not str(out_arg).strip():
        return '--out is empty'
    out, root = Path(out_arg).resolve(), Path(root).resolve()
    if out == Path.cwd().resolve():
        return f'--out {out} is the current directory'
    if _is_within(root, out):
        return f'--out {out} is the repo root or one of its ancestors'
    for src in _source_dirs(config, root):
        if _is_within(out, src):
            return f'--out {out} is inside the source folder {src}'
        if _is_within(src, out):
            return f'--out {out} contains the source folder {src}'
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--root', default=str(Path(__file__).resolve().parent.parent),
                    help='base for relative paths in the config (default: repo root)')
    ap.add_argument('--drop-underfilled', action='store_true')
    ap.add_argument('--check', action='store_true',
                    help='validate the config and sources; write nothing')
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out)

    try:
        with open(args.config, encoding='utf-8') as fh:
            config = json.load(fh)
        samples = collect(config, root)
    except (OSError, ValueError) as e:
        print(f'config error: {e}', file=sys.stderr)
        return 1

    labels = list(config['classes'])
    planned = {label: {'train': 0, 'test': 0} for label in labels}
    for s in samples:
        planned[s.label][s.split] += 1

    if args.check:
        print(f'config OK: {len(labels)} classes, {len(samples)} candidate images')
        for label, c in planned.items():
            print(f'  {label}: train {c["train"]}, test {c["test"]}')
        return 0

    reason = unsafe_out_reason(args.out, root, config)
    if reason:
        print(f'refusing to use --out: {reason}. Nothing was deleted or written.',
              file=sys.stderr)
        return 1

    leaks = find_leaks(samples)
    if leaks:
        print(f'LEAKAGE: {len(leaks)} file(s) are in both train and test:', file=sys.stderr)
        for tr, te in leaks[:20]:
            print(f'  train {tr}\n  test  {te}', file=sys.stderr)
        return 2

    for split in ('train', 'test'):
        shutil.rmtree(out / split, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    counts, unreadable, skipped = _write_samples(samples, out, labels)
    summary = summarize(counts, config['minTrain'], config['minTest'])
    summary['unreadable'] = unreadable

    if args.drop_underfilled:
        for label in summary['underfilled']:
            for split in ('train', 'test'):
                shutil.rmtree(out / split / label, ignore_errors=True)

    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for label, c in counts.items():
        flag = '  <-- short' if label in summary['underfilled'] else ''
        print(f'{label}: train {c["train"]}/{config["minTrain"]}, '
              f'test {c["test"]}/{config["minTest"]}{flag}')
    print(f'{len(summary["underfilled"])} underfilled, {unreadable} unreadable, '
          f'{skipped} crops under {MIN_CROP}px skipped')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
