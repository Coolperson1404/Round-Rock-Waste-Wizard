#!/usr/bin/env python3
"""Turn a JSON config of image sources into Create ML training folders.

    python3 ml/prepare_data.py --config ml/sources.json --out ml/build/items
        [--drop-underfilled [--allow-drop-hazard]] [--check] [--root <repo root>]

Output: <out>/{train,test}/<label>/<tag>_<stem>[_<annId>].jpg and <out>/summary.json.
With ownPhotos, ml/data/own-test-unlisted/<bin>/ holds photos of items outside the
taxonomy; they go to <out>/test/unlisted__<bin>/ and are never trained on.
--drop-underfilled removes only the *train* data of short classes; their test photos
stay so the report measures the blind spot. It refuses hhw/center classes unless
--allow-drop-hazard is given.
Exit codes: 0 ok, 1 config error or refusal, 2 train/test leakage (identical file bytes).
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
# CATS ids in www/index.html; the folder names allowed under own-test-unlisted/.
BINS = ('recycle', 'trash', 'center', 'hhw', 'brush', 'bulk', 'other')
UNLISTED_PREFIX = 'unlisted__'
UNLISTED_DIR = 'ml/data/own-test-unlisted'
# Classes that must never be silently left out of training.
GUARDED_BINS = ('hhw', 'center')

# ann_id is an optional extra so COCO crops from one image get distinct filenames.
# img_size is the (width, height) the COCO annotation was drawn on, when known.
Sample = collections.namedtuple('Sample', 'label split src bbox tag ann_id img_size')
Sample.__new__.__defaults__ = (None, None)


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
    if 'maxTrain' in config:
        cap = config['maxTrain']
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise ValueError('config "maxTrain" must be a positive integer')
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


def _coco_samples(label, src, root, test_percent, cache, strict=True):
    ann_path = root / src['annotations']
    if not ann_path.is_file():
        if strict:
            raise ValueError(f'{label}: COCO annotations not found: {ann_path}')
        if ann_path not in cache:
            print(f'warning: COCO annotations not found: {ann_path}. Classes that use it '
                  f'get no samples from it (run with --check to treat this as an error).',
                  file=sys.stderr)
            cache[ann_path] = None
        return []
    if ann_path not in cache:
        with open(ann_path, encoding='utf-8') as fh:
            data = json.load(fh)
        cats = {c['id']: c['name'] for c in data.get('categories', [])}
        files = {i['id']: i['file_name'] for i in data.get('images', [])}
        sizes = {i['id']: (i['width'], i['height']) for i in data.get('images', [])
                 if isinstance(i.get('width'), int) and isinstance(i.get('height'), int)}
        cache[ann_path] = (cats, files, sizes, data.get('annotations', []))
    cats, files, sizes, anns = cache[ann_path]
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
        out.append(Sample(label, split, images_dir / name, list(ann['bbox']), 'coco', ann['id'],
                          sizes.get(ann['image_id'])))
    return out


def _unlisted_samples(root):
    """Test-only photos of items outside the taxonomy, labelled unlisted__<bin>."""
    base = root / UNLISTED_DIR
    if not base.is_dir():
        return []
    out = []
    for folder in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith('.')):
        if folder.name not in BINS:
            raise ValueError(f'{UNLISTED_DIR}/{folder.name}: folder must be a bin id, '
                             f'one of {list(BINS)}')
        out += [Sample(UNLISTED_PREFIX + folder.name, 'test', p, None, 'own')
                for p in _images_in(folder)]
    return out


def _cap_key(sample, root):
    key = _rel(sample.src, root)
    if sample.ann_id is not None:
        key += f'#{sample.ann_id}'
    return hashlib.sha1(key.encode('utf-8')).hexdigest(), key


def _cap_train(samples, cap, root):
    """Keep at most `cap` train samples per class: the first by SHA-1 of the source path."""
    keep = set()
    by_label = collections.defaultdict(list)
    for i, s in enumerate(samples):
        if s.split == 'train':
            by_label[s.label].append(i)
    for idx in by_label.values():
        keep.update(sorted(idx, key=lambda i: _cap_key(samples[i], root))[:cap])
    return [s for i, s in enumerate(samples) if s.split != 'train' or i in keep]


def collect(config, root, strict=True):
    """Every planned sample. strict=False turns a missing COCO annotations file into a
    warning (zero samples from it); an unknown category name is always an error."""
    root = Path(root)
    _validate(config)
    tp = config['testPercent']
    cache = {}
    samples = []
    for label, spec in config['classes'].items():
        for src in spec.get('sources', []):
            if src['type'] == 'coco':
                samples += _coco_samples(label, src, root, tp, cache, strict)
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
    if config.get('ownPhotos'):
        samples += _unlisted_samples(root)
    if 'maxTrain' in config:
        samples = _cap_train(samples, config['maxTrain'], root)
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
    """Returns (counts per label and split, stats). Unlisted labels are counted too."""
    counts = {label: {'train': 0, 'test': 0} for label in labels}
    stats = {'unreadable': 0, 'skipped': 0, 'orientation_mismatch': 0}
    for s in samples:
        try:
            if s.bbox is not None:
                full = load_image(s.src, max_side=None)
                if s.img_size is not None and full.size != tuple(s.img_size):
                    # The boxes were drawn on a differently oriented image.
                    stats['orientation_mismatch'] += 1
                    continue
                cropped = crop_box(full, s.bbox)
                if cropped is None:
                    stats['skipped'] += 1
                    continue
                img = shrink(cropped)
            else:
                img = load_image(s.src)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            stats['unreadable'] += 1
            continue
        dest = out / s.split / s.label
        dest.mkdir(parents=True, exist_ok=True)
        img.save(dest / _out_name(s, dest), 'JPEG', quality=90)
        counts.setdefault(s.label, {'train': 0, 'test': 0})[s.split] += 1
    return counts, stats


def _source_dirs(config, root):
    """Every directory the config reads from, including implicit own-photo dirs."""
    dirs = []
    if config.get('ownPhotos'):
        dirs += [root / 'ml/data/own', root / 'ml/data/own-test', root / UNLISTED_DIR]
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


def unwipeable_reason(out):
    """Why <out>/train and <out>/test must not be wiped, or None. They may be wiped
    when <out>/summary.json marks them as a previous run, or when they are absent
    or empty."""
    if (out / 'summary.json').is_file():
        return None
    for split in ('train', 'test'):
        d = out / split
        if d.exists() and (not d.is_dir() or any(d.iterdir())):
            return (f'{d} is not empty and {out / "summary.json"} does not exist, '
                    f'so it is not a previous output of this script')
    return None


def guarded_drops(underfilled, config):
    """Underfilled labels whose bin must not be dropped without consent."""
    classes = config['classes']
    return sorted(l for l in underfilled
                  if l in classes and classes[l].get('bin') in GUARDED_BINS)


def _refuse_guarded(labels):
    print('refusing --drop-underfilled: these hhw/center classes are short and would be '
          f'left out of training: {", ".join(labels)}. Add photos, or pass '
          '--allow-drop-hazard to drop them anyway. Nothing was written.', file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--root', default=str(Path(__file__).resolve().parent.parent),
                    help='base for relative paths in the config (default: repo root)')
    ap.add_argument('--drop-underfilled', action='store_true',
                    help='leave short classes out of training (their test photos stay)')
    ap.add_argument('--allow-drop-hazard', action='store_true',
                    help='with --drop-underfilled, also drop short hhw/center classes')
    ap.add_argument('--check', action='store_true',
                    help='validate the config and sources; write nothing')
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out)

    try:
        with open(args.config, encoding='utf-8') as fh:
            config = json.load(fh)
        samples = collect(config, root, strict=args.check)
    except (OSError, ValueError) as e:
        print(f'config error: {e}', file=sys.stderr)
        return 1

    labels = list(config['classes'])
    planned = {label: {'train': 0, 'test': 0} for label in labels}
    for s in samples:
        planned.setdefault(s.label, {'train': 0, 'test': 0})[s.split] += 1
    open_planned = {l: c['test'] for l, c in planned.items() if l not in config['classes']}
    class_planned = {l: c for l, c in planned.items() if l in config['classes']}

    if args.check:
        print(f'config OK: {len(labels)} classes, {len(samples)} candidate images')
        for label, c in class_planned.items():
            print(f'  {label}: train {c["train"]}, test {c["test"]}')
        for label, n in sorted(open_planned.items()):
            print(f'  {label}: test {n} (open-set, never trained)')
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

    guard = args.drop_underfilled and not args.allow_drop_hazard
    if guard:
        short = summarize(class_planned, config['minTrain'], config['minTest'])['underfilled']
        blocked = guarded_drops(short, config)
        if blocked:
            _refuse_guarded(blocked)
            return 1

    reason = unwipeable_reason(out)
    if reason:
        print(f'refusing to wipe --out: {reason}. Nothing was deleted or written.',
              file=sys.stderr)
        return 1

    # Remove the old summary first: a run that dies part-way leaves none behind.
    (out / 'summary.json').unlink(missing_ok=True)
    for split in ('train', 'test'):
        shutil.rmtree(out / split, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    all_counts, stats = _write_samples(samples, out, labels)
    counts = {l: all_counts[l] for l in labels}
    open_set = {l: c['test'] for l, c in sorted(all_counts.items()) if l not in config['classes']}
    summary = summarize(counts, config['minTrain'], config['minTest'])

    if guard:
        # Unreadable files or tiny crops can make a class short only after writing.
        blocked = guarded_drops(summary['underfilled'], config)
        if blocked:
            for split in ('train', 'test'):
                shutil.rmtree(out / split, ignore_errors=True)
            _refuse_guarded(blocked)
            return 1

    summary['dropped'] = list(summary['underfilled']) if args.drop_underfilled else []
    for label in summary['dropped']:
        shutil.rmtree(out / 'train' / label, ignore_errors=True)
    summary['open_set'] = open_set
    summary['unreadable'] = stats['unreadable']
    summary['orientation_mismatch'] = stats['orientation_mismatch']

    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for label, c in counts.items():
        flag = '  <-- short' if label in summary['underfilled'] else ''
        if label in summary['dropped']:
            flag = '  <-- short, train dropped (test kept)'
        print(f'{label}: train {c["train"]}/{config["minTrain"]}, '
              f'test {c["test"]}/{config["minTest"]}{flag}')
    for label, n in open_set.items():
        print(f'{label}: test {n} (open-set, never trained)')
    print(f'{len(summary["underfilled"])} underfilled, {len(summary["dropped"])} dropped, '
          f'{stats["unreadable"]} unreadable, {stats["skipped"]} crops under {MIN_CROP}px skipped, '
          f'{stats["orientation_mismatch"]} crops skipped for orientation mismatch')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
