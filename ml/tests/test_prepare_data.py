import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

import prepare_data  # noqa: E402


def make_jpeg(path, size=(80, 80), color=(200, 30, 30), exif_orientation=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', size, color)
    kwargs = {}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs['exif'] = exif
    img.save(path, 'JPEG', **kwargs)


class TmpRoot(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write_config(self, classes, **overrides):
        cfg = {'model': 'T', 'ownPhotos': True, 'testPercent': 15,
               'minTrain': 40, 'minTest': 10, 'classes': classes}
        cfg.update(overrides)
        p = self.root / 'cfg.json'
        p.write_text(json.dumps(cfg))
        return p


class HashSplitTests(unittest.TestCase):
    def test_hash_split_deterministic_and_proportional(self):
        self.assertEqual(prepare_data.hash_split('a/b.jpg', 15),
                         prepare_data.hash_split('a/b.jpg', 15))
        splits = [prepare_data.hash_split(f'img_{i}.jpg', 15) for i in range(1000)]
        self.assertTrue(set(splits) <= {'train', 'test'})
        frac = splits.count('test') / 1000
        self.assertTrue(0.12 <= frac <= 0.18, frac)


class CropBoxTests(unittest.TestCase):
    def test_crop_box_pads_and_rejects_small(self):
        img = Image.new('RGB', (200, 200), (1, 2, 3))
        self.assertEqual(prepare_data.crop_box(img, [50, 50, 100, 100]).size, (120, 120))
        self.assertIsNone(prepare_data.crop_box(img, [0, 0, 30, 30]))

    def test_crop_box_clamps_to_image(self):
        img = Image.new('RGB', (200, 200), (1, 2, 3))
        self.assertEqual(prepare_data.crop_box(img, [120, 120, 100, 100]).size, (90, 90))
        self.assertEqual(prepare_data.crop_box(img, [100, 100, 100, 100]).size, (110, 110))


class CollectTests(TmpRoot):
    def coco_file(self, cat_name):
        d = self.root / 'taco'
        make_jpeg(d / 'b1' / '1.jpg', size=(300, 300))
        ann = {'images': [{'id': 1, 'file_name': 'b1/1.jpg'}],
               'categories': [{'id': 7, 'name': cat_name}],
               'annotations': [{'id': 11, 'image_id': 1, 'category_id': 7,
                                'bbox': [10, 10, 200, 200]}]}
        (d / 'annotations.json').write_text(json.dumps(ann))

    def coco_cfg(self, wanted):
        return {'model': 'T', 'ownPhotos': False, 'testPercent': 15,
                'minTrain': 1, 'minTest': 1,
                'classes': {'soda': {'bin': 'recycle', 'sources': [
                    {'type': 'coco', 'annotations': 'taco/annotations.json',
                     'images': 'taco', 'categories': [wanted], 'split': 'train'}]}}}

    def test_unknown_coco_category_raises(self):
        self.coco_file('Drink can')
        with self.assertRaises(ValueError) as cm:
            prepare_data.collect(self.coco_cfg('Soda can'), self.root)
        self.assertIn('Drink can', str(cm.exception))

    def test_coco_collects_one_sample_per_annotation(self):
        self.coco_file('Drink can')
        samples = prepare_data.collect(self.coco_cfg('Drink can'), self.root)
        self.assertEqual(len(samples), 1)
        s = samples[0]
        self.assertEqual((s.label, s.split, s.bbox, s.tag, s.ann_id),
                         ('soda', 'train', [10, 10, 200, 200], 'coco', 11))

    def test_missing_coco_file_raises(self):
        with self.assertRaises(ValueError):
            prepare_data.collect(self.coco_cfg('Drink can'), self.root)

    def test_missing_folder_contributes_nothing(self):
        cfg = {'model': 'T', 'ownPhotos': False, 'testPercent': 15, 'minTrain': 1,
               'minTest': 1, 'classes': {'x': {'bin': 'none', 'sources': [
                   {'type': 'folder', 'path': 'nope', 'split': 'train'}]}}}
        self.assertEqual(prepare_data.collect(cfg, self.root), [])

    def test_folder_hash_split_assigns_both_sides(self):
        for i in range(60):
            make_jpeg(self.root / 'f' / f'img_{i}.jpg')
        cfg = {'model': 'T', 'ownPhotos': False, 'testPercent': 15, 'minTrain': 1,
               'minTest': 1, 'classes': {'x': {'bin': 'none', 'sources': [
                   {'type': 'folder', 'path': 'f', 'split': 'hash'}]}}}
        splits = {s.split for s in prepare_data.collect(cfg, self.root)}
        self.assertEqual(splits, {'train', 'test'})

    def test_own_photos_implicit_and_optional(self):
        make_jpeg(self.root / 'ml/data/own/x/a.jpg')
        make_jpeg(self.root / 'ml/data/own-test/x/b.jpg')
        cfg = {'model': 'T', 'ownPhotos': True, 'testPercent': 15, 'minTrain': 1,
               'minTest': 1, 'classes': {'x': {'bin': 'none', 'sources': []},
                                         'y': {'bin': 'none', 'sources': []}}}
        samples = prepare_data.collect(cfg, self.root)
        got = {(s.label, s.split, s.src.name) for s in samples}
        self.assertEqual(got, {('x', 'train', 'a.jpg'), ('x', 'test', 'b.jpg')})


class LoadImageTests(TmpRoot):
    def test_exif_orientation_applied(self):
        p = self.root / 'r.jpg'
        make_jpeg(p, size=(100, 50), exif_orientation=6)
        self.assertEqual(prepare_data.load_image(p).size, (50, 100))

    def test_downsizes_but_never_upscales(self):
        big, small = self.root / 'big.jpg', self.root / 'small.jpg'
        make_jpeg(big, size=(800, 400))
        make_jpeg(small, size=(100, 50))
        self.assertEqual(prepare_data.load_image(big).size, (384, 192))
        self.assertEqual(prepare_data.load_image(small).size, (100, 50))

    @unittest.skipUnless(shutil.which('sips'), 'sips not available')
    def test_heic_goes_through_sips(self):
        jpg, heic = self.root / 'a.jpg', self.root / 'a.heic'
        make_jpeg(jpg, size=(120, 60))
        r = subprocess.run(['sips', '-s', 'format', 'heic', str(jpg), '--out', str(heic)],
                           capture_output=True)
        if r.returncode != 0 or not heic.exists():
            self.skipTest('sips cannot write heic here')
        self.assertEqual(prepare_data.load_image(heic).size, (120, 60))


class LeakAndSummaryTests(TmpRoot):
    def test_find_leaks_by_content(self):
        a, b, c = (self.root / n for n in ('a.jpg', 'b.jpg', 'c.jpg'))
        make_jpeg(a)
        shutil.copy(a, b)
        make_jpeg(c, color=(0, 0, 255))
        S = prepare_data.Sample
        samples = [S('x', 'train', a, None, 't'), S('x', 'test', b, None, 't'),
                   S('x', 'test', c, None, 't')]
        self.assertEqual(prepare_data.find_leaks(samples), [(a, b)])

    def test_summarize(self):
        counts = {'a': {'train': 40, 'test': 10}, 'b': {'train': 3, 'test': 10},
                  'c': {'train': 40, 'test': 9}}
        s = prepare_data.summarize(counts, 40, 10)
        self.assertEqual(s['underfilled'], ['b', 'c'])
        self.assertEqual(s['counts'], counts)


class MainTests(TmpRoot):
    def run_main(self, cfg_path, *extra):
        return prepare_data.main(['--config', str(cfg_path), '--out', str(self.root / 'out'),
                                  '--root', str(self.root), *extra])

    def test_leak_detected_exits_2(self):
        make_jpeg(self.root / 'ml/data/own/x/a.jpg')
        (self.root / 'ml/data/own-test/x').mkdir(parents=True)
        shutil.copy(self.root / 'ml/data/own/x/a.jpg', self.root / 'ml/data/own-test/x/a_copy.jpg')
        cfg = self.write_config({'x': {'bin': 'none', 'sources': []}})
        self.assertEqual(self.run_main(cfg), 2)
        self.assertFalse((self.root / 'out' / 'summary.json').exists())

    def test_ok_run_writes_layout_and_summary(self):
        make_jpeg(self.root / 'ml/data/own/x/a.jpg', color=(1, 1, 1))
        make_jpeg(self.root / 'ml/data/own-test/x/b.jpg', color=(9, 9, 9))
        (self.root / 'ml/data/own/x/bad.jpg').write_bytes(b'not an image')
        cfg = self.write_config({'x': {'bin': 'none', 'sources': []}})
        self.assertEqual(self.run_main(cfg), 0)
        out = self.root / 'out'
        self.assertEqual([p.name for p in (out / 'train' / 'x').iterdir()], ['own_a.jpg'])
        self.assertEqual([p.name for p in (out / 'test' / 'x').iterdir()], ['own_b.jpg'])
        summary = json.loads((out / 'summary.json').read_text())
        self.assertEqual(summary['counts'], {'x': {'train': 1, 'test': 1}})
        self.assertEqual(summary['underfilled'], ['x'])
        self.assertEqual(summary['unreadable'], 1)

    def test_drop_underfilled(self):
        for i in range(3):
            make_jpeg(self.root / f'ml/data/own/x/{i}.jpg', color=(i * 40, 0, 0))
        cfg = self.write_config({'x': {'bin': 'none', 'sources': []}})
        self.assertEqual(self.run_main(cfg), 0)
        out = self.root / 'out'
        self.assertEqual(json.loads((out / 'summary.json').read_text())['underfilled'], ['x'])
        self.assertTrue((out / 'train' / 'x').is_dir())
        self.assertEqual(self.run_main(cfg, '--drop-underfilled'), 0)
        self.assertEqual(json.loads((out / 'summary.json').read_text())['underfilled'], ['x'])
        self.assertFalse((out / 'train' / 'x').exists())
        self.assertFalse((out / 'test' / 'x').exists())

    def test_check_writes_nothing_and_flags_missing_coco(self):
        cfg = self.write_config({'x': {'bin': 'none', 'sources': []}})
        self.assertEqual(self.run_main(cfg, '--check'), 0)
        self.assertFalse((self.root / 'out').exists())
        bad = self.write_config({'x': {'bin': 'none', 'sources': [
            {'type': 'coco', 'annotations': 'missing.json', 'images': 'm',
             'categories': ['A'], 'split': 'train'}]}})
        self.assertEqual(self.run_main(bad, '--check'), 1)
        self.assertEqual(self.run_main(bad), 1)


if __name__ == '__main__':
    unittest.main()
