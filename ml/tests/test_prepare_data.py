import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

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

    def test_missing_coco_file_warns_when_not_strict(self):
        err = io.StringIO()
        with redirect_stderr(err):
            samples = prepare_data.collect(self.coco_cfg('Drink can'), self.root, strict=False)
        self.assertEqual(samples, [])
        self.assertIn('COCO annotations not found', err.getvalue())

    def test_unknown_coco_category_raises_even_when_not_strict(self):
        self.coco_file('Drink can')
        with self.assertRaises(ValueError):
            prepare_data.collect(self.coco_cfg('Soda can'), self.root, strict=False)

    def test_coco_sample_carries_annotated_size(self):
        d = self.root / 'taco'
        make_jpeg(d / 'b1' / '1.jpg', size=(300, 200))
        ann = {'images': [{'id': 1, 'file_name': 'b1/1.jpg', 'width': 300, 'height': 200}],
               'categories': [{'id': 7, 'name': 'Drink can'}],
               'annotations': [{'id': 11, 'image_id': 1, 'category_id': 7,
                                'bbox': [10, 10, 100, 100]}]}
        (d / 'annotations.json').write_text(json.dumps(ann))
        s = prepare_data.collect(self.coco_cfg('Drink can'), self.root)[0]
        self.assertEqual(s.img_size, (300, 200))

    def test_unlisted_lookalikes_are_test_only(self):
        make_jpeg(self.root / 'ml/data/own-test-unlisted/hhw/oil.jpg')
        make_jpeg(self.root / 'ml/data/own-test-unlisted/recycle/toy.jpg')
        cfg = {'model': 'T', 'ownPhotos': True, 'testPercent': 15, 'minTrain': 1,
               'minTest': 1, 'classes': {'x': {'bin': 'none', 'sources': []}}}
        got = {(s.label, s.split, s.src.name) for s in prepare_data.collect(cfg, self.root)}
        self.assertEqual(got, {('unlisted__hhw', 'test', 'oil.jpg'),
                               ('unlisted__recycle', 'test', 'toy.jpg')})

    def test_unlisted_ignored_without_own_photos(self):
        make_jpeg(self.root / 'ml/data/own-test-unlisted/hhw/oil.jpg')
        cfg = {'model': 'T', 'ownPhotos': False, 'testPercent': 15, 'minTrain': 1,
               'minTest': 1, 'classes': {'x': {'bin': 'none', 'sources': []}}}
        self.assertEqual(prepare_data.collect(cfg, self.root), [])

    def test_unlisted_unknown_bin_folder_raises(self):
        make_jpeg(self.root / 'ml/data/own-test-unlisted/hazardous/oil.jpg')
        cfg = {'model': 'T', 'ownPhotos': True, 'testPercent': 15, 'minTrain': 1,
               'minTest': 1, 'classes': {'x': {'bin': 'none', 'sources': []}}}
        with self.assertRaises(ValueError) as cm:
            prepare_data.collect(cfg, self.root)
        self.assertIn('hazardous', str(cm.exception))

    def max_train_cfg(self, **extra):
        cfg = {'model': 'T', 'ownPhotos': False, 'testPercent': 15, 'minTrain': 1,
               'minTest': 1, 'classes': {'x': {'bin': 'none', 'sources': [
                   {'type': 'folder', 'path': 'f', 'split': 'train'},
                   {'type': 'folder', 'path': 'g', 'split': 'test'}]}}}
        cfg.update(extra)
        return cfg

    def test_max_train_caps_deterministically_by_sha1(self):
        for i in range(10):
            make_jpeg(self.root / 'f' / f'img_{i}.jpg')
        for i in range(6):
            make_jpeg(self.root / 'g' / f't_{i}.jpg')
        cfg = self.max_train_cfg(maxTrain=4)
        first = prepare_data.collect(cfg, self.root)
        train = sorted(s.src.name for s in first if s.split == 'train')
        self.assertEqual(len(train), 4)
        expected = sorted(sorted((f'img_{i}.jpg' for i in range(10)),
                                 key=lambda n: prepare_data.hashlib.sha1(
                                     f'f/{n}'.encode()).hexdigest())[:4])
        self.assertEqual(train, expected)
        self.assertEqual(sum(1 for s in first if s.split == 'test'), 6)
        again = prepare_data.collect(cfg, self.root)
        self.assertEqual(sorted(s.src.name for s in again if s.split == 'train'), train)

    def test_max_train_absent_keeps_everything(self):
        for i in range(10):
            make_jpeg(self.root / 'f' / f'img_{i}.jpg')
        self.assertEqual(len(prepare_data.collect(self.max_train_cfg(), self.root)), 10)

    def test_max_train_must_be_positive_int(self):
        with self.assertRaises(ValueError):
            prepare_data.collect(self.max_train_cfg(maxTrain='500'), self.root)
        with self.assertRaises(ValueError):
            prepare_data.collect(self.max_train_cfg(maxTrain=0), self.root)

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
    def run_main(self, cfg_path, *extra, out=None):
        return prepare_data.main(['--config', str(cfg_path),
                                  '--out', str(out if out is not None else self.root / 'out'),
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

    def test_drop_underfilled_drops_train_only(self):
        for i in range(3):
            make_jpeg(self.root / f'ml/data/own/x/{i}.jpg', color=(i * 40, 0, 0))
        make_jpeg(self.root / 'ml/data/own-test/x/t.jpg', color=(0, 0, 200))
        cfg = self.write_config({'x': {'bin': 'none', 'sources': []}})
        self.assertEqual(self.run_main(cfg), 0)
        out = self.root / 'out'
        summary = json.loads((out / 'summary.json').read_text())
        self.assertEqual(summary['underfilled'], ['x'])
        self.assertEqual(summary['dropped'], [])
        self.assertTrue((out / 'train' / 'x').is_dir())
        self.assertEqual(self.run_main(cfg, '--drop-underfilled'), 0)
        summary = json.loads((out / 'summary.json').read_text())
        self.assertEqual(summary['underfilled'], ['x'])
        self.assertEqual(summary['dropped'], ['x'])
        self.assertFalse((out / 'train' / 'x').exists())
        self.assertEqual([p.name for p in (out / 'test' / 'x').iterdir()], ['own_t.jpg'])

    def test_drop_underfilled_refuses_hazard_and_center_classes(self):
        make_jpeg(self.root / 'ml/data/own/paint/a.jpg', color=(1, 1, 1))
        make_jpeg(self.root / 'ml/data/own/foam/a.jpg', color=(2, 2, 2))
        make_jpeg(self.root / 'ml/data/own/x/a.jpg', color=(3, 3, 3))
        cfg = self.write_config({'paint': {'bin': 'hhw', 'sources': []},
                                 'foam': {'bin': 'center', 'sources': []},
                                 'x': {'bin': 'recycle', 'sources': []}})
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(self.run_main(cfg, '--drop-underfilled'), 1)
        self.assertIn('foam', err.getvalue())
        self.assertIn('paint', err.getvalue())
        self.assertFalse((self.root / 'out').exists())
        self.assertEqual(self.run_main(cfg, '--drop-underfilled', '--allow-drop-hazard'), 0)
        summary = json.loads((self.root / 'out' / 'summary.json').read_text())
        self.assertEqual(summary['dropped'], ['foam', 'paint', 'x'])

    def test_drop_refusal_also_covers_classes_short_after_unreadables(self):
        # Planned counts look full (2 train, 1 test with minTrain 2), but one
        # train file is unreadable, so the hhw class is only short after writing.
        make_jpeg(self.root / 'ml/data/own/paint/a.jpg', color=(1, 1, 1))
        (self.root / 'ml/data/own/paint/bad.jpg').write_bytes(b'not an image')
        make_jpeg(self.root / 'ml/data/own-test/paint/t.jpg', color=(5, 5, 5))
        cfg = self.write_config({'paint': {'bin': 'hhw', 'sources': []}},
                                minTrain=2, minTest=1)
        self.assertEqual(self.run_main(cfg, '--drop-underfilled'), 1)
        out = self.root / 'out'
        self.assertFalse((out / 'summary.json').exists())
        self.assertFalse((out / 'train').exists())
        self.assertFalse((out / 'test').exists())

    def test_unlisted_written_to_test_and_summarised(self):
        make_jpeg(self.root / 'ml/data/own/x/a.jpg', color=(1, 1, 1))
        make_jpeg(self.root / 'ml/data/own-test-unlisted/hhw/oil.jpg', color=(9, 9, 9))
        cfg = self.write_config({'x': {'bin': 'recycle', 'sources': []}})
        self.assertEqual(self.run_main(cfg), 0)
        out = self.root / 'out'
        self.assertEqual([p.name for p in (out / 'test' / 'unlisted__hhw').iterdir()],
                         ['own_oil.jpg'])
        self.assertFalse((out / 'train' / 'unlisted__hhw').exists())
        summary = json.loads((out / 'summary.json').read_text())
        self.assertEqual(summary['open_set'], {'unlisted__hhw': 1})
        self.assertNotIn('unlisted__hhw', summary['counts'])
        self.assertEqual(summary['underfilled'], ['x'])

    def test_unlisted_leak_detected(self):
        make_jpeg(self.root / 'ml/data/own/x/a.jpg')
        (self.root / 'ml/data/own-test-unlisted/hhw').mkdir(parents=True)
        shutil.copy(self.root / 'ml/data/own/x/a.jpg',
                    self.root / 'ml/data/own-test-unlisted/hhw/a.jpg')
        cfg = self.write_config({'x': {'bin': 'recycle', 'sources': []}})
        self.assertEqual(self.run_main(cfg), 2)

    def test_refuses_to_wipe_unrelated_dirs(self):
        make_jpeg(self.root / 'ml/data/own/x/a.jpg')
        notes = self.root / 'out' / 'train' / 'notes.txt'
        notes.parent.mkdir(parents=True)
        notes.write_text('mine')
        cfg = self.write_config({'x': {'bin': 'none', 'sources': []}})
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(self.run_main(cfg), 1)
        self.assertTrue(notes.exists())
        self.assertIn('summary.json', err.getvalue())
        self.assertFalse((self.root / 'out' / 'summary.json').exists())

    def test_rewipes_previous_output(self):
        make_jpeg(self.root / 'ml/data/own/x/a.jpg')
        cfg = self.write_config({'x': {'bin': 'none', 'sources': []}})
        self.assertEqual(self.run_main(cfg), 0)
        stale = self.root / 'out' / 'train' / 'gone' / 'old.jpg'
        make_jpeg(stale)
        self.assertEqual(self.run_main(cfg), 0)
        self.assertFalse(stale.exists())

    def test_empty_existing_dirs_are_fine(self):
        make_jpeg(self.root / 'ml/data/own/x/a.jpg')
        (self.root / 'out' / 'train').mkdir(parents=True)
        (self.root / 'out' / 'test').mkdir(parents=True)
        cfg = self.write_config({'x': {'bin': 'none', 'sources': []}})
        self.assertEqual(self.run_main(cfg), 0)

    def test_crashed_run_leaves_no_stale_summary(self):
        make_jpeg(self.root / 'ml/data/own/x/a.jpg')
        cfg = self.write_config({'x': {'bin': 'none', 'sources': []}})
        self.assertEqual(self.run_main(cfg), 0)
        self.assertTrue((self.root / 'out' / 'summary.json').exists())
        with mock.patch.object(prepare_data, '_write_samples', side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                self.run_main(cfg)
        self.assertFalse((self.root / 'out' / 'summary.json').exists())

    def test_decompression_bomb_counts_as_unreadable(self):
        make_jpeg(self.root / 'ml/data/own/x/a.jpg', size=(80, 80))
        cfg = self.write_config({'x': {'bin': 'none', 'sources': []}})
        with mock.patch.object(prepare_data.Image, 'MAX_IMAGE_PIXELS', 100):
            self.assertEqual(self.run_main(cfg), 0)
        summary = json.loads((self.root / 'out' / 'summary.json').read_text())
        self.assertEqual(summary['unreadable'], 1)

    def test_coco_orientation_mismatch_skipped_and_counted(self):
        d = self.root / 'taco'
        # Stored 300x200 with EXIF "rotate 90": it loads as 200x300, but the
        # annotation says 300x200, so the boxes are in the wrong frame.
        make_jpeg(d / 'b1' / 'rot.jpg', size=(300, 200), exif_orientation=6)
        make_jpeg(d / 'b1' / 'ok.jpg', size=(300, 200), color=(0, 90, 0))
        ann = {'images': [{'id': 1, 'file_name': 'b1/rot.jpg', 'width': 300, 'height': 200},
                          {'id': 2, 'file_name': 'b1/ok.jpg', 'width': 300, 'height': 200}],
               'categories': [{'id': 7, 'name': 'Drink can'}],
               'annotations': [{'id': 11, 'image_id': 1, 'category_id': 7, 'bbox': [10, 10, 100, 100]},
                               {'id': 12, 'image_id': 1, 'category_id': 7, 'bbox': [120, 10, 100, 100]},
                               {'id': 13, 'image_id': 2, 'category_id': 7, 'bbox': [10, 10, 100, 100]}]}
        (d / 'annotations.json').write_text(json.dumps(ann))
        cfg = self.write_config({'soda': {'bin': 'recycle', 'sources': [
            {'type': 'coco', 'annotations': 'taco/annotations.json', 'images': 'taco',
             'categories': ['Drink can'], 'split': 'train'}]}}, ownPhotos=False)
        self.assertEqual(self.run_main(cfg), 0)
        out = self.root / 'out'
        self.assertEqual([p.name for p in (out / 'train' / 'soda').iterdir()], ['coco_ok_13.jpg'])
        summary = json.loads((out / 'summary.json').read_text())
        self.assertEqual(summary['orientation_mismatch'], 2)

    def test_check_writes_nothing_and_flags_missing_coco(self):
        cfg = self.write_config({'x': {'bin': 'none', 'sources': []}})
        self.assertEqual(self.run_main(cfg, '--check'), 0)
        self.assertFalse((self.root / 'out').exists())
        bad = self.write_config({'x': {'bin': 'none', 'sources': [
            {'type': 'coco', 'annotations': 'missing.json', 'images': 'm',
             'categories': ['A'], 'split': 'train'}]}})
        self.assertEqual(self.run_main(bad, '--check'), 1)
        self.assertFalse((self.root / 'out').exists())

    def test_real_run_warns_on_missing_coco_and_continues(self):
        make_jpeg(self.root / 'ml/data/own/x/a.jpg')
        cfg = self.write_config({'x': {'bin': 'none', 'sources': [
            {'type': 'coco', 'annotations': 'missing.json', 'images': 'm',
             'categories': ['A'], 'split': 'train'}]}})
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(self.run_main(cfg), 0)
        self.assertIn('warning', err.getvalue())
        self.assertIn('missing.json', err.getvalue())
        summary = json.loads((self.root / 'out' / 'summary.json').read_text())
        self.assertEqual(summary['counts'], {'x': {'train': 1, 'test': 0}})


class UnsafeOutTests(TmpRoot):
    run_main = MainTests.run_main

    def cfg(self):
        make_jpeg(self.root / 'ml/data/own/x/a.jpg')
        return self.write_config({'x': {'bin': 'none', 'sources': []}})

    def test_out_equal_to_root_refused_and_nothing_deleted(self):
        cfg = self.cfg()
        keep = self.root / 'train' / 'keep.jpg'
        make_jpeg(keep)
        self.assertEqual(self.run_main(cfg, out=self.root), 1)
        self.assertTrue(keep.exists())
        self.assertFalse((self.root / 'summary.json').exists())

    def test_out_inside_source_folder_refused(self):
        cfg = self.cfg()
        self.assertEqual(self.run_main(cfg, out=self.root / 'ml/data/own/x/build'), 1)
        self.assertEqual(self.run_main(cfg, out=self.root / 'ml/data/own'), 1)
        self.assertFalse((self.root / 'ml/data/own/x/build').exists())

    def test_out_containing_source_folder_refused_and_keeps_existing(self):
        keep = self.root / 'out' / 'train' / 'keep.jpg'
        make_jpeg(keep)
        make_jpeg(self.root / 'out' / 'src' / 'p.jpg')
        cfg = self.write_config({'x': {'bin': 'none', 'sources': [
            {'type': 'folder', 'path': 'out/src', 'split': 'train'}]}})
        self.assertEqual(self.run_main(cfg), 1)
        self.assertTrue(keep.exists())

    def test_out_empty_or_cwd_refused(self):
        cfg = self.cfg()
        self.assertEqual(self.run_main(cfg, out=''), 1)
        self.assertEqual(self.run_main(cfg, out='.'), 1)

    def test_out_ancestor_of_root_refused(self):
        cfg = self.cfg()
        self.assertEqual(self.run_main(cfg, out=self.root.parent), 1)


if __name__ == '__main__':
    unittest.main()
