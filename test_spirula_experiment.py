"""Boundary tests for the operator-only native process adapter."""
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from experimental.spirula_backend import checked_images, checked_run, run, train_command


class SpirulaExperimentTests(unittest.TestCase):
    def test_no_default_or_customer_opt_in(self):
        with self.assertRaises(PermissionError):
            run('/not/a/binary', '/missing/images', '/missing/output')

    def test_budget_rejection_before_launch(self):
        for iterations in [True, 0, 99, 5001]:
            with self.assertRaises(ValueError):
                train_command('/b', '/d', '/o', iterations, 200000)
        for cap in [True, 999, 200001]:
            with self.assertRaises(ValueError):
                train_command('/b', '/d', '/o', 100, cap)

    def test_only_bounded_local_regular_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(10):
                (root / f'{i}.png').write_bytes(b'fixture')
            self.assertEqual(len(checked_images(root)), 10)
            (root / '0.png').unlink()
            (root / '0.png').symlink_to(root / '1.png')
            with self.assertRaises(ValueError):
                checked_images(root)

    def test_count_and_file_type_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                checked_images(root)
            for i in range(10):
                (root / f'{i}.png').write_bytes(b'fixture')
            (root / 'model.py').write_text('never executed')
            with self.assertRaises(ValueError):
                checked_images(root)

    def test_timeout_terminates_native_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(TimeoutError):
                checked_run([sys.executable, '-c', 'import time;time.sleep(30)'],
                            root / 'log', time.monotonic() + 0.1)

    def test_failed_process_cannot_be_success(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                checked_run([sys.executable, '-c', 'raise SystemExit(9)'],
                            Path(directory) / 'log', time.monotonic() + 3)

    def test_binary_pin_before_input_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'binary'
            binary.write_bytes(b'changed upstream binary')
            output = Path(directory) / 'output'
            with self.assertRaises(ValueError):
                run(binary, '/missing', output, experimental=True)
            self.assertFalse(output.exists())

    def test_completed_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'binary'; binary.write_bytes(b'fixture')
            images = root / 'images'; images.mkdir()
            for i in range(10):
                (images / f'{i}.png').write_bytes(b'fixture')
            output = root / 'completed'; output.mkdir()
            with patch('experimental.spirula_backend.digest', return_value=__import__(
                    'experimental.spirula_backend', fromlist=['BINARY_SHA256']).BINARY_SHA256):
                with self.assertRaisesRegex(ValueError, 'new_output'):
                    run(binary, images, output, experimental=True)


if __name__ == '__main__':
    unittest.main()
