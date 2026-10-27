import importlib.util
from pathlib import Path
import tempfile
import unittest
from gsplat_backend import contained_image
from reconstruction_worker import handler
from package_gsplat import package


class ReconstructionContracts(unittest.TestCase):
    def test_rejects_paths_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'photo.png').write_bytes(b'synthetic')
            self.assertEqual(contained_image(root,'photo.png'),root/'photo.png')
            (root/'link.png').symlink_to(root/'photo.png')
            for name in ['../photo.png','missing.png','link.png']:
                with self.assertRaises(ValueError):contained_image(root,name)

    def test_no_public_or_arbitrary_operator_input(self):
        for value in [None,{},[],{'input':{}},{'input':{'dataset_url':'https://invalid.test'}},
                      {'input':{'action':'synthetic_smoke','iterations':999999}}]:
            self.assertEqual(handler(value)['failure_code'],'private_reconstruction_not_yet_enabled')

    def test_reproducible_source_bound_package(self):
        a,b=package(),package()
        self.assertEqual(a,b)
        self.assertRegex(a['env']['ARQARY_RECON_SHA256'],r'^[a-f0-9]{64}$')
        self.assertIn('@sha256:',a['image'])
        self.assertNotIn('RUNPOD_API_KEY',a['env'])

    def test_default_runtime_contains_no_graphdeco_install(self):
        source=Path(__file__).with_name('bootstrap_gsplat.py').read_text()
        self.assertIn('gsplat==1.5.3',source)
        self.assertNotIn('graphdeco-inria',source)
        self.assertNotIn('diff-gaussian-rasterization',source)


if __name__=='__main__':unittest.main()
