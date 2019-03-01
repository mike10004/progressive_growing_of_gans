import numpy
import tempfile
import os
import glob
import os.path
from unittest import TestCase
from typing import Optional
import utils
import logging
import imagery

from utils import serialization


_log = logging.getLogger(__name__)


def _find_network_pkl_path() -> Optional[str]:
    current = os.getcwd()
    while current != '/':
        networks_dir = os.path.join(current, 'networks')
        try:
            return utils.find_network_pickle(networks_dir=networks_dir)
        except IOError:
            pass
        current = os.path.dirname(current)
    _log.info("network pickle file not found; some tests will be skipped")
    return None

_GLOBAL_TF_SESSION = None

class ImageGeneratorTest(TestCase):

    @classmethod
    def setUpClass(cls):
        global _GLOBAL_TF_SESSION
        if _GLOBAL_TF_SESSION is None:
            import tfutil
            _GLOBAL_TF_SESSION = tfutil.create_session({}, True)

    def setUp(self):
        self.network_pkl_path = _find_network_pkl_path()
        self.random_state = numpy.random.RandomState(0xdeadbeef)

    def test_load_network(self):
        if self.network_pkl_path is None:
            self.skipTest("network pickle not present")
        g = imagery.Generator(self.network_pkl_path, self.random_state)
        g._load_network(_GLOBAL_TF_SESSION)

    def test_generate_images(self):
        if self.network_pkl_path is None:
            self.skipTest("network pickle not present")
        g = imagery.Generator(self.network_pkl_path, self.random_state)
        num_pngs = 3
        with tempfile.TemporaryDirectory() as output_dir:
            g.generate_images(_GLOBAL_TF_SESSION, output_dir, num_pngs)
            png_files = glob.glob(os.path.join(output_dir, "*.png"))
            self.assertEqual(num_pngs, len(png_files))

    def test_use_predefined_latents(self):
        if self.network_pkl_path is None:
            self.skipTest("network pickle not present")
        with tempfile.TemporaryDirectory() as output_dir1:
            g = imagery.Generator(self.network_pkl_path, self.random_state)
            g.generate_images(_GLOBAL_TF_SESSION, output_dir1, 1)
            png_files = glob.glob(os.path.join(output_dir1, "*.png"))
            latents_files = glob.glob(os.path.join(output_dir1, "*-latents.json"))
            assert len(png_files) == 1 and len(latents_files) == 1, "multiple png/latents files found"
            generated_image_file, latents_file = png_files[0], latents_files[0]
            with open(latents_file, 'r') as ifile:
                predefined_latents = serialization.deserialize_numpy_array(ifile)
            with open(generated_image_file, 'rb') as ifile:
                generated_image_bytes = ifile.read()
        with tempfile.TemporaryDirectory() as output_dir2:
            g = imagery.Generator(self.network_pkl_path, self.random_state)
            g.get_latents = lambda Gs, index: predefined_latents
            g.generate_images(_GLOBAL_TF_SESSION, output_dir2, 1)
            png_files = glob.glob(os.path.join(output_dir2, "*.png"))
            assert len(png_files) == 1
            with open(png_files[0], 'rb') as ifile:
                duplicate_image_bytes = ifile.read()
        self.assertEqual(generated_image_bytes, duplicate_image_bytes)



