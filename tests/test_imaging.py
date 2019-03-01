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
            _log.info("skipping because network pickle file not found")
            return
        g = imagery.Generator(self.network_pkl_path, self.random_state)
        g._load_network(_GLOBAL_TF_SESSION)

    def test_generate_images(self):
        if self.network_pkl_path is None:
            _log.info("skipping because network pickle file not found")
            return
        g = imagery.Generator(self.network_pkl_path, self.random_state)
        num_pngs = 3
        with tempfile.TemporaryDirectory() as output_dir:
            g.generate_images(_GLOBAL_TF_SESSION, output_dir, num_pngs)
            png_files = glob.glob(os.path.join(output_dir, "*.png"))
            self.assertEqual(num_pngs, len(png_files))


