import os
import misc
import config
import logging
import os.path
import pickle

import numpy as np
import utils.serialization as serialization

from typing import Iterable

_log = logging.getLogger(__name__)

_SINGLE_IMAGE_GRID = (1, 1)
_AUX_CONFIG = 'config'
_AUX_LATENTS = 'latents'


class ImageGenerator(object):

    def __init__(self, network_pkl_pathname: str, random_state: np.random.RandomState,
                 image_shrink:int=1, aux_discards:Iterable[str]=tuple(), minibatch_size:int=8):
        self.network_pkl_pathname = network_pkl_pathname
        self.random_state = random_state
        self.image_shrink = image_shrink
        self.minibatch_size = minibatch_size
        self.grid_size = _SINGLE_IMAGE_GRID
        self.discards = frozenset(aux_discards)
        self.tf_initialized = False
        self.tf_config = {}

    def _write_aux(self, kind, output_dir, png_prefix, png_idx, data: np.ndarray):
        if not kind in self.discards:
            output_pathname = os.path.join(output_dir, f"{png_prefix}-{png_idx}-{kind}.json")
            with open(output_pathname, 'w') as ofile:
                serialization.serialize_numpy_array(data, ofile)

    def _dump_config(self, output_dir):
        if _AUX_CONFIG not in self.discards:
            try:
                config_export_pathname = os.path.join(output_dir, "config.txt")
                os.makedirs(os.path.dirname(config_export_pathname), exist_ok=True)
                with open(config_export_pathname, 'w') as fout:
                    for k, v in sorted(config.__dict__.items()):
                        if not k.startswith('_'):
                            fout.write("%s = %s\n" % (k, str(v)))
            except (IOError, ValueError, TypeError) as e:
                _log.info("error dumping config file: %s", e)

    # noinspection PyUnusedLocal,PyUnresolvedReferences
    def _load_network(self, tf_session: 'tensorflow.Session'):
        # TODO fix unpickling so that the argument session is used instead of the default
        _log.debug("loading network from %s", self.network_pkl_pathname)
        with open(self.network_pkl_pathname, 'rb') as ifile:
            return pickle.load(ifile)

    # noinspection PyUnresolvedReferences
    def generate_images(self, tf_session: 'tensorflow.Session', output_dir: str, num_pngs: int=1, png_prefix: str="progan"):
        self._dump_config(output_dir)
        _G, _D, Gs = self._load_network(tf_session)
        for png_idx in range(num_pngs):
            _log.debug("%d / %d generating...", png_idx, num_pngs)
            latents = misc.random_latents(np.prod(self.grid_size), Gs, random_state=self.random_state)
            labels = np.zeros([latents.shape[0], 0], np.float32)
            images = Gs.run(latents, labels, minibatch_size=self.minibatch_size, num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5, out_shrink=self.image_shrink, out_dtype=np.uint8)
            image = images[0]
            output_pathname = os.path.join(output_dir, f"{png_prefix}-{png_idx}.png")
            os.makedirs(os.path.dirname(output_pathname), exist_ok=True)
            misc.save_image(image, output_pathname, drange=[0,255])
            _log.debug("%d / %d generated: %s", png_idx, num_pngs, os.path.basename(output_pathname))
            self._write_aux(_AUX_LATENTS, output_dir, png_prefix, png_idx, latents)


# alias for more natural usage as imaging.Generator
Generator = ImageGenerator
