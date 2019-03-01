#!/usr/bin/env python3

import os
import sys
import imagery
import logging
import os.path
import utils
import argparse

import numpy as np

from utils import serialization


_log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", metavar="DIR", nargs='?', help="directory for output images")
    parser.add_argument("--network", metavar="FILE", help="set network model path; by default, the 'networks' subdir is searched")
    parser.add_argument("--networks-dir", metavar="DIR", help="set network search directory")
    parser.add_argument("--seed", type=int, metavar="N", help="set random number generator seed")
    parser.add_argument("--image-shrink", choices=(0, 1), type=int, default=1, metavar="N", help="set image_shrink parameter")
    parser.add_argument("--log-level", "-l", choices=('INFO', 'DEBUG', 'WARNING', 'ERROR'), metavar="LEVEL", default='INFO', help="set log level")
    parser.add_argument("--num-images", "-n", type=int, metavar="N", default=1, help="set number of images to generate")
    parser.add_argument("--discard", metavar="CATEGORY", help="discard auxiliary data ('config' or 'latents')")
    parser.add_argument("--filename-prefix", metavar="STRING", default="progan", help="set prefix of filenames")
    parser.add_argument("--tensorflow-seed", type=int, metavar="N", help="set tensorflow randomization seed")
    parser.add_argument("--latents", metavar="FILE", help="read latents from FILE")
    args = parser.parse_args()
    logging.basicConfig(level=logging.__dict__[args.log_level])
    random_state = np.random.RandomState(args.seed)
    network_pkl_pathname = utils.find_network_pickle(args.network, args.networks_dir)
    output_dir = args.output_dir or os.path.join(os.getcwd(), 'outputs', utils.timestamp())
    discards = tuple() if args.discard is None else args.discard.split(',')
    generator = imagery.Generator(network_pkl_pathname, random_state, args.image_shrink, discards)
    if args.latents is not None:
        with open(args.latents, 'r') as ifile:
            latents_source = serialization.deserialize_numpy_array(ifile)
        def get_only_latents(*args):
            return latents_source
        def get_latents_by_index(_, index):
            return latents_source[index]
        if args.num_images == 1:
            generator.get_latents = get_only_latents
        else:
            generator.get_latents = get_latents_by_index
    config_dict = {}
    import tensorflow # defer these imports in case user just wanted --help
    import tfutil
    tensorflow.set_random_seed(args.tensorflow_seed if args.tensorflow_seed is not None else np.random.randint(1 << 31))
    assert tensorflow.get_default_session() is None, "only one tensorflow session allowed due to bad pickling"
    with tfutil.create_session(config_dict, False) as tf_session:
        generator.generate_images(tf_session, output_dir, args.num_images, args.filename_prefix)
    if args.output_dir is None:
        print(f"{output_dir} contains generated images", file=sys.stderr)
    return 0


if __name__ == '__main__':
    exit(main())