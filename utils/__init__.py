import os.path
import glob
import datetime


def timestamp(now=None) -> str:
    """Produces a filename-safe timestamp string."""
    now = now if now is not None else datetime.datetime.now()
    now_iso = now.isoformat(timespec='seconds')
    now_iso = now_iso.replace(':', '')
    now_iso = now_iso.replace('-', '')
    return now_iso


def find_network_pickle(network: str=None, networks_dir: str=None):
    if network is not None:
        return network
    networks_dir = networks_dir or os.path.join(os.getcwd(), 'networks')
    pkl_files = glob.glob(os.path.join(networks_dir, '*.pkl'))
    if not pkl_files:
        raise IOError("no network pickle files found in " + networks_dir)
    return pkl_files[-1]  # return lexicographically last to get file w/latest timestamp


