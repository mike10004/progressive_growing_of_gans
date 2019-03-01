import json
from typing import TextIO
import numpy

class NumpyAwareJSONEncoder(json.JSONEncoder):

    def default(self, obj):  # pylint: disable=E0202
        if isinstance(obj, numpy.ndarray):
            if obj.ndim == 1:
                return obj.tolist()
            else:
                return [self.default(obj[i]) for i in range(obj.shape[0])]
        return json.JSONEncoder.default(self, obj)


def serialize_numpy_array(data: numpy.ndarray, ofile: TextIO, **kwargs):
    return json.dump(data, ofile, cls=NumpyAwareJSONEncoder, **kwargs)


def deserialize_numpy_array(ifile: TextIO) -> numpy.ndarray:
    return numpy.array(json.load(ifile))
