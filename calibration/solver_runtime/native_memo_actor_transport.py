"""Continue the native pickle memo across one common/private actor stream.

Each child still loads its own common graph. Keeping one Unpickler for both
authenticated files preserves native MEMOIZE numbering and all graph aliases;
assigning a copied memo to a second Unpickler does not preserve that numbering.
"""
from __future__ import annotations

import io
import pickle


FORMAT = 'rinkspace.native-common-private-pickle-memo.v1'


def dump_private_with_common_memo(output, value, common_memo):
    pickler = pickle.Pickler(output, protocol=5)
    pickler.memo = common_memo
    pickler.dump(value)
    return len(common_memo)


class _FileSequence(io.RawIOBase):
    def __init__(self, *streams):
        super().__init__()
        self.streams = streams
        self.position = 0

    def readable(self):
        return True

    def readinto(self, target):
        view = memoryview(target)
        done = 0
        try:
            while done < len(view) and self.position < len(self.streams):
                count = self.streams[self.position].readinto(view[done:])
                if count is None:
                    raise OSError('Owned actor payload cannot use nonblocking input')
                if count == 0:
                    self.position += 1
                else:
                    done += count
            return done
        finally:
            view.release()


def load_common_then_private(common_input, private_input):
    with io.BufferedReader(_FileSequence(common_input, private_input),
                           buffer_size=1024 * 1024) as stream:
        loader = pickle.Unpickler(stream)
        geometry = loader.load()
        common_entries = len(loader.memo.copy())
        block = loader.load()
        if stream.read(1):
            raise ValueError('Unexpected trailing owned actor payload')
        return geometry, block, common_entries
