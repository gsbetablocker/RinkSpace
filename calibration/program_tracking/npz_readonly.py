"""Read existing uncompressed checkpoint arrays without allocating a copy.

Only ZIP_STORED NPY members are mapped. The archive stays immutable; this is
for authenticated frozen residual/Jacobian replay, never writable state.
"""
from pathlib import Path
import struct
import zipfile

import numpy as np


def mapped_array(archive, name):
    archive = Path(archive)
    with zipfile.ZipFile(archive) as container:
        member = container.getinfo(name + '.npy')
        if member.compress_type != zipfile.ZIP_STORED or member.flag_bits & 1:
            raise ValueError('Read-only replay requires uncompressed, unencrypted NPY members')
        with archive.open('rb') as handle:
            handle.seek(member.header_offset)
            header = handle.read(30)
            signature, *fields = struct.unpack('<4s5H3I2H', header)
            if signature != b'PK\x03\x04':
                raise ValueError('Invalid NPZ member header')
            handle.seek(fields[-2] + fields[-1], 1)
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError('Unsupported NPY version for frozen replay')
            if dtype.hasobject:
                raise ValueError('Object arrays cannot be used in frozen numerical replay')
            offset = handle.tell()
    # np.memmap cannot open an empty array; such arrays own no large storage.
    if not np.prod(shape, dtype=np.int64):
        value = np.empty(shape, dtype=dtype)
        value.flags.writeable = False
        return value
    return np.memmap(archive, dtype=dtype, mode='r', offset=offset,
                     shape=shape, order='F' if fortran else 'C')
