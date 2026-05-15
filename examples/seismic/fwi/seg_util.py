import struct
import numpy as np
from segyio.tracefield import keys as trkeys


def ibm_to_ieee(ibm_bytes):
    """Convert 4-byte IBM float to Python float."""
    if len(ibm_bytes) != 4:
        raise ValueError("IBM float must be 4 bytes")

    b = struct.unpack(">I", ibm_bytes)[0]
    if b == 0:
        return 0.0

    sign = (b >> 31) & 0x01
    exponent = (b >> 24) & 0x7F
    fraction = b & 0x00FFFFFF

    mantissa = fraction / float(0x1000000)
    value = mantissa * (16 ** (exponent - 64))

    return -value if sign else value


def read_raw_trace_header(f, trace_idx, byte_loc, fmt):
    """
    byte_loc = SEG-Y byte location (1-based, like 197)
    fmt examples: '4R_IBM', '4R_IEEE'
    """
    # 240-byte trace header
    raw = f.header[trace_idx].buf  # raw bytes

    start = byte_loc - 1  # convert to python 0-based
    chunk = raw[start : start + 4]

    if fmt == "4R_IBM":
        return ibm_to_ieee(chunk)

    elif fmt == "4R_IEEE":
        return struct.unpack(">f", chunk)[0]

    else:
        raise ValueError(fmt)


def get_headers(f, header_list, remap_string=None):
    out = {}

    # standard headers
    for h in header_list:
        name = list(trkeys.keys())[list(trkeys.values()).index(h)]
        out[name] = np.array(f.attributes(h)[:])

    # remapped fields
    if remap_string:
        entries = [x.strip() for x in remap_string.split("/")]
        for entry in entries:
            name, nbytes, ftype, offset = [x.strip() for x in entry.split(",")]
            fmt = f"{nbytes}_{ftype}"

            vals = np.array([read_raw_trace_header(f, i, int(offset), fmt) for i in range(f.tracecount)])

            out[name] = vals

    return out
