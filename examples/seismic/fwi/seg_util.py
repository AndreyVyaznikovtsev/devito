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


def get_source_dict(i, data, hdr, wavelet_data, zond_data):
    """
    Return geometry + traces for i-th source.

    Parameters
    ----------
    i : int
        Source index (0-based over unique sou_sloc)
    data : ndarray
        Shape (ntraces, nsamples)
    hdr : dict-like
        Header arrays
    wavelet_data : ndarray
        Shape (nshots, nwavelet_samples)
    zond_data : pd.DataFrame
        Parsed zond table

    Returns
    -------
    dict
    """
    sou_sloc_u = np.unique(hdr["sou_sloc"])
    rec_sloc_u = np.unique(hdr["rec_sloc"])
    sx_u = zond_data.sx.unique()
    rx_u = np.unique(zond_data.rx.values)

    s_sloc = sou_sloc_u[i]
    src_mask = hdr["sou_sloc"] == s_sloc

    # SEG-Y subset
    rec_slocs = hdr["rec_sloc"][src_mask]
    traces = data[src_mask]
    rz_all = -hdr["ReceiverGroupElevation"][src_mask]
    pick1_all = hdr["pick1"][src_mask]
    pick2_all = hdr["pick2"][src_mask]

    # Zond subset
    zsrc = zond_data[zond_data.sx == sx_u[i]]
    rx = zsrc["rx"].values

    # map zond rx -> segy rec_sloc
    rec_sloc_zond = rec_sloc_u[np.isin(rx_u, rx)]
    keep = np.isin(rec_slocs, rec_sloc_zond)

    # aligned arrays
    rz = rz_all[keep]
    traces = traces[keep]
    pick1 = pick1_all[keep]
    pick2 = pick2_all[keep]

    # source coordinates
    first = np.where(src_mask)[0][0]
    sou_x = hdr["SourceX"][first]
    sou_z = -hdr["SourceSurfaceElevation"][first]

    return {
        "sou_x": sou_x,
        "sou_z": sou_z,
        "data": traces,
        "rec_x": rx,
        "rec_z": rz,
        "pick1": pick1,
        "pick2": pick2,
        "wavelet": wavelet_data[i],
    }
