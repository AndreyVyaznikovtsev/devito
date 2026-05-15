import pandas as pd
from io import StringIO


def read_zond(path):
    """
    Read zond .dat file with three sections:
      1. main data table
      2. geocoord
      3. topo

    Returns:
        data : pd.DataFrame
        geocoord : dict
        topo : pd.DataFrame
    """
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip() for line in f if line.strip()]

    # --- locate section markers ---
    geo_idx = next(i for i, line in enumerate(lines) if line.startswith("geocoord"))
    topo_idx = next(i for i, line in enumerate(lines) if line.startswith("topo"))

    # --- main data section ---
    data_lines = lines[:geo_idx]
    data = pd.read_csv(StringIO("\n".join(data_lines)), sep=r"\s+", engine="python", na_values="*")

    # --- geocoord section ---
    geo_parts = lines[geo_idx].split()
    # expected: geocoord sx sy rx ry
    geocoord = {
        "sx": float(geo_parts[1]),
        "sy": float(geo_parts[2]),
        "rx": float(geo_parts[3]),
        "ry": float(geo_parts[4]),
    }

    # --- topo section ---
    topo_lines = lines[topo_idx + 1 :]
    topo = pd.read_csv(StringIO("\n".join(topo_lines)), sep=r"\s+", engine="python", names=["offset", "elevation"])

    return data, geocoord, topo
