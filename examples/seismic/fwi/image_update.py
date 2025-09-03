import argparse
import importlib
from examples.seismic.fwi_core.image import update_vp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config (e.g., rtm.mex_4_5.config)")
    parser.add_argument("--iter", type=int, required=True, help="Iteration number")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    update_vp(config, args.iter)


if __name__ == "__main__":
    main()
