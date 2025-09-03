import argparse, importlib.util
from examples.seismic.rtm_core.wavefield import main_wavefield_driver

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.py")
    parser.add_argument("--iter", type=int, required=True, help="Iteration number")

    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    main_wavefield_driver(config, args.iter)
