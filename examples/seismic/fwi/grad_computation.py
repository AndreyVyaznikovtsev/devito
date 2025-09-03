import argparse, importlib.util
from examples.seismic.fwi_core.gradient import main_compute_gradients_batched

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.py")
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    main_compute_gradients_batched(args.iter, config, args.batch_size)
