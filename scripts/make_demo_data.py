"""Generate SYNTHETIC smoke-test data into data/demo/ (not the H&M dataset)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.common import load_config, setup_logging, set_seed, dataset_paths
from src.utils.synthetic import generate_demo_data

if __name__ == "__main__":
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    s = cfg["synthetic"]
    paths = generate_demo_data(dataset_paths(cfg, "demo")["raw"], s["n_customers"], s["n_articles"],
                               s["start_date"], s["end_date"], s["avg_transactions_per_customer"], cfg["seed"])
    for k, v in paths.items():
        print(f"{k}: {v}")
    print("NOTE: this is SYNTHETIC data for smoke testing only.")
