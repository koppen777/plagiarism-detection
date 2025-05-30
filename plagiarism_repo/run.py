import argparse
import os
from scripts import detection_pipeline  # type: ignore

def main():
    parser = argparse.ArgumentParser(description="Run plagiarism detection pipeline")
    parser.add_argument("--data_dir", required=True, help="Root directory containing susp/, src/, pairs")
    parser.add_argument("--output_dir", required=True, help="Directory to write XML predictions")
    parser.add_argument("--truth_dir", default=None, help="(Optional) truth XML directory for local eval")
    args = parser.parse_args()

    os.environ["DATA_DIR"] = args.data_dir
    os.environ["OUTPUT_DIR"] = args.output_dir
    if args.truth_dir:
        os.environ["TRUTH_DIR"] = args.truth_dir

    # Entry: run_detection() should be defined at bottom of detection_pipeline.py
    detection_pipeline.run_detection()

if __name__ == "__main__":
    main()
