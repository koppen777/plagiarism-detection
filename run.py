from scripts.detection_pipeline import run_plagiarism_detection
import sys

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python run.py PAIRS_FILE SUSP_DIR SRC_DIR OUTPUT_DIR")
        sys.exit(1)

    pairs_file, susp_dir, src_dir, output_dir = sys.argv[1:]
    run_plagiarism_detection(pairs_file, susp_dir, src_dir, output_dir)

