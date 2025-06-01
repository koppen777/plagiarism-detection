from scripts.detection_pipeline import run_plagiarism_detection
import sys

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python run.py PAIRS_FILE SUSP_DIR SRC_DIR OUTPUT_DIR")
        sys.exit(1)

    pairs_file, susp_dir, src_dir, output_dir = sys.argv[1:]
    run_plagiarism_detection(pairs_file, susp_dir, src_dir, output_dir)
# run.py
# import argparse
# from scripts.detection_pipeline import run_plagiarism_detection
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Plagiarism Detection Runner")
#     parser.add_argument("pairs_file", type=str, help="Path to the pairs file")
#     parser.add_argument("susp_dir", type=str, help="Directory containing suspicious documents")
#     parser.add_argument("src_dir", type=str, help="Directory containing source documents")
#     parser.add_argument("output_dir", type=str, help="Directory to store output XML files")
#
#     args = parser.parse_args()
#
#     run_plagiarism_detection(
#         pairs_file=args.pairs_file,
#         susp_dir=args.susp_dir,
#         src_dir=args.src_dir,
#         output_dir=args.output_dir
#     )
