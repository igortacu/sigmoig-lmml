"""Keep only sentences that start with the letters S, T, O, or P.

This script reads `input.txt` next to this file and writes `output.txt` keeping
only the lines whose first non-whitespace character is one of the specified
letters (default: STOP).
"""
from pathlib import Path
import argparse
DEFAULT_LETTERS = set("STOP")
def should_drop_line(line: str, letters: set) -> bool:

	stripped = line.lstrip()
	if not stripped:
		return False
	first = stripped[0]
	return first.upper() in letters


def process_file(in_path: Path, out_path: Path, letters: set):
	text = in_path.read_text(encoding="utf-8")
	lines = text.splitlines()
	# Keep only lines that match the letter filter (i.e., first letter in set)
	out_lines = [l for l in lines if should_drop_line(l, letters)]
	out_path.write_text("\n".join(out_lines), encoding="utf-8")


def main(argv=None):
	p = argparse.ArgumentParser(description="Remove sentences that start with specified letters (default: STOP).")
	p.add_argument("-i", "--input", default="input.txt", help="Input text file (default: input.txt)")
	p.add_argument("-o", "--output", default="output.txt", help="Output text file (default: output.txt)")
	p.add_argument("-l", "--letters", default="STOP", help="Letters which, if a sentence starts with them, cause the sentence to be removed (default: STOP)")
	args = p.parse_args(argv)

	base = Path(__file__).parent
	in_path = Path(args.input)
	if not in_path.is_absolute():
		in_path = base / in_path
	out_path = Path(args.output)
	if not out_path.is_absolute():
		out_path = base / out_path

	letters = set(args.letters.upper())

	if not in_path.exists():
		print(f"Input file not found: {in_path}")
		return 2

	process_file(in_path, out_path, letters)
	print(f"Wrote {out_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

