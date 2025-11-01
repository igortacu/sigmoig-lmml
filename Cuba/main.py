"""Remove sentences that start with the letters S, T, O, or P
"""
from pathlib import Path
import argparse
import sys
import re
DEFAULT_LETTERS = set("STOP")
def should_drop_line(line: str, letters: set) -> bool:
	"""Return True if the line's first non-space character is in letters.

	Empty lines are preserved.
	"""
	stripped = line.lstrip()
	if not stripped:
		return False
	first = stripped[0]
	return first.upper() in letters


def process_file(in_path: Path, out_path: Path, letters: set):
	text = in_path.read_text(encoding="utf-8")
	lines = text.splitlines()
	out_lines = [l for l in lines if not should_drop_line(l, letters)]
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

