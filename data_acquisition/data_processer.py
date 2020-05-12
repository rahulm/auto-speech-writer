"""
Given a CSV generated by the Web Scraper IO tool, process the data for our use.
"""
import argparse
import csv
import sys
from typing import Text


# This field_size_limit is needed to import the data CSV
max_int = sys.maxsize
while True:
  try:
    csv.field_size_limit(max_int)
    break
  except OverflowError:
    max_int = int(max_int/10)


def process_data(
  input_file_loc: Text,
  output_file_loc: Text,
) -> None:
  print("Input csv location: {}".format(input_file_loc))
  print("Output csv location: {}".format(output_file_loc))
  
  num_entries: int = 0

  with open(input_file_loc, "r", newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(
      csv_file, quoting=csv.QUOTE_ALL
    )
    for entry in reader:
      num_entries += 1

      print("{}".format(repr(entry["speech-speaker_name"])))
  
  print("\nnum_entries: {}".format(num_entries))


def parse_args() -> argparse.Namespace:
  parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Process web scraped data.")
  parser.add_argument(
    "-i", "--input", type=str, required=True,
    help="Input csv location"
  )

  parser.add_argument(
    "-o", "--output", type=str, required=True,
    help="Location (csv) where to output processed data"
  )

  return parser.parse_args()


def main() -> None:
  args: argparse.Namespace = parse_args()
  process_data(
    input_file_loc = args.input,
    output_file_loc = args.output
  )


if __name__ == "__main__":
  main()
