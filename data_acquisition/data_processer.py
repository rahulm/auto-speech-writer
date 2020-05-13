"""
Given a CSV generated by the Web Scraper IO tool, process the data for our use.
"""
import argparse
import csv
import sys
from typing import Any, Callable, Dict, List, Text


# This field_size_limit is needed to import the data CSV
max_int = sys.maxsize
while True:
  try:
    csv.field_size_limit(max_int)
    break
  except OverflowError:
    max_int = int(max_int/10)


class EntryParser():
  def __init__(self,
    output_field: Text,
    input_field: Text,
    parser: Callable[[Any], Text]
  ) -> None:
    self.output_field: Text = output_field
    self.input_field: Text = input_field
    self.parser: Callable[[Text], Text] = parser


def process_data(
  input_file_loc: Text,
  output_file_loc: Text,
) -> None:
  print("Input csv location: {}".format(input_file_loc))
  print("Output csv location: {}".format(output_file_loc))

  """
  Original csv headers:
    web-scraper-order,
    web-scraper-start-url,
    speech_list_page,
    speech_list_page-href,
    speech_page,
    speech_page-href,
    speech-speech_name,
    speech-subtitle,
    speech-speaker_name,
    speech-raw_html,
    speech-transcript_json
  """
  
  # TODO: try to parse date from the subtitle (maybe just the year)
  output_csv_parsers: List[EntryParser] = [
    EntryParser(
      output_field="title",
      input_field="speech-speech_name",
      parser=lambda x:x
    ),
    EntryParser(
      output_field="speaker",
      input_field="speech-speaker_name",
      parser=lambda x:x
    ),
    EntryParser(
      output_field="transcript",
      input_field="speech-transcript_json",
      parser=lambda x:x
    ),

    # Extra metadata (experimental)
    EntryParser(
      output_field="subtitle",
      input_field="speech-subtitle",
      parser=lambda x:x
    ),
    EntryParser(
      output_field="year",
      input_field="speech-subtitle",
      parser=lambda x:x
    )
  ]

  output_csv_fields: List[Text] = [p.output_field for p in output_csv_parsers]

  # output_csv_fields = [
  #   "title",
  #   "speaker",
  #   "transcript",
  #   # Extra metadata (experimental)
  #   "subtitle",
  #   "year"
  # ]

  with open(input_file_loc, "r", newline="", encoding="utf-8-sig") as input_file:
    input_reader = csv.DictReader(
      input_file, quoting=csv.QUOTE_ALL
    )

    with open(output_file_loc, "w", newline="", encoding="utf-8") as output_file:
      output_writer = csv.DictWriter(
        output_file, quoting=csv.QUOTE_ALL, fieldnames=output_csv_fields
      )

      num_input_entries: int = 0

      for entry in input_reader:
        num_input_entries += 1

        # Processing each piece of data, as needed
        output_dict: Dict[Text, Text] = {
          entry_parser.output_field : entry_parser.parser(
            entry[entry_parser.input_field]
          )
          for entry_parser in output_csv_parsers
        }

        try:
          output_writer.writerow(output_dict)
        except Exception as e:
          print(e)
          print("\n\n")
          print(output_dict)
          exit(1)
      
      print("\nNumber of input entries: {}".format(num_input_entries))


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
