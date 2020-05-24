"""
Given a CSV containing cleaned AmericanRhetoric data, convert for fine-tuning
of the GPT2, or other model.
"""
import argparse
import csv
import sys
from typing import Callable, Dict, List, Text

from loggers import LoggerFactory

# This field_size_limit is needed to import the data CSV
max_int = sys.maxsize
while True:
  try:
    csv.field_size_limit(max_int)
    break
  except OverflowError:
    max_int = int(max_int/10)


class FormatTemplate():
  def __init__(
    self,
    name: Text,
    convert_fn: Callable[[Dict], Text]
  ) -> None:
    self.name: Text = name
    self.convert: Callable[[Dict], Text] = convert_fn


def convert_tag_all_except_transcript() -> Callable[[Dict], Text]:
  input_headers: List[Text] = [
    "title",
    "speaker",
    "year",
    "summary"
  ]
  transcript_header: Text = "transcript"
  def inner_function(d: Dict) -> Text:
    output_row_list: List[Text] = [
      '<{}="{}">'.format(header, d[header])
      for header in input_headers
    ]
    output_row_list.append(d[transcript_header])
    return '\n'.join(output_row_list)
  return inner_function


FORMAT_TEMPLATES: Dict[Text, FormatTemplate] = {
  "tag_all_except_transcript" : FormatTemplate(
    name="tag_all_except_transcript",
    convert_fn=convert_tag_all_except_transcript()
  )
}


FORMAT_NAMES: List[Text] = list(FORMAT_TEMPLATES.keys())
FORMAT_DEFAULT: Text = "tag_all_except_transcript"


def convert_data(
  input_file_loc: Text,
  output_file_loc: Text,
  format_name: Text
) -> None:
  format_template: FormatTemplate = FORMAT_TEMPLATES[format_name]
  num_entries: int = 0

  with open(input_file_loc, "r", newline="", encoding="utf-8") as input_file:
    input_reader = csv.DictReader(
      input_file, quoting=csv.QUOTE_ALL
    )
    with open(output_file_loc, "w", encoding="utf-8") as output_file:
      for input_entry in input_reader:
        output_txt: Text = format_template.convert(input_entry)
        output_file.write(output_txt)
        output_file.write("\n")
        num_entries += 1
  
  print("Number of entries: {}".format(num_entries))


def parse_args() -> argparse.Namespace:
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Convert cleaned CSV into a format that can be used for fine-tuning."
  )
  parser.add_argument(
    "-i", "--input", type=str, required=True,
    help="Input csv location. Should be output of the process_ra_data.py script"
  )

  parser.add_argument(
    "-o", "--output", type=str, required=True,
    help="Location of where to output the converted data"
  )

  parser.add_argument(
    "-l", "--log", type=str, required=False,
    help="(Optional) Location where to output .txt log file"
  )
  
  parser.add_argument(
    "-f", "--format", type=str, default=FORMAT_DEFAULT,
    help="The type of format to convert into. Choose from {}".format(
      FORMAT_NAMES
    ),
    choices=FORMAT_NAMES
  )

  return parser.parse_args()


def main() -> None:
  args: argparse.Namespace = parse_args()
  
  log_file_loc = args.log if args.log else ""
  with LoggerFactory(log_file_loc) as logger_factory:
    logger_factory.set_loggers()

    convert_data(
      input_file_loc=args.input,
      output_file_loc=args.output,
      format_name=args.format
    )


if __name__ == "__main__":
  main()
