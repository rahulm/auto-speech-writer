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


FORMAT_TEMPLATES: Dict[Text, FormatTemplate] = {
  "tag_except_transcript" : FormatTemplate(
    name="tag_except_transcript",
    convert_fn=lambda x: str(x)
  )
}


FORMAT_NAMES: List[Text] = list(FORMAT_TEMPLATES.keys())
FORMAT_DEFAULT: Text = "tag_except_transcript"


def convert_data(input_file_loc: Text, output_file_loc: Text) -> None:
  return


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

    # convert_data(

    # )

    # process_data(
    #   input_file_loc = args.input,
    #   output_file_loc = args.output
    # )


if __name__ == "__main__":
  main()
