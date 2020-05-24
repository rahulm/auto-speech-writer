"""
Given a CSV containing cleaned AmericanRhetoric data, convert for fine-tuning
of the GPT2, or other model.
"""
import argparse
import csv
import sys
from typing import Callable, Dict, List, Text

import pandas as pd
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
  input_list: List[Dict[Text, Text]],
  output_file_loc: Text,
  format_name: Text
) -> None:
  print("\n---Converting data---")
  print("Output location: {}".format(output_file_loc))
  print("Format name: {}".format(format_name))

  format_template: FormatTemplate = FORMAT_TEMPLATES[format_name]
  num_entries: int = 0

  with open(output_file_loc, "w", encoding="utf-8") as output_file:
    for input_entry in input_list:
      output_txt: Text = format_template.convert(input_entry)
      output_file.write(output_txt)
      output_file.write("\n")
      num_entries += 1
  
  print("Number of entries: {}".format(num_entries))


def split_and_convert_data(
  input_file_loc: Text,
  output_file_loc: Text,
  format_name: Text,
  train_split: float,
  val_split: float,
  random_seed: int
) -> None:
  print("Input csv location: {}".format(input_file_loc))
  print("Output location: {}".format(output_file_loc))
  print("Format name: {}".format(format_name))
  print("Training split: {}".format(train_split))
  print("Validation split: {}".format(val_split))
  print("Random seed: {}".format(random_seed))

  # First read the csv
  input_df = pd.read_csv(input_file_loc, quoting=csv.QUOTE_ALL)

  if (train_split == 1.0) and (val_split == 0.0):
    # If no train-val-test split is needed, just perform the conversion directly
    convert_data(
      input_list=input_df.to_dict('records'),
      output_file_loc=output_file_loc,
      format_name=format_name
    )


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

  parser.add_argument(
    "-s", "--split", type=float, required=False, nargs=2, default=[1.0, 0.0],
    help="""The train-val-test split, if desired.
Takes two arguments, of the form 'train val', where both are floats.
The 'train' value is the percentage of the total data to be trained on.
The 'val' value is the percentage of the training data to use for validation.
The remaining (1 - 'train') will be set aside for testing.
Defaults to 'train' = 1 (all training).
    """
  )

  parser.add_argument(
    "-r", "--random", type=int, required=False, default=1234,
    help="The random seed to use. Defaults to 1234"
  )

  return parser.parse_args()


def main() -> None:
  args: argparse.Namespace = parse_args()
  
  log_file_loc = args.log if args.log else ""
  with LoggerFactory(log_file_loc) as logger_factory:
    logger_factory.set_loggers()

    split_and_convert_data(
      input_file_loc=args.input,
      output_file_loc=args.output,
      format_name=args.format,
      train_split=args.split[0],
      val_split=args.split[1],
      random_seed=args.random
    )


if __name__ == "__main__":
  main()
