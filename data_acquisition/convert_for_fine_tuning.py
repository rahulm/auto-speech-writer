"""
Given a CSV containing cleaned AmericanRhetoric data, convert for fine-tuning
of the GPT2, or other model.
"""
import argparse
import csv
import os
import sys
from typing import Callable, Dict, List, Text, Tuple

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
    convert_fn: Callable[[Dict, bool], Text]
  ) -> None:
    self.name: Text = name
    self.convert: Callable[[Dict, bool], Text] = convert_fn


def make_tag(header: Text, content: Text) -> Text:
  return '<{}="{}">'.format(header, content)


# NOTE: This method does not actually cover all cases. May need to redo.
END_OF_SENTENCE_PUNCTUATION: List[Text] = [
  ".",
  "!",
  "?",
  ".'",
  ".\"",
  "!'",
  "!\"",
  "?'",
  "?\""
]
def truncate_incomplete_sentences(x: Text) -> Text:
  last_punc: int = max(x.rfind(p) for p in END_OF_SENTENCE_PUNCTUATION)
  if last_punc == -1:
    return x
  return x[:last_punc + 1]


def make_tag_with_truncate(
  header: Text,
  content: Text,
  truncate_summary: bool
) -> Text:
  if truncate_summary and (header == "summary"):
    return make_tag(header, truncate_incomplete_sentences(content))
  else:
    return make_tag(header, content)


def convert_tag_all_except_transcript() -> Callable[[Dict, bool], Text]:
  input_headers: List[Text] = [
    "title",
    "speaker",
    "year",
    "summary"
  ]
  transcript_header: Text = "transcript"
  def inner_function(d: Dict, truncate_summary: bool) -> Text:
    output_row_list: List[Text] = [
      make_tag_with_truncate(
        header=header,
        content=d[header],
        truncate_summary=truncate_summary
      )
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
  format_name: Text,
  truncate_summaries: bool
) -> None:
  print("\n---Converting data---")
  print("Output location: {}".format(output_file_loc))
  print("Format name: {}".format(format_name))
  print("Truncate summaries: {}".format(truncate_summaries))

  format_template: FormatTemplate = FORMAT_TEMPLATES[format_name]
  num_entries: int = 0

  with open(output_file_loc, "w", encoding="utf-8") as output_file:
    for input_entry in input_list:
      output_txt: Text = format_template.convert(
        input_entry,
        truncate_summaries
      )
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
  random_seed: int,
  truncate_summaries: bool
) -> None:
  print("Input csv location: {}".format(input_file_loc))
  print("Output location: {}".format(output_file_loc))
  print("Format name: {}".format(format_name))
  print("Training split: {}".format(train_split))
  print("Validation split: {}".format(val_split))
  print("Random seed: {}".format(random_seed))
  print("Truncate summaries: {}".format(truncate_summaries))

  # First read the csv
  input_df = pd.read_csv(input_file_loc, quoting=csv.QUOTE_ALL)

  # Compile the tuples of (data split, output file name) to process
  split_pairs: List[Tuple[List, Text]] = []

  if (train_split == 1.0) and (val_split == 0.0):
    # No train-test split, so add to split_pairs
    split_pairs.append((input_df.to_dict("records"), output_file_loc))
  else:
    # Get the output file prefix and extension
    output_file_prefix, output_file_ext = os.path.splitext(output_file_loc)

    if train_split < 1.0:
      # Perform a train-test split
      test_split: float = 1.0 - train_split
      test_df = input_df.sample(
        frac=test_split,
        replace=False,
        random_state=random_seed
      )
      test_output_file_loc: Text = "{}-{}{}".format(
        output_file_prefix,
        "test",
        output_file_ext
      )
      split_pairs.append((test_df.to_dict("records"), test_output_file_loc))

      train_df = input_df.loc[~input_df.index.isin(test_df.index)]
    else:
      # no train-test split
      train_df = input_df
    
    if val_split > 0.0:
      # There is a train-val split

      val_df = train_df.sample(
        frac=val_split,
        replace=False,
        random_state=random_seed
      )
      val_output_file_loc: Text = "{}-{}{}".format(
        output_file_prefix,
        "val",
        output_file_ext
      )
      split_pairs.append((val_df.to_dict("records"), val_output_file_loc))

      train_df = train_df.loc[~train_df.index.isin(val_df.index)]
    
    train_output_file_loc: Text = "{}-{}{}".format(
      output_file_prefix,
      "train",
      output_file_ext
    )
    split_pairs.append((train_df.to_dict("records"), train_output_file_loc))
  
  
  for split_data, split_output_file_loc in split_pairs:
    convert_data(
      input_list=split_data,
      output_file_loc=split_output_file_loc,
      format_name=format_name,
      truncate_summaries=truncate_summaries
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

  parser.add_argument(
    "-t", "--truncate", action="store_true", default=False,
    help="""Remove unfinished sentences from the summaries.
This is temporary, need to handle earlier when generating summaries."""
  )

  args = parser.parse_args()

  split_train, split_val = args.split
  if (split_train < 0) or (split_train > 1):
    parser.error("Train split {} is invalid.".format(split_train))
  if (split_val < 0) or (split_val > 1):
    parser.error("Val split {} is invalid.".format(split_val))
  
  return args


def main() -> None:
  args: argparse.Namespace = parse_args()
  
  log_file_loc = args.log if args.log else ""
  with LoggerFactory(log_file_loc) as logger_factory:
    logger_factory.set_loggers()

    print("{}\n\n".format(args))

    split_and_convert_data(
      input_file_loc=args.input,
      output_file_loc=args.output,
      format_name=args.format,
      train_split=args.split[0],
      val_split=args.split[1],
      random_seed=args.random,
      truncate_summaries=args.truncate
    )


if __name__ == "__main__":
  main()
