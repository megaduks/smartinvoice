import plac
import json

from glob import glob
from pathlib import Path

@plac.annotations(
    input_path = ("Path to the input directory with OCR files", "option", "i", Path),
    output_format = ("Output format (json, jsonl, txt)", "option", "f", str),
    output_path = ("Path to the output file", "option", "o", Path)
)
def main(input_path: Path, output_format: str, output_path: Path) -> None:

    RESULT = []

    input_files = input_path.glob("*.txt")

    for input_file in input_files:
        with input_file.open(encoding='utf8') as f:
            line = f.readline().replace('"', '').replace("'", '')
            RESULT.append({'text': line})

    with output_path.open(mode="wt") as o:
        if output_format == 'jsonl':
            json.dump(RESULT, o)
        elif output_format == 'json':
            for entry in RESULT:
                json.dump(entry, o)
                o.write('\n')
        elif output_format == 'txt':
            for entry in RESULT:
                o.write(entry['text'])


if __name__ == "__main__":
    plac.call(main)
