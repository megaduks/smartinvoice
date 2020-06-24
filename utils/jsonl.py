import json
import plac
import codecs

from pathlib import Path


@plac.annotations(
    input_dir=("Input directory with text files", "option", "i", str),
    output_file=("Output file where JSONL for Prodigy will be saved", "option", "o", str)
)
def dir2jsonl(input_dir: str, output_file: str) -> None:
    """Transforms all text files in a given directory to JSONL format and saves them all to a file"""

    LINES = []

    file_list = Path(input_dir).glob('*.txt')

    for file in file_list:
        with codecs.open(file, mode='r', encoding='utf-8') as f:
            line = f.readline()
            LINES.append(json.dumps({'text': line}, ensure_ascii=True))

    Path(output_file).open('w').write('\n'.join(LINES))


if __name__ == '__main__':
    plac.call(dir2jsonl)
