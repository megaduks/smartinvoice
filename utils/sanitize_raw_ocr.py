import plac
import spacy

from pathlib import Path
from tokenizers import sanitizer


@plac.annotations(
    input_dir=("Input directory with input raw OCR text files", "option", "i", str),
    output_dir=("Output directory to store sanitized OCR text files", "option", "o", str)
)
def main(input_dir: str, output_dir: str) -> None:
    """Applies a sanitizer to all input raw OCR files in a given directory"""

    nlp = spacy.blank('pl')
    nlp.add_pipe(sanitizer, name="sanitizer", first=True)

    input_files = Path(input_dir).glob('*.txt')
    output_dir = Path(output_dir)

    for input_file in input_files:
        with open(input_file, 'rt') as f:
            doc = nlp(f.readline())
            output_file = output_dir / input_file.name
            output_file.write_text(doc.text)


if __name__ == '__main__':
    plac.call(main)
