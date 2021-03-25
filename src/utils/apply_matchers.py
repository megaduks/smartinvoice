import plac
import spacy

from pathlib import Path
from typing import List
from settings import MODELS


@plac.annotations(
    input_dir=("Input directory with text files", "option", "i", Path),
    matchers=("Comma-separated list of matchers to be applied", "option", "m", str)
)
def main(input_dir: Path, matchers: List[str]) -> None:
    """Applies matchers to a set of text files in a directory"""

    nlp = spacy.load('pl_model')
    DOCS = []

    input_files = input_dir.glob('*.txt')
    matchers = map(str.strip, matchers.split(','))

    for input_file in input_files:
        with open(input_file, 'rt') as f:
            DOCS.append(f.readline())

    for matcher in matchers:
        matcher = MODELS[matcher]['matcher_factory'](nlp)
        nlp.add_pipe(matcher, last=True)

    for doc in nlp.pipe(DOCS):
        print([(e.label_, e.text) for e in doc.ents if e.label_ in MODELS])


if __name__ == '__main__':
    plac.call(main)
