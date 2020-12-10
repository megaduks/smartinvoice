import plac
import spacy
import logging

from pathlib import Path
from typing import List
from settings import MODELS

from tokenizers import create_custom_tokenizer


@plac.annotations(
    input_dir=("Input directory with text files", "option", "i", Path),
    output_dir=("Output directory where text files are stored", "option", "o", Path),
    matchers=("Comma-separated list of matchers to be applied", "option", "m", str)
)
def main(input_dir: Path, output_dir: Path, matchers: List[str]) -> None:
    """Applies matchers to a set of text files in a directory"""

    FORMAT = "%(levelname)-5s %(asctime)-15s %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)
    logging.info('Cleaning raw OCR files')

    logging.info('Loading language model')
    nlp = spacy.load('pl_core_news_lg')
    nlp.tokenizer = create_custom_tokenizer(nlp=nlp)

    DOCS = []
    RESULTS = []
    BAD_CHARACTERS = list('!#^&*()+{}[]|?<>=')

    input_files = input_dir.glob('*.txt')
    matchers = map(str.strip, matchers.split(','))

    for input_file in input_files:
        with open(input_file, 'rt') as f:
            _file = ' '.join([line.rstrip() for line in f])
            DOCS.append(_file)

    logging.info('Loading matchers')
    for matcher in matchers:
        matcher = MODELS[matcher]['matcher_factory'](nlp)
        nlp.add_pipe(matcher, after='parser')

    logging.info('Injecting matched entities')
    for doc in nlp.pipe(DOCS):

        _text = doc.text
        offset = 0

        for e in doc.ents:
            if e.label_ != 'date':
                injection = f' xxx{e.label_} '
                _text = _text[:e.start_char + offset] + injection + _text[e.start_char + offset:]
                offset += len(injection)

        for c in BAD_CHARACTERS:
            _text = _text.replace(c, '')

        new_doc = nlp.make_doc(_text)
        RESULTS.append(new_doc)

    logging.info('Saving processed OCR files')
    input_files = input_dir.glob('*.txt')
    for i, input_file in enumerate(input_files):
        output_file = output_dir / input_file.name
        output_file.write_text(RESULTS[i].text)


if __name__ == '__main__':
    plac.call(main)
