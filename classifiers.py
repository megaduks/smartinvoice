import spacy
import plac

from spacy.language import Model, Language
from tokenizers import create_custom_tokenizer
from matchers import remove_REGON_token
from typing import Dict, List
from glob import glob
from pathlib import Path

from settings import MODELS

Language.factories['remove_REGON_token'] = remove_REGON_token


class InvoiceClassifier:

    def __init__(self):
        self.models = {}

    def fit(self, document: str, models: List[str] = None) -> Dict:
        """Applies models to the raw document to extract NERs"""

        RESULT = dict()

        models = MODELS if not models else models

        for model in models:
            if model not in self.models:
                nlp = spacy.load(MODELS[model]['model_path'])
                nlp.tokenizer = create_custom_tokenizer(nlp)
                nlp.add_pipe(remove_REGON_token, after='ner')

                self.models[model] = nlp

            doc = self.models[model](document)

            if model in [e.label_ for e in doc.ents]:
                RESULT.update({model: [doc[e.start:e.end] for e in doc.ents if e.label_ == model]})

        return RESULT


@plac.annotations(
    input_dir=("Input directory", "option", "i", Path),
    models=("Comma-separated list of models to be applied", "option", "m", str)
)
def main(input_dir: Path, models: List):
    """Loads the model, set up the pipeline and train the entity recognizer."""
    clf = InvoiceClassifier()

    input_files = input_dir.glob('*.txt')

    models = map(str.strip, models.split(',')) if models else None

    for input_file in input_files:
        with open(input_file,'rt') as f:
            print(f"{input_file} : {clf.fit(f.readline(), models=models)}")


if __name__ == '__main__':

    plac.call(main)
