import spacy
import plac
import pandas as pd

from spacy.language import Model, Language
from tokenizers import create_custom_tokenizer
from matchers import remove_REGON_token
from typing import Dict, List
from glob import glob
from pathlib import Path
from ludwig.api import LudwigModel

from settings import MODELS

Language.factories['remove_REGON_token'] = remove_REGON_token


class InvoicePhotoClassifier:

    def __init__(self, model_path: Path):
        self.model = LudwigModel.load(model_path)

    def predict(self, image_file: Path) -> bool:
        """Verifies if the photo contains an image of an invoice"""
        df = pd.DataFrame(
            {
                'image_path': [image_file]
            }
        )
        prediction = self.model.predict(data_df=df)

        return prediction.class_predictions[0] == 'invoice'


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


class InvoiceNERClassifier:

    def __init__(self, nlp: spacy.language.Language):
        self.nlp = nlp

    def fit(self, document: str) -> Dict:

        RESULT = dict()

        def strip_ent_labels(label: str) -> str:
            """Strips all spaCy ent labels (xxxDate, xxxTime, xxxMoney) from the entity body"""
            tokens_to_strip = ['%', 'zl', 'zł', 'PLN', 'PL']

            clean_label = [
                token
                for token in label.split()
                if not token.startswith('xxx')
                and not token in tokens_to_strip
            ]
            return ' '.join(clean_label)

        doc = self.nlp(document)

        for ent in doc.ents:
            RESULT.update({ent.label_: strip_ent_labels(doc[ent.start:ent.end].text)})

        return RESULT


@plac.annotations(
    input_dir=("Input directory", "option", "i", Path),
    model=("Directory with invoice entity recognizer model", "option", "m", Path)
)
def main(input_dir: Path, model: Path):
    """Loads the model, set up the pipeline and train the entity recognizer."""
    nlp = spacy.load(model)
    clf = InvoiceNERClassifier(nlp=nlp)

    input_files = input_dir.glob('*.txt')

    for input_file in input_files:
        with open(input_file,'rt') as f:
            print(f"{input_file} : {clf.fit(f.readline())}")


if __name__ == '__main__':

    plac.call(main)
