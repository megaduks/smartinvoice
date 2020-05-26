import spacy
import numpy.random as random
import plac

from typing import List
from spacy.tokens import Doc
from spacy.language import Model, Language
from spacy.util import minibatch, compounding

from pathlib import Path

from settings import MODELS
from tokenizers import create_custom_tokenizer


def _load_raw_data(nlp: Model, input_dir: Path, clean_str: bool = True) -> List[Doc]:
    """Reads raw TXT files, removes noisy characters and returns pre-processed documents

    Args:
        nlp: spaCy model object
        input_dir: directory with raw TXT files
        clean_str: should we remove unnecesary noise in strings
    Returns:
        list of spaCy Doc objects
    """
    TEXTS = []

    files = input_dir.glob('*.txt')

    for file in files:
        with open(file, 'r') as f:
            TEXTS.append(f.read())

    return nlp.pipe(TEXTS)


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    ner_model=("Required NER model name", "option", "e", str),
    input_dir=("Optional input directory", "option", "i", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model: str = None, ner_model: str = None, input_dir: Path = None, n_iter: int = 100):
    """Loads the model, set up the pipeline and train the entity recognizer."""

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print(f"Loaded model '{model}'")
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    nlp.tokenizer = create_custom_tokenizer(nlp)

    matcher = MODELS[ner_model]['matcher_factory'](nlp)

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
        nlp.add_pipe(matcher, name=MODELS[ner_model]['matcher_name'], before='ner')
        ner.add_label(matcher.label)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")
        nlp.add_pipe(matcher, name=MODELS[ner_model]['matcher_name'], before='ner')
        ner.add_label(matcher.label)

    docs = _load_raw_data(nlp, input_dir=input_dir)

    TRAIN_DATA = []

    for doc in docs:
        spans = [doc[e.start:e.end] for e in doc.ents if e.label_ == matcher.label]
        entities = [(span.start_char, span.end_char, matcher.label) for span in spans]

        if entities:
            training_example = (doc.text, {"entities": entities})
            TRAIN_DATA.append(training_example)

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec", MODELS[ner_model]['matcher_name']]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for _ in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.25,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    output_dir = Path(MODELS[ner_model]['model_path'])

    if not output_dir.exists():
        output_dir.mkdir()

    with nlp.disable_pipes(MODELS[ner_model]['matcher_name']):
        nlp.to_disk(output_dir)


if __name__ == '__main__':
    plac.call(main)
