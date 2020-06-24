import spacy

from spacy.language import Language, Tokenizer, Doc
from settings import NOISE_CHARACTERS


def create_custom_tokenizer(nlp: Language) -> Tokenizer:
    """Extend the default Tokenizer with the ability to tokenize by the '/' character """

    custom_infixes = ['\.\.\.+', '(?<=[0-9])-(?=[0-9])', '[!&:,()/.]']

    infix_re = spacy.util.compile_infix_regex(tuple(list(nlp.Defaults.infixes) + custom_infixes))
    suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab,
                     nlp.Defaults.tokenizer_exceptions,
                     infix_finditer=infix_re.finditer,
                     suffix_search=suffix_re.search,
                     token_match=None)


def sanitizer(doc: Doc) -> Doc:
    """Removes noise characters from the document"""

    nlp = spacy.blank('pl')

    sanitized_text = doc.text

    for char in NOISE_CHARACTERS:
        sanitized_text = sanitized_text.replace(char, ' ')

    sanitized_text = ' '.join(sanitized_text.split())

    return nlp(sanitized_text)
