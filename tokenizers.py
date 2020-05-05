import spacy

from spacy.language import Language, Tokenizer


def create_custom_tokenizer(nlp: Language) -> Tokenizer:
    """Extend the default Tokenizer with the ability to tokenize by the '/' character """

    custom_infixes = ['\.\.\.+', '(?<=[0-9])-(?=[0-9])', '[!&:,()/]']

    infix_re = spacy.util.compile_infix_regex(tuple(list(nlp.Defaults.infixes) + custom_infixes))
    suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab,
                     nlp.Defaults.tokenizer_exceptions,
                     infix_finditer=infix_re.finditer,
                     suffix_search=suffix_re.search,
                     token_match=None)
