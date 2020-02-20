import spacy
import re
from spacy.tokens import Doc

nlp = spacy.load('pl_model')


def _tagger(doc: Doc, regex: str, tag: str) -> Doc:
    """
    Finds instances of dates in the document and annotates it with date markers
    :param doc: input document
    :param regex: regular expression representing tagged item
    :param tag: string tag to enclose matches in input document
    :return: document with tagged items enclosed in tags
    """
    result = doc.text

    for match in re.finditer(regex, doc.text):
        start, end = match.span()
        span = doc.char_span(start, end)

        if span:
            result = result.replace(doc.text[start:end], f'<{tag}>{doc.text[start:end]}</{tag}>')

    return nlp(result)

def nip_tagger(doc: Doc) -> Doc:
    """
    Finds instances of Polish VAT ID (NIP) in the document and annotates it with NIP markers

    :param doc: input document
    :return: document with NIP instances enclosed in <NIP> tags
    """
    expression = r"((PL)*)(\d{10}|\d{3}-\d{3}-\d{2}-\d{2}|\d{3}-\d{2}-\d{2}-\d{3})[a-zA-Z]*"

    return _tagger(doc, regex=expression, tag='NIP')


def bank_number_tagger(doc: Doc) -> Doc:
    """
    Finds instances of bank account numbers in the document and annotates it with markers

    :param doc: input document
    :return: document with bank account numbers enclosed in <BANK_NO> tags
    """
    expression = r"((PL)*)(\d{2} (\d{4}\s*){6}|\d{2} \d{24}|\d{26})[a-zA-Z]*"

    return _tagger(doc, regex=expression, tag='BANK_NO')


def date_tagger(doc: Doc) -> Doc:
    """
    Finds instances of dates in the document and annotates it with date markers
    :param doc: input document
    :return: document with date instances enclosed in <DATE> tags
    """
    expression = r"(\d{1,2}[-.\s]\d{1,2}[-.\s]\d{4}|\d{4}[-.\s]\d{1,2}[-.\s]\d{1,2})"

    return _tagger(doc, regex=expression, tag='DATE')


def regon_tagger(doc: Doc) -> Doc:
    """
    Finds instances of REGON ID in the document and annotates it with markers

    :param doc: input document
    :return: document with REGON instances enclosed in <REGON> tags
    """
    expression = r"\d{9}"

    return _tagger(doc, regex=expression, tag='REGON')