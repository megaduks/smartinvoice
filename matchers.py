import spacy
import re
from spacy.tokens import Doc

nlp = spacy.load('pl_model')


def NIP_matcher(doc: Doc) -> Doc:
    """
    Finds instances of Polish VAT ID (NIP) in the document and returns the annotated text
    :param doc: input document
    :return: spacy Doc with VAT ID number enclosed in <NIP> tags
    """
    expression = r"((PL)*)(\d{10}|\d{3}-\d{3}-\d{2}-\d{2}|\d{3}-\d{2}-\d{2}-\d{3})[a-zA-Z]*"

    result = doc.text

    for match in re.finditer(expression, doc.text):
        start, end = match.span()
        span = doc.char_span(start, end)

        if span:
            result = result.replace(doc.text[start:end], f'<NIP>{doc.text[start:end]}</NIP>')

    return nlp(result)


def bank_number_matcher(doc: Doc) -> Doc:
    """
    Finds instances of bank account numbers in the document and returns the annotated text

    :param doc: input document
    :return: spacy Doc with bank account numbers enclosed in <BANK_NO> tags
    """
    expression = r"((PL)*)(\d{2} (\d{4}\s*){6}|\d{2} \d{24}|\d{26})[a-zA-Z]*"

    result = doc.text

    for match in re.finditer(expression, doc.text):
        start, end = match.span()
        span = doc.char_span(start, end)

        if span:
            result = result.replace(doc.text[start:end], f'<BANK_NO>{doc.text[start:end]}</BANK_NO>')

    return nlp(result)
