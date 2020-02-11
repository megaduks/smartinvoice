import spacy
import re


def NIPmatcher(doc: spacy.tokens.Doc) -> str:
    """
    Finds instances of Polish VAT ID (NIP) in the text and returns the annotated string
    :param doc: input document
    :return: string with VAT ID number enclosed in <NIP> tags
    """
    expression = r"((PL)*)(\d{10}|\d{3}-\d{3}-\d{2}-\d{2}|\d{3}-\d{2}-\d{2}-\d{3})"

    result = doc.text

    for match in re.finditer(expression, doc.text):
        start, end = match.span()
        span = doc.char_span(start, end)

        if span:
            result = result.replace(doc.text[start:end], f'<NIP>{doc.text[start:end]}</NIP>')

    return result