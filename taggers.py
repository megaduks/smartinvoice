import spacy

from spacy.matcher import Matcher
from spacy.tokens import Doc
from spacy.language import  Language

Doc.set_extension('invoice_type', default=None)

# TODO: faktura do paragonu
# TODO: rachunek
# TODO: faktura uproszczona
# TODO: faktura marża
# TODO: faktura zaliczkowa
# TODO: faktura końcowa


class InvoiceTypeTagger():
    """Simple tagger which detects the type of the invoice and adds document property invoice_type"""
    def __init__(self, nlp: Language):
        self.nlp = nlp
        self.matcher = Matcher(self.nlp.vocab)

    def __call__(self, doc: Doc, *args, **kwargs) -> Doc:
        """detects the type of the invoice and sets the value of the document property"""

        patterns = [
            [{'LOWER': 'faktura'}, {'LOWER': 'proforma'}],
            [{'LOWER': 'faktura'}, {'LOWER': 'pro'}, {'LOWER': 'forma'}],
            [{'LOWER': 'faktura'}, {'LOWER': 'pro-forma'}],
            [{'LOWER': 'f-ra'}, {'LOWER': 'proforma'}],
            [{'LOWER': 'f-ra'}, {'LOWER': 'pro'}, {'LOWER': 'forma'}],
            [{'LOWER': 'f-ra'}, {'LOWER': 'pro-forma'}],
        ]
        self.matcher.add("PROFORMA", None, *patterns)

        patterns = [
            [{'LOWER': 'faktura'}, {'LOWER': 'zaliczkowa'}],
            [{'LOWER': 'f-ra'}, {'LOWER': 'zaliczkowa'}],
        ]
        self.matcher.add('ZALICZKOWA', None, *patterns)

        patterns = [
            [{'LOWER': 'faktura'}, {'LOWER': 'korygująca'}],
            [{'LOWER': 'faktura'}, {'LOWER': 'korygujaca'}],
            [{'LOWER': 'f-ra'}, {'LOWER': 'korygująca'}],
            [{'LOWER': 'f-ra'}, {'LOWER': 'korygujaca'}],
        ]
        self.matcher.add('KORYGUJĄCA', None, *patterns)

        patterns = [
            [{'LOWER': 'faktura'}, {'LOWER': 'vat-mp'}],
            [{'LOWER': 'f-ra'}, {'LOWER': 'vat-mp'}],
            [{'LOWER': 'metoda'}, {'LOWER': 'kasowa'}],
        ]
        self.matcher.add('MAŁEGO PODATNIKA', None, *patterns)

        patterns = [
            [{'LOWER': 'faktura'}, {'LOWER': 'vat'}, {'LOWER': 'rr'}],
            [{'LOWER': 'f-ra'}, {'LOWER': 'vat'}, {'LOWER': 'rr'}],
        ]
        self.matcher.add('ROLNIK RYCZAŁTOWY', None, *patterns)

        patterns = [
            [{'LOWER': 'faktura'}, {'LOWER': 'vat'}],
            [{'LOWER': 'faktura'}, {'LOWER': 'nr'}, {'LIKE_NUM': True}],
            [{'LOWER': 'faktura'}, {'LOWER': 'nr'}, {'IS_PUNCT': True}, {'LIKE_NUM': True}],
            [{'LOWER': 'f-ra'}, {'LOWER': 'vat'}],
            [{'LOWER': 'f-ra'}, {'LOWER': 'nr'}, {'LIKE_NUM': True}],
            [{'LOWER': 'f-ra'}, {'LOWER': 'nr'}, {'IS_PUNCT': True}, {'LIKE_NUM': True}],
        ]
        self.matcher.add('VAT', None, *patterns)

        for match_id, start, end in self.matcher(doc):
            doc._.invoice_type = self.nlp.vocab.strings[match_id]

        return doc
