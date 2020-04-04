from spacy.matcher import Matcher
from spacy.tokens import Doc, Span


class InvoiceMatcher():
    """Generic class which is extended with particular patterns and labels"""

    def __init__(self, nlp, label):
        """Creates a new NIP matcher using a shared vocabulary object"""
        self.matcher = Matcher(nlp.vocab)
        self.label = label

    def __call__(self, doc: Doc) -> Doc:
        matches = self.matcher(doc)
        spans = []
        for match_id, start, end in matches:
            spans.append(Span(doc, start, end, label=self.label))
        doc.ents = list(doc.ents) + spans
        return doc


class NIPMatcher(InvoiceMatcher):
    """Extracts matched NIP instances and adds them as document entities with the label NIP"""

    def __init__(self, nlp):
        """Creates a new NIP matcher using a shared vocabulary object"""
        super(NIPMatcher, self).__init__(nlp, label='NIP')
        patterns = [
            # an optional 2-letter country code followed by 10 consecutive digits
            [{'TEXT': {'REGEX': '([a-zA-Z][a-zA-Z])?\d{10}'}}],
            # 10 consecutive digits followed by an optional 2-letter country code
            [{'TEXT': {'REGEX': '(\d{10}[a-zA-Z][a-zA-Z])?'}}],
            # an optional 2-letter country code followed by 3-3-2-2 digit pattern (either dashes or spaces)
            [{'TEXT': {'REGEX': '([a-zA-Z][a-zA-Z])?\d{3}(-|\s)\d{3}(-|\s)\d{2}(-|\s)\d{2}'}}],
            # 3-3-2-2 digit pattern (either dashes or spaces) followed by an optional 2-letter country code
            [{'TEXT': {'REGEX': '\d{3}(-|\s)\d{3}(-|\s)\d{2}(-|\s)\d{2}([a-zA-Z][a-zA-Z])?'}}],
            # an optional 2-letter country code followed by 3-2-2-3 digit pattern (either dashes or spaces)
            [{'TEXT': {'REGEX': '([a-zA-Z][a-zA-Z])?\d{3}(-|\s)\d{2}(-|\s)\d{2}(-|\s)\d{3}'}}],
            # 3-2-2-3 digit pattern (either dashes or spaces) followed by an optional 2-letter country code
            [{'TEXT': {'REGEX': '\d{3}(-|\s)\d{2}(-|\s)\d{2}(-|\s)\d{3}([a-zA-Z][a-zA-Z])?'}}]
        ]
        self.matcher.add(self.label, None, *patterns)


class BankNumberMatcher(InvoiceMatcher):
    """Extracts matched bank account numbers and adds them as document entities with the label BANK_ACCOUNT_NO"""

    def __init__(self, nlp):
        """Creates a new bank number account matcher using a shared vocabulary object"""
        super(BankNumberMatcher, self).__init__(nlp, label='BANK_ACCOUNT_NO')
        patterns = [
            # an optional 2-letter country code followed by 26 consecutive digits
            [{'TEXT': {'REGEX': '([a-zA-Z][a-zA-Z])?\d{26}'}}],
            # 26 consecutive digits followed by an optional 2-letter country code
            [{'TEXT': {'REGEX': '(\d{26}[a-zA-Z][a-zA-Z])?'}}],
            # an optional 2-letter country code followed by 2-4-4-4-4-4-4 digit pattern (either dashes or spaces)
            [{'TEXT': {'REGEX': '([a-zA-Z][a-zA-Z])?\d{2}(-|\s)(\d{4}(-|\s)){5}\d{4}'}}],
            # an optional 2-letter country code followed by 2-24 digit pattern (either dashes or spaces)
            [{'TEXT': {'REGEX': '([a-zA-Z][a-zA-Z])?\d{2}(-|\s)\d{24}'}}],
            # 2-4-4-4-4-4-4 digit pattern (either dashes or spaces) followed by an optional 2-letter country code
            [{'TEXT': {'REGEX': '\d{2}(-|\s)(\d{4}(-|\s)){5}\d{4}([\sa-zA-Z][a-zA-Z])?'}}],
            # 2-24 digit pattern (either dashes or spaces) followed by an optional 2-letter country code
            [{'TEXT': {'REGEX': '\d{2}(-|\s)\d{24}([\sa-zA-Z][a-zA-Z])?'}}]
        ]
        self.matcher.add(self.label, None, *patterns)
