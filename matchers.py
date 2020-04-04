from spacy.matcher import Matcher
from spacy.tokens import Doc, Span


class NIPMatcher():
    """Extracts matched NIP instances and adds them as document entities with the label NIP"""

    def __init__(self, nlp):
        """Creates a new NIP matcher using a shared vocabulary object"""
        self.matcher = Matcher(nlp.vocab)
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
        self.matcher.add('NIP', None, *patterns)

    def __call__(self, doc: Doc) -> Doc:
        matches = self.matcher(doc)
        spans = []
        for match_id, start, end in matches:
            spans.append(Span(doc, start, end, label="NIP"))
        doc.ents = list(doc.ents) + spans
        return doc
