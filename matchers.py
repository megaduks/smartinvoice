from spacy.matcher import Matcher
from spacy.tokens import Doc, Span


class InvoiceMatcher():
    """Generic class which is extended with particular patterns and labels"""

    def __init__(self, nlp, label):
        """Creates a new matcher using a shared vocabulary object and sets the entity label"""
        self.matcher = Matcher(nlp.vocab)
        self.label = label

    def __call__(self, doc: Doc) -> Doc:
        matches = self.matcher(doc)
        spans = []
        seen_tokens = set()
        entities = doc.ents
        for match_id, start, end in matches:
            # spaCy does not allow overlapping entity matches so if two or more patterns can be matched
            # we need to select the longest matching
            if start not in seen_tokens and (end - 1) not in seen_tokens:
                spans.append(Span(doc, start, end, label=self.label))
                entities = [
                    e for e in entities if not (e.start < end and e.end > start)
                ]
                seen_tokens.update(range(start, end))
        doc.ents = tuple(entities) + tuple(spans)
        return doc


class NIPMatcher(InvoiceMatcher):
    """Extracts matched NIP instances and adds them as document entities with the label NIP"""

    def __init__(self, nlp):
        """Creates a new NIP matcher using a shared vocabulary object"""
        super(NIPMatcher, self).__init__(nlp, label='NIP')
        patterns = [
            # 10 consecutive digits
            [
                {'TEXT': {'REGEX': '\d{10}'}}
            ],
            # 3 3 2 2 digit format
            [
                {'TEXT': {'REGEX': '\d{3}'}},
                {'TEXT': {'REGEX': '\d{3}'}},
                {'TEXT': {'REGEX': '\d{2}'}},
                {'TEXT': {'REGEX': '\d{2}'}},
            ],
            # 3-3-2-2 digit format
            [
                {'TEXT': {'REGEX': '\d{3}'}},
                {'TEXT': '-'},
                {'TEXT': {'REGEX': '\d{3}'}},
                {'TEXT': '-'},
                {'TEXT': {'REGEX': '\d{2}'}},
                {'TEXT': '-'},
                {'TEXT': {'REGEX': '\d{2}'}},
            ],
            # 3 2 2 3 digit format
            [
                {'TEXT': {'REGEX': '\d{3}'}},
                {'TEXT': {'REGEX': '\d{2}'}},
                {'TEXT': {'REGEX': '\d{2}'}},
                {'TEXT': {'REGEX': '\d{3}'}},
            ],
            # 3-2-2-3 digit format
            [
                {'TEXT': {'REGEX': '\d{3}'}},
                {'TEXT': '-'},
                {'TEXT': {'REGEX': '\d{2}'}},
                {'TEXT': '-'},
                {'TEXT': {'REGEX': '\d{2}'}},
                {'TEXT': '-'},
                {'TEXT': {'REGEX': '\d{3}'}},
            ],
        ]
        self.matcher.add(self.label, None, *patterns)


class BankNumberMatcher(InvoiceMatcher):
    """Extracts matched bank account numbers and adds them as document entities with the label BANK_ACCOUNT_NO"""

    def __init__(self, nlp):
        """Creates a new bank number account matcher using a shared vocabulary object"""
        super(BankNumberMatcher, self).__init__(nlp, label='BANK_ACCOUNT_NO')
        patterns = [
            # 26 consecutive digits
            [
                {'TEXT': {'REGEX': '\d{26}'}}
            ],
            # 24 digit format
            [
                {'TEXT': {'REGEX': '\d{24}'}},
            ],
            # 2-24 digit format
            [
                {'TEXT': {'REGEX': '\d{2}'}},
                {'TEXT': '-', 'OP': '?'},
                {'TEXT': {'REGEX': '\d{24}'}},
            ],
            # 4 4 4 4 4 4 digit format
            [
                {'TEXT': {'REGEX': '\d{4}'}},
                {'TEXT': {'REGEX': '\d{4}'}},
                {'TEXT': {'REGEX': '\d{4}'}},
                {'TEXT': {'REGEX': '\d{4}'}},
                {'TEXT': {'REGEX': '\d{4}'}},
                {'TEXT': {'REGEX': '\d{4}'}},
            ],
            # 4-4-4-4-4-4 digit format
            [
                {'TEXT': {'REGEX': '\d{4}'}},
                {'TEXT': '-'},
                {'TEXT': {'REGEX': '\d{4}'}},
                {'TEXT': '-'},
                {'TEXT': {'REGEX': '\d{4}'}},
                {'TEXT': '-'},
                {'TEXT': {'REGEX': '\d{4}'}},
                {'TEXT': '-'},
                {'TEXT': {'REGEX': '\d{4}'}},
                {'TEXT': '-'},
                {'TEXT': {'REGEX': '\d{4}'}},
            ],

        ]
        self.matcher.add(self.label, None, *patterns)


class REGONMatcher(InvoiceMatcher):
    """Extracts matched REGON numbers and adds them as document entities with the label REGON"""

    def __init__(self, nlp):
        """Creates a new REGON matcher using a shared vocabulary object"""
        super(REGONMatcher, self).__init__(nlp, label='REGON')
        patterns = [
            [
                {'TEXT': {'REGEX': '\d{9}'}}
            ],
        ]
        self.matcher.add(self.label, None, *patterns)


class MoneyMatcher(InvoiceMatcher):
    """Extracts matched monetary amounts expressed in Polish zloty and adds them as entites with the label PLN"""

    def __init__(self, nlp):
        """Creates a new money matcher for Polish zloty usign a shared vocabulary object"""
        super(MoneyMatcher, self).__init__(nlp, label='PLN')
        patterns = [
            [
                {'POS': 'NUM'}, {'LOWER': 'zł'}
            ],
            [
                {'POS': 'NUM'}, {'LOWER': 'zl'}
            ],
            [
                {'POS': 'NUM'}, {'LOWER': 'pln'}
            ],
        ]
        self.matcher.add(self.label, None, *patterns)