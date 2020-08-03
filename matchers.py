from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token

from tokenizers import create_custom_tokenizer


def remove_REGON_token(doc: Doc) -> Doc:
    """Removes matched REGON and REGON: tokens from the matched span"""
    new_ents = []
    for ent in doc.ents:
        if ent.label_ == 'regon':
            if doc[ent.start + 1].is_punct:
                new_ent = Span(doc, ent.start + 2, ent.end, label=ent.label)
            else:
                new_ent = Span(doc, ent.start + 1, ent.end, label=ent.label)
            new_ents.append(new_ent)
        else:
            new_ents.append(ent)
    doc.ents = new_ents
    return doc


class InvoiceMatcher():
    """Generic class which is extended with particular patterns and labels"""

    def __init__(self, nlp, label):
        """Creates a new matcher using a shared vocabulary object and sets the entity label"""
        nlp.tokenizer = create_custom_tokenizer(nlp)
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
        super(NIPMatcher, self).__init__(nlp, label='nip')

        # TODO: add matching expressions for PLxxxxxxxxxx, PLxxx xxx xx xx, etc.
        # TODO: add recognition of KRS numbers (identical format as NIP)

        patterns = [
            # 10 consecutive digits
            [
                {'TEXT': 'PL', 'OP': '?'},
                {'IS_DIGIT': True, 'LENGTH': {"==": 10}},
            ],
            # 3 3 2 2 or 3-3-2-2 pattern
            [
                {'TEXT': 'PL', 'OP': '?'},
                {'IS_DIGIT': True, 'SHAPE': 'ddd'},
                {'IS_PUNCT': True, 'OP': '?'},
                {'IS_DIGIT': True, 'SHAPE': 'ddd'},
                {'IS_PUNCT': True, 'OP': '?'},
                {'IS_DIGIT': True, 'SHAPE': 'dd'},
                {'IS_PUNCT': True, 'OP': '?'},
                {'IS_DIGIT': True, 'SHAPE': 'dd'},
            ],
            # 3 2 2 3 or 3-2-2-3 pattern
            [
                {'TEXT': 'PL', 'OP': '?'},
                {'IS_DIGIT': True, 'SHAPE': 'ddd'},
                {'IS_PUNCT': True, 'OP': '?'},
                {'IS_DIGIT': True, 'SHAPE': 'dd'},
                {'IS_PUNCT': True, 'OP': '?'},
                {'IS_DIGIT': True, 'SHAPE': 'dd'},
                {'IS_PUNCT': True, 'OP': '?'},
                {'IS_DIGIT': True, 'SHAPE': 'ddd'},
            ]
        ]
        self.matcher.add(self.label, None, *patterns)


class BankAccountMatcher(InvoiceMatcher):
    """Extracts matched bank account numbers and adds them as document entities with the label BANK_ACCOUNT_NO"""

    def __init__(self, nlp):
        """Creates a new bank number account matcher using a shared vocabulary object"""
        super(BankAccountMatcher, self).__init__(nlp, label='bank_account_no')

        # TODO: allow for PL to be glued to the first/last digit of the account number
        # TODO: fix the colon as the separator, now "konto:1234 1234 1234 ..." will not be matched

        patterns = [
            # 26 consecutive digits
            [
                {'TEXT': 'PL', 'OP': '?'},
                {'IS_DIGIT': True, 'LENGTH': {"==": 26}},
            ],
            # 24 digit format
            [
                {'TEXT': 'PL', 'OP': '?'},
                {'IS_DIGIT': True, 'LENGTH': {"==": 24}},
            ],
            # 2-24 digit format
            [
                {'TEXT': 'PL', 'OP': '?'},
                {'IS_DIGIT': True, 'LENGTH': {"==": 2}},
                {'TEXT': '-', 'OP': '?'},
                {'IS_DIGIT': True, 'LENGTH': {"==": 24}},
            ],
            # 4 4 4 4 4 4 digit format and 4-4-4-4-4-4 digit format
            [
                {'TEXT': 'PL', 'OP': '?'},
                {'IS_DIGIT': True, 'LENGTH': {"==": 2}, 'OP': '?'},
                {'TEXT': '-', 'OP': '?'},
                {'IS_DIGIT': True, 'LENGTH': {"==": 4}},
                {'TEXT': '-', 'OP': '?'},
                {'IS_DIGIT': True, 'LENGTH': {"==": 4}},
                {'TEXT': '-', 'OP': '?'},
                {'IS_DIGIT': True, 'LENGTH': {"==": 4}},
                {'TEXT': '-', 'OP': '?'},
                {'IS_DIGIT': True, 'LENGTH': {"==": 4}},
                {'TEXT': '-', 'OP': '?'},
                {'IS_DIGIT': True, 'LENGTH': {"==": 4}},
                {'TEXT': '-', 'OP': '?'},
                {'IS_DIGIT': True, 'LENGTH': {"==": 4}},
            ],
        ]
        self.matcher.add(self.label, None, *patterns)


class REGONMatcher(InvoiceMatcher):
    """Extracts matched REGON numbers and adds them as document entities with the label REGON"""

    def __init__(self, nlp):
        """Creates a new REGON matcher using a shared vocabulary object"""
        super(REGONMatcher, self).__init__(nlp, label='regon')
        patterns = [
            [
                {'LOWER': 'regon'},
                {'IS_PUNCT': True, 'OP': '?'},
                {'IS_DIGIT': True, 'LENGTH': {'==': 9}}
            ],
        ]
        self.matcher.add(self.label, None, *patterns)


class InvoiceNumberMatcher(InvoiceMatcher):
    """Extracts the invoice number and adds it as a document entity with the label INVOICE_NUMBER"""

    def __init__(self, nlp):
        """Creates a new REGON matcher using a shared vocabulary object"""
        super(InvoiceNumberMatcher, self).__init__(nlp, label='invoice_no')
        patterns = [
            [
                {'LOWER': 'faktura'},
                {'LOWER': 'vat', 'OP': '?'},
                {'LOWER': 'nr', 'OP': '?'},
                {'IS_PUNCT': True, 'OP': '?'},
                {'IS_DIGIT': True},
                {'TEXT': '/'},
                {'IS_DIGIT': True},
                {'TEXT': '/', 'OP': '?'},
                {'IS_DIGIT': True, 'OP': '?'},
                {'TEXT': '/', 'OP': '?'},
                {'IS_DIGIT': True, 'OP': '?'},
            ],
        ]
        self.matcher.add(self.label, None, *patterns)


class MoneyMatcher(InvoiceMatcher):
    """Extracts matched monetary amounts expressed in Polish zloty and adds them as entites with the label PLN"""

    def __init__(self, nlp):
        """Creates a new money matcher for Polish zloty using a shared vocabulary object"""
        super(MoneyMatcher, self).__init__(nlp, label='money')
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
            [
                {'POS': 'NUM'}, {'ENT_TYPE': "MONEY"}
            ],
        ]
        self.matcher.add(self.label, None, *patterns)


class GrossValueMatcher(InvoiceMatcher):
    """Extracts elements which resemble invoice gross amount"""

    def __init__(self, nlp):
        """Creates a new invoice gross value matcher"""
        super(GrossValueMatcher, self).__init__(nlp, label='gross_val')
        patterns = [
            [
                {'LOWER': 'brutto'},
                {'ENT_TYPE': 'MONEY'}
            ],
            [
                {'LOWER': {"IN": ["brutto", "brutta"]}},
                {'IS_DIGIT': True},
                {'TEXT': {"IN": ['-', '.']}},
                {'IS_DIGIT': True, 'LENGTH': {"=": 2}},
            ],
            [
                {'LOWER': {"IN": ["brutto", "brutta"]}},
                {'IS_DIGIT': True},
            ],
        ]
        self.matcher.add(self.label, None, *patterns)


class DateMatcher(InvoiceMatcher):
    """Extracts everything that looks like a date from text"""

    def __init__(self, nlp):
        """Creates a new date matcher"""
        super(DateMatcher, self).__init__(nlp, label='data')
        patterns = [
            [
                {'IS_DIGIT': True, 'LENGTH': {"==": 4}},
                {'TEXT': {"IN": ['-', '.']}},
                {'IS_DIGIT': True, 'LENGTH': {"<=": 2}},
                {'TEXT': {"IN": ['-', '.']}},
                {'IS_DIGIT': True, 'LENGTH': {"<=": 2}},

            ],
            [
                {'IS_DIGIT': True, 'LENGTH': {"<=": 2}},
                {'TEXT': {"IN": ['-', '.']}},
                {'IS_DIGIT': True, 'LENGTH': {"<=": 2}},
                {'TEXT': {"IN": ['-', '.']}},
                {'IS_DIGIT': True, 'LENGTH': {"==": 4}},

            ],
        ]
        self.matcher.add(self.label, None, *patterns)
