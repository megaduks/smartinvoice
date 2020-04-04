import unittest
import spacy

from ..matchers import NIPMatcher, BankNumberMatcher


class MatchersTestCase(unittest.TestCase):

    def __init__(self,  *args, **kwargs):
        super(MatchersTestCase, self).__init__(*args, **kwargs)
        self.nlp = spacy.load('en_core_web_sm')

    def test_NIP(self):
        test_strings = [
            'NIP: 1234567890',
            'NIP: PL 1234567890',
            'NIP: PL1234567890',
            'NIP: 1234567890 PL',
            'NIP: 1234567890PL',
            'NIP: 123 456 78 90',
            'NIP: 123-456-78-90',
            'NIP: PL 123 456 78 90',
            'NIP: PL123-456-78-90',
            'NIP: 123 456 78 90 PL',
            'NIP: 123-456-78-90PL',
            'NIP: 123 45 67 890',
            'NIP: 123-45-67-890',
            'NIP: PL 123 45 67 890',
            'NIP: PL123-45-67-890',
            'NIP: 123-45-67-890 PL',
            'NIP: 123 45 67 890PL',
        ]

        self.nlp.add_pipe(NIPMatcher(self.nlp), before='ner')
        for test_string in test_strings:
            doc = self.nlp(test_string)
            self.assertTrue('NIP' in [e.label_ for e in doc.ents])

    def test_bank_account_numbers(self):
        test_strings = [
            'mbank: 20 123456789012345678901234',
            'mbank: PL 20 1234 1234 1234 1234 1234 1234',
            'mbank: 20-1234-1234-1234-1234-1234-1234PL'
        ]

        self.nlp.add_pipe(BankNumberMatcher(self.nlp), before='ner')
        for test_string in test_strings:
            doc = self.nlp(test_string)
            self.assertTrue('BANK_ACCOUNT_NO' in [e.label_ for e in doc.ents])


if __name__ == '__main__':
    unittest.main()
