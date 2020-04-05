import unittest
import spacy

from ..matchers import NIPMatcher, BankNumberMatcher, REGONMatcher


class MatchersTestCase(unittest.TestCase):

    def __init__(self,  *args, **kwargs):
        super(MatchersTestCase, self).__init__(*args, **kwargs)

    def test_NIP(self):
        positive_test_strings = [
            'NIP: 1234567890',
            'NIP:1234567890',
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

        negative_test_strings = [
            'NIP: 123 456 789',
            'NIP:123-456-789',
            '12 34 56 78 9 NIP',
            'qwertyuiop',
            'ala ma kota',
        ]

        nlp = spacy.load('en_core_web_sm')
        matcher = NIPMatcher(nlp)
        nlp.add_pipe(matcher, before='ner')

        for doc in list(nlp.pipe(positive_test_strings)):
            self.assertTrue(matcher.label in [e.label_ for e in doc.ents])

        for doc in list(nlp.pipe(negative_test_strings)):
            self.assertFalse(matcher.label in [e.label_ for e in doc.ents])

    def test_bank_account_numbers(self):
        positive_test_strings = [
            'mbank: 20123456789012345678901234',
            'mbank: 20 123456789012345678901234',
            'mbank: 20-123456789012345678901234',
            'mbank: 123412341234123412341234',
            'mbank: 20 1234 1234 1234 1234 1234 1234',
            'mbank: 20-1234-1234-1234-1234-1234-1234PL',
            'mbank: 1234 1234 1234 1234 1234 1234 konto',
            'mbank konto:1234-1234-1234-1234-1234-1234',
        ]

        negative_test_strings = [
            'mbank: to nie jest numer konta',
            'mbank: 12345678901234567890',
        ]

        nlp = spacy.load('en_core_web_sm')
        matcher = BankNumberMatcher(nlp)
        nlp.add_pipe(matcher, before='ner')

        for doc in nlp.pipe(positive_test_strings):
            self.assertTrue(matcher.label in [e.label_ for e in doc.ents])

        for doc in nlp.pipe(negative_test_strings):
            self.assertFalse(matcher.label in [e.label_ for e in doc.ents])

    def test_REGON(self):
        positive_test_strings = [
            'REGON: 123456789',
            'REG.N: 123456789 PL',
            '123456789 REGON'
        ]

        negative_test_strings = [
            'REGON: 123 456 789',
            'REG: 12345678',
            'REG: 123-456-789'
        ]

        nlp = spacy.load('en_core_web_sm')
        matcher = REGONMatcher(nlp)
        nlp.add_pipe(matcher, before='ner')

        for doc in nlp.pipe(positive_test_strings):
            self.assertTrue(matcher.label in [e.label_ for e in doc.ents])

        for doc in nlp.pipe(negative_test_strings):
            self.assertFalse(matcher.label in [e.label_ for e in doc.ents])


if __name__ == '__main__':
    unittest.main()
