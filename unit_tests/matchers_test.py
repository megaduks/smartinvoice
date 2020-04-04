import unittest
import spacy

from ..matchers import NIPMatcher


class MatchersTestCase(unittest.TestCase):

    def __init__(self,  *args, **kwargs):
        super(MatchersTestCase, self).__init__(*args, **kwargs)
        self.nlp = spacy.load('en_core_web_sm')
        self.nip_matcher = NIPMatcher(self.nlp)
        self.nlp.add_pipe(self.nip_matcher, before='ner')

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

        for test_string in test_strings:
            doc = self.nlp(test_string)
            self.assertTrue('NIP' in [e.label_ for e in doc.ents])


if __name__ == '__main__':
    unittest.main()
