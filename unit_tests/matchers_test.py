import unittest
import spacy

from spacy.lang.pl import Polish

from ..matchers import match_NIP


class MatchersTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nlp = Polish()

    def test_NIP(self):
        _tests = [
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

        for test in _tests:
            _doc = self.nlp(test)
            ent_labels = [
                self.nlp.vocab.strings[ent_id]
                for ent_id, start, end
                in match_NIP()(_doc)
            ]
            self.assertTrue('NIP' in ent_labels)


if __name__ == '__main__':
    unittest.main()
