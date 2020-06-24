import unittest
import spacy

import tokenizers


class TokenizersTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TokenizersTestCase, self).__init__(*args, **kwargs)
        self.nlp = spacy.blank('pl')

    def test_tokenization(self):
        test_strings = [
            (
                'FAKTURA nr 5121/H0525', ['FAKTURA', 'nr', '5121', '/', 'H0525']
            ),
            (
                'Faktura Vat nr: 1965/2019/08/TCFK',
                ['Faktura', 'Vat', 'nr', ':', '1965', '/', '2019', '/', '08', '/', 'TCFK']
            ),
            (
                '2020-01-01', ['2020', '-', '01', '-', '01']
            ),
            (
                '2020.01.01', ['2020', '.', '01', '.', '01']
            ),
        ]
        self.nlp.tokenizer = tokenizers.create_custom_tokenizer(self.nlp)

        for _input, _output in test_strings:
            self.assertCountEqual(map(str, self.nlp(_input)), _output)

    def test_sanitization(self):
        _input = self.nlp('Ala! | ma{kota}')
        _output = self.nlp('Ala ma kota')
        self.assertEqual(tokenizers.sanitizer(_input).text, _output.text)


if __name__ == '__main__':
    unittest.main()
