import unittest
import spacy

from tokenizers import create_custom_tokenizer


class TokenizersTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TokenizersTestCase, self).__init__(*args, **kwargs)

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
        nlp = spacy.load('pl_model')
        nlp.tokenizer = create_custom_tokenizer(nlp)

        for _input, _output in test_strings:
            self.assertCountEqual(map(str, nlp(_input)), _output)


if __name__ == '__main__':
    unittest.main()
