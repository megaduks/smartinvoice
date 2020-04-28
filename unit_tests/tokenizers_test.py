import unittest
import spacy

from tokenizers import create_custom_tokenizer

class TokenizersTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TokenizersTestCase, self).__init__(*args, **kwargs)

    def test_tokenize_invoice(self):
        test_strings = [
            (
                'FAKTURA nr 5121/H0525', ['FAKTURA', 'nr', '5121', '/', 'H0525']
            ),
            (
                'Faktura Vat nr: 1965/2019/08/TCFK', ['Faktura', 'Vat', 'nr', ':', '1965', '/', '2019', '/', '08', '/', 'TCFK']
            )
        ]
        nlp = spacy.load('en_core_web_sm')
        nlp.tokenizer = create_custom_tokenizer(nlp)

        for input, output in test_strings:
            self.assertCountEqual(map(str, nlp(input)), output)


if __name__ == '__main__':
    unittest.main()
