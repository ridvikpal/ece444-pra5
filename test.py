import unittest
from application import _predict_text

class TestPrediction(unittest.TestCase):
    def test_fake_news_1(self):
        test_message = '''
            The world is ending!!!!
        '''
        result = _predict_text(test_message)
        self.assertEqual(result, 'FAKE')
    
    def test_fake_news_2(self):
        test_message = '''
            The Blue Jays won the world series!!!!
        '''
        result = _predict_text(test_message)
        self.assertEqual(result, 'FAKE')

    def test_real_news_1(self):
        test_message = '''
            Mark Carney is officially the prime minister of Canada.
        '''
        result = _predict_text(test_message)
        self.assertEqual(result, 'REAL')

    def test_real_news_2(self):
        test_message = '''
            McLaren is on track to win the Formula 1 championship this year.
        '''
        result = _predict_text(test_message)
        self.assertEqual(result, 'REAL')

        