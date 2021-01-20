import unittest
import pandas as pd
from analysis import calc_diff, calc_ratio

class TestCalcDiff(unittest.TestCase):
    
    def test_calc_diff_one_year_diff_scores(self):
        df_dict = {'Year':[2000,2000,2000,2000], 'Score':[40,30,20,10], 'Bodyweight':[70, 71, 72, 70]}
        df = pd.DataFrame(df_dict)
        self.assertEqual(calc_diff(df,2000),(30,20))
        
    def test_calc_diff_one_year_same_scores(self):
        df_dict = {'Year':[2000,2000,2000,2000], 'Score':[40,40,40,40], 'Bodyweight':[70, 71, 72, 70]}
        df = pd.DataFrame(df_dict)
        self.assertEqual(calc_diff(df,2000),(0,0))
    
    def test_calc_diff_many_years_diff_scores(self):
        df_dict = {'Year':[2000,2000,2000,2000,2008,2008], 'Score':[40,30,30,20,35,2], 'Bodyweight':[70, 71, 72, 70, 72, 74]}
        df = pd.DataFrame(df_dict)
        self.assertEqual(calc_diff(df,2000),(20,10))
        
    def test_calc_diff_year_not_present(self):
        df_dict = {'Year':[2000,2000,2000,2000,2008,2008], 'Score':[40,30,30,20,35,2], 'Bodyweight':[70, 71, 72, 70, 72, 74]}
        df = pd.DataFrame(df_dict)
        self.assertRaises(IndexError,calc_diff,df,2004)
        
    def test_calc_same_scores_different_unsorted_bodyweight(self):
        df_dict = {'Year':[2000,2000,2000,2000], 'Score':[40,50,40,10], 'Bodyweight':[70, 74, 72, 70]}
        df = pd.DataFrame(df_dict)
        self.assertEqual(calc_diff(df,2000),(40,10))
        

        
class TestCalcRatio(unittest.TestCase):
    
    def test_calc_ratio_both_positive_numbers(self):
        self.assertEqual(calc_ratio(4,2),2)
    
    def test_calc_ratio_equal_numbers(self):
        self.assertEqual(calc_ratio(4,4),1)
        
    def test_calc_ratio_numerator_zero(self):
        self.assertEqual(calc_ratio(0,1),0)
        
    def test_calc_ratio_denominator_zero(self):
        self.assertRaises(ZeroDivisionError,calc_ratio,2,0)
        
    def test_calc_ratio_numerator_negative(self):
        self.assertEquals(calc_ratio(-1,1),'Division using a negative number')
    
    def test_calc_ratio_denominator_negative(self):
        self.assertEquals(calc_ratio(1,-1),'Division using a negative number')
    
        
if __name__ == '__main__':
    unittest.main()


