import pandas as pd

class Scores:
    
    def __init__(self, df):
        
        self.df   = df
        
    def qualified_athletes(self):
        return self.df[self.df.qualified]
    
    def valid_lifts(self):
        self.df = self.qualified_athletes()
        return self.df[self.df.lift_valid]
    
    def best_lift(self):
        self.df = self.valid_lifts()
        if self.df.empty:
            return
        else:
            return self.df.loc[ self.df['lift'].idxmax() ]
        
        

