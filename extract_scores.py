import pandas as pd 

class ProcessDf:
    
    def __init__(self, df):
        self.df   = df
    
    def dates(self, column):
        self.df[column] = pd.to_datetime(self.df[column], format = '%d/%m/%Y')
        return self.df


class ProcessSportsEvents(ProcessDf):

    def __init__(self, df):
        super().__init__(df)

    def process_data(self):
        self.df = self.df.dropna()
        self.df['qualified'] = self.df['qualified'].astype(bool)
        self.df['lift_valid'] = self.df['lift_valid'].astype(bool)
        return self.df
    
    def valid_entries(self):
        self.df = self.df[self.df['qualified']]
        self.df = self.df[self.df['lift_valid']]
        return self.df
    
    def best_lift(self):
        self.df = self.valid_entries()
        if self.df.empty:
            return
        else:
            return self.df.loc[self.df['lift'].idxmax()]

