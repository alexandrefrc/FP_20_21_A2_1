import pandas as pd

from extract_scores import Scores as sc

#Functions

athletes = pd.read_csv('athletes.csv')

sports_events = pd.read_csv('sports_events.csv')

sports_events['qualified'] = sports_events['qualified'].astype(bool)

sports_events = sports_events.dropna()

sports_events['lift_valid'] = sports_events['lift_valid'].astype(bool)

#sports_events = sc(sports_events)

#print(sports_events.qualified_athletes())

#print(sports_events.valid_lifts())


df_grouped = sports_events.groupby(['year','athlete_id','lift_type'])

results = pd.DataFrame(columns = sports_events.columns)

for key, group in df_grouped:
    group = sc(group)
    results = results.append(group.best_lift())
    
results_grouped = results.groupby(['year','athlete_id'])

final_score_dict = { }
    
for key, group in results_grouped:
    year = group.iloc[0,0]
    athlete_id = group.iloc[0,1]
    cj = float(group.iloc[0,-2])
    snatch = float(group.iloc[1,-2])
    final_score = cj + snatch
    final_score_dict[athlete_id] = [year,snatch,cj,final_score]
    

final_score_df = pd.DataFrame.from_dict(final_score_dict, orient='index',
                                        columns=['Year', 'Snatch',
                                             'Clean and Jerk',
                                             'Final Score'])

final_score_df.index.name = 'Athlete_id'


#print(final_score_df.sort_values(by=['Year','Final Score'], ascending=False))

#print(athletes)


for index, row in final_score_df.iterrows():
    athlete_info = athletes.loc[(athletes['id'] == index) & (athletes['year'] == row['Year'])]
    #final_score_df.loc[final_score_df.index[index & row['Year']], 'Name'] = athlete_info['name']
    #print(final_score_df.loc(index & row['Year']))
    #print(final_score_df.loc[final_score_df.index[index & row['Year']], 'Name'])


#merged = pd.merge(final_score_df, athletes, left_on='Athlete_id', right_on='id')
merged = final_score_df.merge(athletes, how='left', left_on='Athlete_id', right_on='id').drop(columns=['year','height'])

column_names = ['name','nation','bodyweight','birthdate','Year','Snatch','Clean and Jerk','Final Score']

merged = merged.reindex(columns=column_names)

print(merged.iloc[0])



    


