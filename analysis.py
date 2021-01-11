import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from extract_scores import Scores as sc


athletes = pd.read_csv('athletes.csv')
athletes['birthdate'] = athletes['birthdate'].astype('datetime64')
sports_events = pd.read_csv('sports_events.csv')
sports_events['date'] = sports_events['date'].astype('datetime64')



sports_events['qualified'] = sports_events['qualified'].astype(bool)
# INCLUIR DROP NA NA CLASSE SCORES
sports_events = sports_events.dropna()
sports_events['lift_valid'] = sports_events['lift_valid'].astype(bool)

sports_events['year'] = pd.DatetimeIndex(sports_events['date']).year

complete_info_df = sports_events.merge(athletes, how='left', left_on='athlete_id', right_on='id')

df_grouped = sports_events.groupby(['year', 'athlete_id', 'lift_type'])

results = pd.DataFrame(columns=sports_events.columns)

for key, group in df_grouped:
    group = sc(group)
    results = results.append(group.best_lift())

results_grouped = results.groupby(['year', 'athlete_id'])

final_score_dict = {}

for key, group in results_grouped:
    year = group.iloc[0, -1]
    athlete_id = group.iloc[0, 1]
    bodyweight = group.iloc[0, 2]
    cj = float(group.iloc[0, -2])
    snatch = float(group.iloc[1, -2])
    final_score = cj + snatch
    final_score_dict[(athlete_id, year)] = [bodyweight, snatch, cj, final_score]

final_score_df = pd.DataFrame.from_dict(final_score_dict, orient='index',
                                        columns=['Bodyweight', 'Snatch',
                                                 'Clean and Jerk',
                                                 'Final Score'])
final_score_df.index = pd.MultiIndex.from_tuples(final_score_df.index)
final_score_df.index.names = ['Athlete_id', 'Year']

complete_info_df.columns = ['Athlete_ID', 'Year', 'Name', 'Nation',
                            'Bodyweight', 'Birthdate', 'Snatch', 'Clean&Jerk', 'Score']
complete_info_df = complete_info_df.sort_values(by='Score', ascending=False)

# Tabela razões entre peso levantado e peso do atleta
# Complementar com médias etc.

position_by_ratio_df = pd.DataFrame()

for index, year_results in complete_info_df.groupby('Year'):
    year_results = year_results.sort_values(by='Score')
    year_results['Snatch/Weight'] = year_results['Snatch'] / \
        year_results['Bodyweight']
    year_results['Clean&Jerk/Weight'] = year_results['Clean&Jerk'] / \
        year_results['Bodyweight']
    year_results['Average_Ratio'] = (
        year_results['Snatch/Weight']+year_results['Clean&Jerk/Weight'])/2
    year_results = year_results.sort_values(
        by='Average_Ratio', ascending=False)
    year_results['Position'] = year_results['Score'].rank(
        ascending=False, method='first')
    year_results['Position_by_ratio'] = year_results['Average_Ratio'].rank(
        ascending=False, method='first')
    year_results = year_results.drop(columns=['Nation', 'Birthdate'])
    position_by_ratio_df = position_by_ratio_df.append(year_results.head())
pd.set_option('display.max_columns', 12)
#print(position_by_ratio_df)

# Plot médias atletas e médias medalhistas
# Rever sintax e melhorar aparência

ax = complete_info_df.groupby('Year').Score.mean().plot(
    xticks=complete_info_df['Year'].unique())

medalists = complete_info_df.groupby('Year').head(n=3)
medalists.groupby('Year').Score.mean().plot(
    xticks=complete_info_df['Year'].unique(), ax=ax)
ax.legend(['All Athletes', 'Medalists'])
ax.set_ylabel('Score (kg)')
xs = complete_info_df['Year'].unique()
ys = complete_info_df.groupby('Year')['Score'].mean()

plt.plot(xs, ys, 'co')

for x, y in zip(xs, ys):

    label = "{:.2f}".format(y)

    plt.annotate(label,  # this is the text
                 (x, y),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, 10),  # distance from text to points (x,y)
                 ha='center')  # horizontal alignment can be left, right or center
plt.show()

# Gráficos de barras com diff entre 1o e ultimo classificado e primeiro e 3 classificado.

def calc_diff(df, info, column='Year'):
    first = df[df[column] == info]['Score'].max()
    last = df[df[column] == info]['Score'].min()
    third_info = df[df[column] == info].sort_values('Score', ascending=False)
    third = third_info['Score'].iloc[2]
    diff_frst_lst = first - last
    diff_frst_thrd = first - third
    return diff_frst_lst, diff_frst_thrd

first_last = []
first_third = []
for year in complete_info_df['Year'].unique():
    max_range, podium_range = calc_diff(complete_info_df, year)
    first_last.append(max_range)
    first_third.append(podium_range)

first_third_range = plt.bar(
    complete_info_df['Year'].unique(), first_third, width=1.8, color='gold')
first_last_range = plt.bar(
    complete_info_df['Year'].unique(), first_last, bottom=first_third, width=1.8)
plt.xticks(complete_info_df['Year'].unique(),
           complete_info_df['Year'].unique())
plt.legend((first_third_range[0], first_last_range[0]),
           ('Medalists', 'All Athletes'))
plt.ylim(0, 180)
plt.ylabel('Scores')
plt.xlabel('Year')
plt.title('Range of Scores')
plt.show()

# Tabela com tentativas falhadas

tries = {}

for year in sports_events['year'].unique():
    year_df = sports_events[sports_events['year']==year]
    total_lifts = len(year_df['athlete_id'].unique())
    snatch_lifts = len(year_df[year_df['lift_type']=='snatch']['athlete_id'].unique())
    cj_lifts = len(year_df[year_df['lift_type']=='cleanjerk']['athlete_id'].unique())
    failed_lifts = year_df['lift_valid'].value_counts()[False]
    failed_snatch = year_df[year_df['lift_type']=='snatch']['lift_valid'].value_counts()[False]
    failed_cj = year_df[year_df['lift_type']=='cleanjerk']['lift_valid'].value_counts()[False]
    winner_id = int(position_by_ratio_df[(position_by_ratio_df['Year']==year) & (position_by_ratio_df['Position']==1)]['Athlete_ID'])
    failed_winner_lifts = len(year_df[(year_df['athlete_id']==winner_id) & (year_df['lift_valid']==False)])
    failed_winner_snatch = len(year_df[(year_df['lift_type']=='snatch') & (year_df['athlete_id']==winner_id) & (year_df['lift_valid']==False)])
    failed_winner_cj = len(year_df[(year_df['lift_type']=='cleanjerk') & (year_df['athlete_id']==winner_id) & (year_df['lift_valid']==False)])
    
    tries[year] = [failed_snatch/snatch_lifts, failed_cj/cj_lifts, failed_lifts/total_lifts, failed_winner_snatch, failed_winner_cj, failed_winner_lifts]
    
failed_lifts_df = pd.DataFrame.from_dict(tries, orient='index',
                                        columns=['failed_snatch_avg', 'failed_cleanjerk_avg', 'failed_avg', 'failed_winner_snatch', 'failed_winner_cleanjerk', 'failed_winner'])

failed_lifts_df.columns = pd.MultiIndex.from_tuples(list(zip(['All Athletes', 'All Athletes', 'All Athletes', 'Winner', 'Winner', 'Winner'], failed_lifts_df.columns)))

# fazer prints com médias de cada coluna. 

# 










