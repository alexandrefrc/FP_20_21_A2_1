import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import extract_scores as es


def cal_age(df):
    time_diff = df['date'] - df['birthdate']
    return time_diff // np.timedelta64(1, 'Y')  

def extract_year(df, n):
    return df.insert(n, 'year', pd.DatetimeIndex(df['date']).year)

def calc_ratio(lift, bodyweight):
    return lift / bodyweight

def calc_diff(df, info, column='Year'):
    first = df[df[column] == info]['Score'].max()
    last = df[df[column] == info]['Score'].min()
    third_info = df[df[column] == info].sort_values('Score', ascending=False)
    third = third_info['Score'].iloc[2]
    diff_frst_lst = first - last
    diff_frst_thrd = first - third
    return diff_frst_lst, diff_frst_thrd

athletes = pd.read_csv('athletes.csv')
sports_events = pd.read_csv('sports_events.csv')

sports_events = es.ProcessSportsEvents(sports_events)
athletes = es.ProcessDf(athletes)

sports_events.process_data()
sports_events = sports_events.dates('date')
athletes = athletes.dates('birthdate')

extract_year(sports_events, 2)

processed_info_df = athletes.merge(
    sports_events, how='right', right_on='athlete_id', left_on='id')

processed_info_df['age'] = processed_info_df.apply(cal_age, axis=1)

processed_info_df = processed_info_df.drop(
    columns=['birthdate', 'date', 'athlete_id', 'height'])

processed_info_df = processed_info_df[['year', 'id', 'name', 'age', 'nation',
                                       'bodyweight', 'group', 'qualified', 'lift_type', 'lift', 'lift_valid']]

best_lifts = pd.DataFrame(columns=processed_info_df.columns)

for key, group in processed_info_df.groupby(['year', 'id', 'lift_type']):
    group = es.ProcessSportsEvents(group)
    best_lifts = best_lifts.append(group.best_lift())

complete_info_dict = {}

for key, result in best_lifts.groupby(['year', 'id']):
    year = result.iloc[0, 0]
    athlete_id = result.iloc[0, 1]
    name = result.iloc[0, 2]
    age = result.iloc[0, 3]
    nation = result.iloc[0, 4]
    group = result.iloc[0, 6]
    bodyweight = result.iloc[0, 5]
    cj = float(result.iloc[0, -2])
    snatch = float(result.iloc[1, -2])
    final_score = cj + snatch
    complete_info_dict[(athlete_id, year)] = [name, age, nation, group, bodyweight, snatch, cj, final_score]

complete_info_df = pd.DataFrame.from_dict(complete_info_dict, orient='index',
                                        columns=['Name', 'Age', 'Nation', 'Group', 'Bodyweight', 'Snatch', 'Clean and Jerk', 'Score'])

complete_info_df.index = pd.MultiIndex.from_tuples(complete_info_df.index)
complete_info_df.index.names = ['Athlete ID', 'Year']
complete_info_df = complete_info_df.reset_index()

complete_info_df = complete_info_df.sort_values(by='Score', ascending=False)

# Tabela razões entre peso levantado e peso do atleta
# Complementar com médias etc.

position_by_ratio_df = pd.DataFrame()

for index, year_results in complete_info_df.groupby('Year'):
    year_results = year_results.copy()
    year_results['Snatch/Weight'] = year_results.apply(lambda x: calc_ratio(x['Snatch'], x['Bodyweight']), axis = 1)
    year_results['Clean&Jerk/Weight'] = year_results.apply(lambda x: calc_ratio(x['Clean and Jerk'], x['Bodyweight']), axis = 1)
    year_results['Average_Ratio'] = (year_results['Snatch/Weight']+year_results['Clean&Jerk/Weight'])/2
    year_results = year_results.sort_values(
        by='Average_Ratio', ascending=False)
    year_results['Position'] = year_results['Score'].rank(ascending=False, method='first')
    year_results['Position_by_ratio'] = year_results['Average_Ratio'].rank(ascending=False, method='first')
    year_results = year_results.drop(columns=['Nation'])
    position_by_ratio_df = position_by_ratio_df.append(year_results.head())
    
pd.set_option('display.max_columns', 12)

print(position_by_ratio_df)

# Plot médias atletas e médias medalhistas
# Rever sintax e melhorar aparência

ax = complete_info_df.groupby('Year').Score.mean().plot(xticks=complete_info_df['Year'].unique(), style='o-')

medalists = complete_info_df.groupby('Year').head(n=3)
medalists.groupby('Year')['Score'].mean().plot(
    xticks=complete_info_df['Year'].unique(), ax=ax, style='o-')
ax.legend(['All Athletes', 'Medalists'], loc = 'upper center')
ax.set_ylabel('Score (kg)')
plt.title('Average Score per Year')
plt.ylim(215, 285)
plt.margins(x = 0.1)

xs_athletes = np.sort(complete_info_df['Year'].unique())
ys_athletes = complete_info_df.groupby('Year')['Score'].mean()
xs_medalists = np.sort(medalists['Year'].unique())
ys_medalists = medalists.groupby('Year')['Score'].mean()


for pair in [(xs_athletes, ys_athletes),(xs_medalists, ys_medalists)]:
    for x, y in zip(pair[0], pair[1]):
        label = "{:.2f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     bbox=dict(boxstyle="round", fc="0.9", ec="gray"),
                     ha='center')  # horizontal alignment can be left, right or center

plt.show()

# Gráficos de barras com diff entre 1o e ultimo classificado e primeiro e 3 classificado.

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
    year_df = sports_events[sports_events['year'] == year]
    total_lifts = len(year_df['athlete_id'].unique())
    snatch_lifts = len(year_df[year_df['lift_type']
                               == 'snatch']['athlete_id'].unique())
    cj_lifts = len(year_df[year_df['lift_type'] ==
                           'cleanjerk']['athlete_id'].unique())
    failed_lifts = year_df['lift_valid'].value_counts()[False]
    failed_snatch = year_df[year_df['lift_type'] ==
                            'snatch']['lift_valid'].value_counts()[False]
    failed_cj = year_df[year_df['lift_type'] ==
                        'cleanjerk']['lift_valid'].value_counts()[False]
    winner_id = int(position_by_ratio_df[(position_by_ratio_df['Year'] == year) & (
        position_by_ratio_df['Position'] == 1)]['Athlete ID'])
    failed_winner_lifts = len(
        year_df[(year_df['athlete_id'] == winner_id) & (year_df['lift_valid'] == False)])
    failed_winner_snatch = len(year_df[(year_df['lift_type'] == 'snatch') & (
        year_df['athlete_id'] == winner_id) & (year_df['lift_valid'] == False)])
    failed_winner_cj = len(year_df[(year_df['lift_type'] == 'cleanjerk') & (
        year_df['athlete_id'] == winner_id) & (year_df['lift_valid'] == False)])
    tries[year] = [failed_snatch/snatch_lifts, failed_cj/cj_lifts, failed_lifts /
                   total_lifts, failed_winner_snatch, failed_winner_cj, failed_winner_lifts]

failed_lifts_df = pd.DataFrame.from_dict(tries, orient='index',
                                         columns=['Failed Snatch', 'Failed Clean and Jerk', 'Overall Failed', 'Failed Snatch', 'Failed Clean and Jerk', 'Overall Failed'])

failed_lifts_df.columns = pd.MultiIndex.from_tuples(list(zip(
    ['All Athletes', 'All Athletes', 'All Athletes', 'Winner', 'Winner', 'Winner'], failed_lifts_df.columns)))

print(failed_lifts_df)

# fazer prints com médias de cada coluna.

# Gráfico Média de Score vs. Idade

score_mean = []
score_std = []

for age in complete_info_df['Age'].unique():
    age_df = complete_info_df[complete_info_df['Age'] == age]
    score_mean.append(age_df['Score'].mean())
    score_std.append(age_df['Score'].std())

plt.bar(
    complete_info_df['Age'].unique(), score_mean, width=0.6, yerr = score_std, capsize = 4, error_kw = dict(ecolor='chocolate',elinewidth=0.5), color='orange')
plt.xticks(np.arange(complete_info_df['Age'].min(), complete_info_df['Age'].max()+1))
plt.ylim(100, 280)
plt.ylabel('Scores')
complete_info_df['Age'].value_counts().sort_index().plot(secondary_y=True, color = 'blue')
plt.ylabel('Number of Athletes')
plt.xlabel('Age')
plt.title('Scores Mean per Age')
