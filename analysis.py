import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import extract_scores as es
import dataframe_image as dfi
import os

def extract_year(df, n, column):
    return df.insert(n, 'year', pd.DatetimeIndex(df[column]).year)


def calc_age(df):
    time_diff = df['date'] - df['birthdate']
    return time_diff // np.timedelta64(1, 'Y')


def calc_ratio(lift, bodyweight):
    if lift < 0 or bodyweight < 0:
        return 'Division using a negative number'
    else:
        return lift / bodyweight


def calc_diff(df, info, column='Year'):
    first = df[df[column] == info]['Score'].max()
    last = df[df[column] == info]['Score'].min()
    third_info = df[df[column] == info].sort_values(['Score', 'Bodyweight'], ascending=[False, True])
    third = third_info['Score'].iloc[2]
    diff_frst_lst = first - last
    diff_frst_thrd = first - third
    return diff_frst_lst, diff_frst_thrd


image_counter = 0
def directoryValidator():
    if os.path.isdir("./images/") == False:
        os.makedirs(os.path.dirname("./images/"))
    else:
        pass


def saveImages(category, graph_name, variable): 
    global image_counter
    directoryValidator() 
    image_counter += 1
    if (category) == "table":
        (graph_name).export((variable), "./images/{}_{}.png".format(str(category),str(image_counter)))
    else:
        print("No valid image Category found.")


athletes = pd.read_csv('athletes.csv')
sports_events = pd.read_csv('sports_events.csv')

sports_events = es.ProcessSportsEvents(sports_events)
athletes = es.ProcessDf(athletes)

sports_events.process_data()
sports_events = sports_events.dates('date')
athletes = athletes.dates('birthdate')
extract_year(sports_events, 2, 'date')


processed_info_df = athletes.merge(
    sports_events, how='right', right_on='athlete_id', left_on='id')

processed_info_df['age'] = processed_info_df.apply(calc_age, axis=1)

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
    complete_info_dict[(athlete_id, year)] = [name, age,
                                              nation, group, bodyweight, snatch, cj, final_score]

complete_info_df = pd.DataFrame.from_dict(complete_info_dict, orient='index',
                                          columns=['Name', 'Age', 'Nation', 'Group', 'Bodyweight', 'Snatch', 'Clean and Jerk', 'Score'])

complete_info_df.index = pd.MultiIndex.from_tuples(complete_info_df.index)
complete_info_df.index.names = ['Athlete ID', 'Year']
complete_info_df = complete_info_df.reset_index()

complete_info_df = complete_info_df.sort_values(
    by=['Year', 'Score', 'Bodyweight'], ascending=[False, False, True])

# Highest and lowest scores ever
print('\nHigher and lowest scores ever:')

highest_score_ever = complete_info_df['Score'].max()
best_athlete = complete_info_df[complete_info_df['Score']
                                == highest_score_ever]['Name'].values
highest_year = complete_info_df[complete_info_df['Score']
                                == highest_score_ever]['Year'].values
highest_year = ', '.join(str(year) for year in highest_year)
print('The highest score was {}, got by {} on the {} edition.'.format(
    highest_score_ever, ', '.join(best_athlete), highest_year))

lowest_score_ever = complete_info_df['Score'].min()
worst_athlete = complete_info_df[complete_info_df['Score']
                                 == lowest_score_ever]['Name'].values
lowest_year = complete_info_df[complete_info_df['Score']
                               == lowest_score_ever]['Year'].values
lowest_year = ', '.join(str(year) for year in lowest_year)
print('The lowest score was {}, got by {} on the {} edition.'.format(
    lowest_score_ever, ', '.join(worst_athlete), lowest_year))

rangescor = highest_score_ever-lowest_score_ever
print('The range of all-time scores is {}.\n'.format(rangescor))


# First and last athlete of each year
print('\nFirst and last place in the various editions:')
for year in complete_info_df['Year'].unique():
    first_place = complete_info_df[complete_info_df['Year'] == year].head(n=1)
    last_place = complete_info_df[complete_info_df['Year'] == year].tail(n=1)
    print("{} won the {} Olympics with a total score of {}.".format(
        first_place['Name'].values[0], year, first_place['Score'].values[0]))
    print("{} became last in that year's Olympics with a total score of {}.".format(
        last_place['Name'].values[0], last_place['Score'].values[0]))
    
    
# Athletes with more that won a medal in more than one edition
medalists = complete_info_df.groupby('Year').head(n=3)

all_medalists = {}
for year in medalists['Year'].unique():
    year_medalists = medalists[medalists['Year'] == year]
    for i in range(len(year_medalists)):
        if year_medalists.iloc[i, 2] in all_medalists:
            all_medalists[year_medalists.iloc[i, 2]
                          ] = all_medalists[year_medalists.iloc[i, 2]] + 1
        else:
            all_medalists[year_medalists.iloc[i, 2]] = 1

n_multiple_medal = 0

for n_medals in all_medalists.values():
    if n_medals > 1:
        n_multiple_medal += 1
print('\n\nAthletes that won more than one medal:')
if n_multiple_medal > 1:
    print('There are {} athletes that won more that one medal:'.format(
        n_multiple_medal))
elif n_multiple_medal == 1:
    print('There is one athlete that won more that one medal:')
else:
    print('There never was an athlete that won more than one medal')

for medalist in all_medalists:
    if all_medalists[medalist] > 1:
        years = ', '.join(str(year) for year in list(
            medalists[medalists['Name'] == medalist]['Year']))
        print('{} won {} medals in {}.'.format(
            medalist, all_medalists[medalist], years))


# Average total scores and lifts scores
print('\n\nAverage score in all editions: {}'.format(
    round(complete_info_df['Score'].mean(), 3)))
print('Average snatch lift in all editions: {}'.format(
    round(complete_info_df['Snatch'].mean(), 3)))
print('Average clean and jerk lift in all editions: {}\n'.format(
    round(complete_info_df['Clean and Jerk'].mean(), 3)))


# Year with more disqualified athletes
n_disqualified = {}
for key, group in processed_info_df.groupby('year'):
    n = 0
    for athlete in group['id'].unique():
        if False in group[group['id'] == athlete]['qualified'].unique():
            n += 1
    n_disqualified[group['year'].values[0]] = n

n_disqualified_df = pd.DataFrame.from_dict(
    n_disqualified, orient='index', columns=['n_disqualified'])

print('\n\nDisqualified athletes:')
if n_disqualified_df['n_disqualified'].min() == 0:
    print("{} didn't have any disqualified athletes.".format(', '.join(str(x) for x in n_disqualified_df[n_disqualified_df['n_disqualified'] == n_disqualified_df['n_disqualified'].min()].index)))
else:
    print("All editions have disqualified athletes.")

print('The maximum number of disqualified athletes was {} in {}.'.format(n_disqualified_df['n_disqualified'].max(), ', '.join(
    str(x) for x in n_disqualified_df[n_disqualified_df['n_disqualified'] == n_disqualified_df['n_disqualified'].max()].index)))
print('The average number of disqualified athletes is {}.'.format(
    n_disqualified_df['n_disqualified'].mean()))


# Youngest and Oldest winners ever
print('\n\nYoungest and Oldest winners ever:')

ages_dict = {}

for year in complete_info_df['Year'].unique():
    winner = complete_info_df[complete_info_df['Year'] == year].head(n=1)
    winner_age = winner['Age'].values[0]
    ages_dict[year] = winner_age

winners_ages = pd.DataFrame.from_dict(ages_dict, orient='index',
                                      columns=['Age'])

youngest_age = winners_ages['Age'].min()
y_w = winners_ages[winners_ages['Age'] == youngest_age]
youngest_winners_year = list(y_w.index)
youngest_winners = [complete_info_df[complete_info_df['Year'] == x].head(n=1)['Name'].values[0] for x in youngest_winners_year]

if len(youngest_winners_year) == 1:
    print('The youngest winner of all time is {}, being {} years-old.'.format(''.join(youngest_winners[0]), youngest_age))
else:
    print('The youngest winners of all time were {}, being {} years-old.'.format(', '.join(youngest_winners), youngest_age))

oldest_age = winners_ages['Age'].max()
o_w = winners_ages[winners_ages['Age'] == oldest_age]
oldest_winners_year = list(o_w.index)
oldest_winners = [complete_info_df[complete_info_df['Year'] == x].head(n=1)['Name'].values[0] for x in oldest_winners_year]

if len(oldest_winners_year) == 1:
    print('The oldest winner of all time is {}, being {} years-old.'.format(''.join(oldest_winners[0]), oldest_age))
else:
    print('The oldest winners of all time were {}, being {} years-old.'.format(', '.join(oldest_winners), oldest_age))

print('The difference between the oldest and youngest winner ever is {} years.\n'.format(oldest_age-youngest_age))


# PLOT 1 - Average score per year for all athletes and medalists.

avg_all_athletes = complete_info_df.groupby('Year')['Score'].mean().plot(
    xticks=complete_info_df['Year'].unique(), style='o-')


medalists.groupby('Year')['Score'].mean().plot(
    xticks=complete_info_df['Year'].unique(), ax=avg_all_athletes, style='o-')
avg_all_athletes.legend(['All Athletes', 'Medalists'], loc='upper center')
avg_all_athletes.set_ylabel('Score (kg)')
plt.title('Average Score per Year')
plt.ylim(215, 285)
plt.margins(x=0.1)

xs_athletes = np.sort(complete_info_df['Year'].unique())
ys_athletes = complete_info_df.groupby('Year')['Score'].mean()
xs_medalists = np.sort(medalists['Year'].unique())
ys_medalists = medalists.groupby('Year')['Score'].mean()

for pair in [(xs_athletes, ys_athletes), (xs_medalists, ys_medalists)]:
    for x, y in zip(pair[0], pair[1]):
        label = "{:.2f}".format(y)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(
            0, 10), bbox=dict(boxstyle="round", fc="0.9", ec="gray"), ha='center')
print('\nAverage Score per Year graph information:')
plt.show()
print('Average medalists score in all editions: {}\n'.format(
    round(medalists['Score'].mean(), 3)))


# PLOT 2 - Bar plot with the range between first and last place and between first and third place in each edition

first_last = []
first_third = []
third_last = []

for year in complete_info_df['Year'].unique():
    max_range, podium_range = calc_diff(complete_info_df, year)
    first_last.append(max_range)
    first_third.append(podium_range)
    third_last.append(max_range-podium_range)

first_third_range = plt.bar(
    complete_info_df['Year'].unique(), first_third, width=1.8, color='gold')
first_last_range = plt.bar(
    complete_info_df['Year'].unique(), third_last, bottom=first_third, width=1.8)
plt.xticks(complete_info_df['Year'].unique(),
           complete_info_df['Year'].unique())
plt.legend((first_third_range[0], first_last_range[0]),
           ('Medalists', 'All Athletes'), loc='upper center')
plt.ylim(0, 180)
plt.ylabel('Scores')
plt.xlabel('Year')
plt.title('Range of Scores')
print('\nDifference between first and last place and difference between first and third place in each edition graph information:')
plt.show()
print('Average difference between first and last place: {}'.format(
    sum(first_last)/len(first_last)))
print('Biggest difference between first and last place: {}'.format(max(first_last)))
print('Lowest difference between first and last place: {}'.format(min(first_last)))
print('Average difference between first and third place: {}'.format(
    sum(first_third)/len(first_third)))
print('Biggest difference between first and third place: {}'.format(max(first_third)))
print('Lowest difference between first and third place: {}'.format(min(first_third)))


# TABLE 1 - Ratios between bodyweight and weight lifted. 

position_by_ratio_df = pd.DataFrame()

for index, year_results in complete_info_df.groupby('Year'):
    year_results = year_results.copy()
    year_results['Snatch/Weight'] = year_results.apply(
        lambda x: calc_ratio(x['Snatch'], x['Bodyweight']), axis=1)
    year_results['Clean&Jerk/Weight'] = year_results.apply(
        lambda x: calc_ratio(x['Clean and Jerk'], x['Bodyweight']), axis=1)
    year_results['Average Ratio'] = (
        year_results['Snatch/Weight']+year_results['Clean&Jerk/Weight'])/2
    year_results = year_results.sort_values(
        by='Average Ratio', ascending=False)
    year_results['Position'] = year_results['Score'].rank(
        ascending=False, method='first').astype(int)
    year_results['Position by ratio'] = year_results['Average Ratio'].rank(
        ascending=False, method='first').astype(int)
    year_results = year_results.drop(columns=['Nation', 'Group', 'Age'])
    position_by_ratio_df = position_by_ratio_df.append(year_results.head())
position_by_ratio_df = position_by_ratio_df.set_index(['Year', 'Athlete ID'])
pd.set_option('display.max_columns', 12)

print('\n\nWeight lifted vs. Bodyweight ratio.\nNew positions are based on the average ratio of Snatch and Clean and Jerk:\n')
print(position_by_ratio_df)
print('All time highest average ratio: {}'.format(
    round(position_by_ratio_df['Average Ratio'].max(), 3)))
print('Average Ratio: {}'.format(
    round(position_by_ratio_df['Average Ratio'].mean(), 3)))
print('Standard Deviation: {}\n'.format(
    round(position_by_ratio_df['Average Ratio'].std(), 5)))


# GRAPH 3
# Gráfico Média de Score vs. Idade

score_mean = []
score_std = []

for age in complete_info_df['Age'].unique():
    age_df = complete_info_df[complete_info_df['Age'] == age]
    score_mean.append(age_df['Score'].mean())
    score_std.append(age_df['Score'].std())

plt.bar(
    complete_info_df['Age'].unique(), score_mean, width=0.6, yerr=score_std, capsize=4, error_kw=dict(ecolor='chocolate', elinewidth=0.5), color='orange')
plt.xticks(
    np.arange(complete_info_df['Age'].min(), complete_info_df['Age'].max()+1))
plt.ylim(100, 280)
plt.ylabel('Scores')
plt.xlabel('Age')
ages = complete_info_df['Age'].value_counts().sort_index()
ages = ages.reindex(pd.RangeIndex(
    ages.index.min(), ages.index.max() + 1)).fillna(0)
ages.plot(secondary_y=True, color='blue')
plt.ylabel('Number of Athletes')
plt.ylim(0, ages.max()+1)
plt.title('Average Scores per Age')
print('\nAverage Scores per Age graph information:')
print('Highest average score: {}'.format(max(score_mean)))
ages_most_part =  ages[ages == ages.max()].index
print('Most common age of participation is {} with {} participants.'.format(
    ', '.join([str(x) for x in ages_most_part]), int(ages.max())))
plt.show()


# TABLE 2 - Invalid lifts per year table.

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

    winner_id = int(complete_info_df[complete_info_df['Year'] == year].head(n=1)['Athlete ID'])
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
failed_lifts_df.index.name = 'Year'
failed_lifts_df.columns = pd.MultiIndex.from_tuples(list(zip(
    ['Average All Athletes', 'Average All Athletes', 'Average All Athletes', 'Winner', 'Winner', 'Winner'], failed_lifts_df.columns)))

print("\n\nAverage invalid lifts per lift type in each year and winner's invalid lifts:\n")
print(failed_lifts_df)
print('\nAverage failed Snatch lifts: {}'.format(
    round(failed_lifts_df['Average All Athletes', 'Failed Snatch'].mean(), 3)))
print('Average failed Clean and Jerk lifts: {}'.format(round(
    failed_lifts_df['Average All Athletes', 'Failed Clean and Jerk'].mean(), 3)))
print('Average failed lifts: {}'.format(
    round(failed_lifts_df['Average All Athletes', 'Overall Failed'].mean(), 3)))
print('{} has the lowest average of failed lifts: {}'.format(failed_lifts_df[failed_lifts_df['Average All Athletes', 'Overall Failed'] == failed_lifts_df[
      'Average All Athletes', 'Overall Failed'].min()].index[0].astype(int), round(failed_lifts_df['Average All Athletes', 'Overall Failed'].min(), 3)))
print('{} has the highest average of failed lifts: {}\n'.format(failed_lifts_df[failed_lifts_df['Average All Athletes', 'Overall Failed'] == failed_lifts_df[
      'Average All Athletes', 'Overall Failed'].max()].index[0].astype(int), round(failed_lifts_df['Average All Athletes', 'Overall Failed'].max(), 3)))


#SAVE TABLES:
saveImages("table",dfi, position_by_ratio_df)
saveImages("table",dfi, failed_lifts_df)
