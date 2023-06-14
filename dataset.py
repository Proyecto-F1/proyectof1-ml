import pandas as pd

weather_df = pd.read_csv('../data/weather.csv')
results_df = pd.read_csv('../data/results.csv')
races_df = pd.read_csv('../data/races.csv')

races_df['raceId'] = races_df['raceId'].astype(int)
results_df['raceId'] = results_df['raceId'].astype(int)
weather_df['raceId'] = weather_df['raceId'].astype(int)

# Filtramos los datos de 2014 en adelante

final_df = results_df.merge(races_df, on='raceId', how='left').merge(
    weather_df, on='raceId', how='left')

final_df.reset_index(drop=True, inplace=True)

final_df['date'] = pd.to_datetime(final_df['date_y'])

final_df = final_df[final_df['date'].dt.year >= 2014]

final_df.reset_index(drop=True, inplace=True)

cols = ['raceId', 'circuitId_y', 'driverId', 'constructorId',
        'position', 'grid', 'statusId', 'temp', 'precipitation']

final_df = final_df.loc[:, cols]

final_df = final_df.dropna()
final_df['circuitId'] = final_df['circuitId_y']
final_df = final_df.drop(columns=['circuitId_y'])

final_df.reset_index(drop=True, inplace=True)
