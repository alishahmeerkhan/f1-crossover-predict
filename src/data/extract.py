import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import fastf1

load_dotenv()

def extractData(session):
    print('Extracting and cleaning lap data...')
    
    laps = session.laps
    laps = laps[laps['IsAccurate'] == True]
    weather_data = laps.get_weather_data()
    
    laps = laps.reset_index(drop=True)
    weather_data = weather_data.reset_index(drop=True)
    merged_df = pd.concat([laps, weather_data], axis=1)

    target_columns = [
        'Driver', 'Compound', 'TyreLife', 
        'LapTime', 'AirTemp', 'TrackTemp', 'Rainfall'
    ]

    ml_df = merged_df[target_columns].copy()

    ml_df['LapTime_s'] = ml_df['LapTime'].dt.total_seconds()
    ml_df = ml_df.drop(columns=['LapTime'])
    
    ml_df = ml_df.dropna()
    
    print(f'Data cleaned. Final dataset shape: {ml_df.shape}')
    return ml_df

def saveData(final_df):
    processed_dir_str = os.getenv('DATA_PATH')
    
    if not processed_dir_str:
        raise ValueError(
            "CRITICAL: 'DATA_PATH' not found in environment variables. "
            "Please check your .env file at the project root."
        )
    
    processed_dir = Path(processed_dir_str)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # CRITICAL FIX: Use processed_dir (the Path object), not processed_dir_str (the string)
    target_file_path = processed_dir / 'data.csv'
    
    print(f'Exporting dataset ({len(final_df)} rows) to destination...')
    final_df.to_csv(target_file_path, index=False)
    
    print(f'Success! Master dataset secured at: {target_file_path.resolve()}')

if __name__ == '__main__':
    final_df = pd.DataFrame()

    curated_races = [
        # --- The Crossover Classics (Wet-to-Dry / Dry-to-Wet) ---
        (2023, 13, 'R'), (2021, 15, 'R'), (2022, 7,  'R'), (2022, 17, 'R'), (2023, 6,  'R'), (2021, 2,  'R'), 
        (2021, 16, 'R'), (2022, 18, 'R'),

        # --- High-Energy / High-Degradation (Thermal & Wear) ---
        (2023, 10, 'R'), (2023, 9,  'R'), (2023, 1,  'R'), (2024, 1,  'R'), (2022, 1,  'R'),

        # --- Low-Energy / Smooth Asphalt (Graining Focus) ---
        (2023, 21, 'R'), (2023, 14, 'R')]

    for race in curated_races:
        print(f'Extracing Data from {race}')
        session = fastf1.get_session(race[0], race[1], race[2])
        session.load(telemetry=False, weather=True)
    
        clean_df = extractData(session)

        final_df = pd.concat([final_df, clean_df], ignore_index=True)
        print('*' * 60)

    print('\nSample of ML-ready DataFrame:')
    print(final_df.head())
    print(final_df.shape)

    print('\nSaving Data in CSV.')
    saveData(final_df)