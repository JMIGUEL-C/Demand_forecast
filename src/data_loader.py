"""
Module for downloading and storing data from the XM public API.
Allows obtaining historical energy demand data for a specific city and date range.
"""
import requests
import json
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_xm_data_range(dataset_id, start_date, end_date, city="MC-Cali", save_path="data/xm_api_data.csv"):
    """
    Downloads historical data from the XM public API for a specific date range and city.
    Data is downloaded month by month to avoid API limits and saved to a CSV file.

    Args:
        dataset_id (str): Dataset ID in the XM API.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        city (str): City or market to query (default 'MC-Cali').
        save_path (str): Path where to save the resulting CSV file.
    Returns:
        pd.DataFrame: DataFrame with downloaded data. If no data, returns empty DataFrame.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    all_dfs = []
    current = start
    while current <= end:
        # Calculate the last day of the current month
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        last_day = min(next_month - timedelta(days=1), end)
        s_date = current.strftime("%Y-%m-%d")
        e_date = last_day.strftime("%Y-%m-%d")
        print(f"Downloading data from {s_date} to {e_date}...")
        url = (
            f"https://www.simem.co/backend-files/api/PublicData"
            f"?startDate={s_date}"
            f"&enddate={e_date}"
            f"&datasetId={dataset_id}"
            f"&columnDestinyName=MercadoComercializacionOperativo"
            f"&values={city}"
        )
        response = requests.get(url)
        if response.status_code == 200:
            data = json.loads(response.content)
            try:
                records = data['result']['records']
                if records:
                    df = pd.DataFrame(records)
                    all_dfs.append(df)
                else:
                    print(f"No records found for {s_date} to {e_date}.")
            except Exception as e:
                print(f"Error extracting records from {s_date} to {e_date}:", e)
        else:
            print(f"Request error for {s_date} to {e_date}: {response.status_code}")
            print(response.text)
        # Move to next month
        current = next_month
    # Concatenate all downloaded DataFrames
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        final_df.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")
        return final_df
    else:
        print("No data obtained in the requested range.")
        return pd.DataFrame()


