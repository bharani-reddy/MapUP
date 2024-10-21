#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:

    unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))

    distance_matrix = pd.DataFrame(np.nan, index=unique_ids, columns=unique_ids)

    for _, row in df.iterrows():
        distance_matrix.at[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.at[row['id_end'], row['id_start']] = row['distance']  # Ensure symmetry

    np.fill_diagonal(distance_matrix.values, 0)

    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if pd.notna(distance_matrix.at[i, k]) and pd.notna(distance_matrix.at[k, j]):
                    new_distance = distance_matrix.at[i, k] + distance_matrix.at[k, j]
                    if pd.isna(distance_matrix.at[i, j]) or new_distance < distance_matrix.at[i, j]:
                        distance_matrix.at[i, j] = new_distance

    return distance_matrix

file_path = r'C:\Users\Reddy\Downloads\dataset-2.csv'
df = pd.read_csv(file_path)

distance_matrix = calculate_distance_matrix(df)

print(distance_matrix)


# In[2]:


import pandas as pd

def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:

    unrolled_data = []
    
    for id_start in df.index:
        for id_end in df.columns:
            distance = df.at[id_start, id_end]
            if id_start != id_end:
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    unrolled_df = pd.DataFrame(unrolled_data)
    
    return unrolled_df

unrolled_df = unroll_distance_matrix(distance_matrix)
print(unrolled_df)


# In[3]:


import pandas as pd

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:

    avg_distance_ref = df[df['id_start'] == reference_id]['distance'].mean()
    
    if pd.isna(avg_distance_ref): 
        return pd.DataFrame(columns=['id_start', 'average_distance'])
    
    lower_bound = avg_distance_ref * 0.90
    upper_bound = avg_distance_ref * 1.10
    
    avg_distances = df.groupby('id_start')['distance'].mean().reset_index()
    
    filtered_ids = avg_distances[(avg_distances['distance'] >= lower_bound) & 
                                  (avg_distances['distance'] <= upper_bound)]
    
    filtered_ids = filtered_ids.sort_values(by='distance', ascending=True)
    
    return filtered_ids

reference_id = 1001400  # Example reference ID
result_df = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
print(result_df)


# In[4]:


import pandas as pd

def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:

    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    for vehicle_type, coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * coefficient
    
    return df

toll_rates_df = calculate_toll_rate(unrolled_df)
print(toll_rates_df)


# In[11]:


import pandas as pd
from datetime import time

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): Input DataFrame containing toll rates.

    Returns:
        pandas.DataFrame: DataFrame with time-based toll rates.
    """
    # Define discount factors
    weekday_discount_factors = {
        'morning': 0.8,   # 00:00 to 10:00
        'afternoon': 1.2, # 10:00 to 18:00
        'evening': 0.8    # 18:00 to 23:59
    }
    weekend_discount_factor = 0.7

    # Days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Prepare a list to collect new rows
    new_rows = []

    # Unique (id_start, id_end) pairs
    unique_pairs = df[['id_start', 'id_end', 'distance']].drop_duplicates()

    # Generate time-based toll rates for weekdays and weekends
    for index, row in unique_pairs.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']

        # Weekdays (Monday to Friday)
        for day in days_of_week[:5]:  # Monday to Friday
            # Morning
            new_rows.append({
                'id_start': id_start,
                'id_end': id_end,
                'distance': distance,
                'start_day': day,
                'start_time': time(0, 0),   # 12:00 AM
                'end_day': day,
                'end_time': time(10, 0),     # 10:00 AM
                'moto': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'moto'].values[0] * weekday_discount_factors['morning'],
                'car': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'car'].values[0] * weekday_discount_factors['morning'],
                'rv': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'rv'].values[0] * weekday_discount_factors['morning'],
                'bus': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'bus'].values[0] * weekday_discount_factors['morning'],
                'truck': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'truck'].values[0] * weekday_discount_factors['morning'],
            })
            # Afternoon
            new_rows.append({
                'id_start': id_start,
                'id_end': id_end,
                'distance': distance,
                'start_day': day,
                'start_time': time(10, 0),    # 10:00 AM
                'end_day': day,
                'end_time': time(18, 0),       # 06:00 PM
                'moto': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'moto'].values[0] * weekday_discount_factors['afternoon'],
                'car': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'car'].values[0] * weekday_discount_factors['afternoon'],
                'rv': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'rv'].values[0] * weekday_discount_factors['afternoon'],
                'bus': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'bus'].values[0] * weekday_discount_factors['afternoon'],
                'truck': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'truck'].values[0] * weekday_discount_factors['afternoon'],
            })
            # Evening
            new_rows.append({
                'id_start': id_start,
                'id_end': id_end,
                'distance': distance,
                'start_day': day,
                'start_time': time(18, 0),    # 06:00 PM
                'end_day': day,
                'end_time': time(23, 59),      # 11:59 PM
                'moto': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'moto'].values[0] * weekday_discount_factors['evening'],
                'car': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'car'].values[0] * weekday_discount_factors['evening'],
                'rv': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'rv'].values[0] * weekday_discount_factors['evening'],
                'bus': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'bus'].values[0] * weekday_discount_factors['evening'],
                'truck': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'truck'].values[0] * weekday_discount_factors['evening'],
            })

        # Weekends (Saturday and Sunday)
        for day in days_of_week[5:]:  # Saturday and Sunday
            new_rows.append({
                'id_start': id_start,
                'id_end': id_end,
                'distance': distance,
                'start_day': day,
                'start_time': time(0, 0),    # 12:00 AM
                'end_day': day,
                'end_time': time(23, 59),      # 11:59 PM
                'moto': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'moto'].values[0] * weekend_discount_factor,
                'car': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'car'].values[0] * weekend_discount_factor,
                'rv': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'rv'].values[0] * weekend_discount_factor,
                'bus': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'bus'].values[0] * weekend_discount_factor,
                'truck': df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'truck'].values[0] * weekend_discount_factor,
            })

    # Create a new DataFrame from the new rows
    return pd.DataFrame(new_rows)

# Sample DataFrame Creation
def create_sample_toll_rates_df() -> pd.DataFrame:
    """
    Create a sample DataFrame to simulate the toll rates.
    
    Returns:
        pd.DataFrame: Sample DataFrame with id_start, id_end, distance, and toll rates.
    """
    # Sample data including the new entries
    data = {
        'id_start': [1001400, 1001402, 1001404, 1001408, 1001400, 1001408],
        'id_end': [1001402, 1001404, 1001406, 1001410, 1001402, 1001410],
        'distance': [10.0, 20.0, 30.0, 11.1, 9.7, 11.1],
        'moto': [9.7, 20.2, 16.0, 12.5, 9.7, 12.5],
        'car': [12.0, 22.0, 18.0, 15.0, 12.0, 15.0],
        'rv': [15.0, 25.0, 20.0, 17.0, 15.0, 17.0],
        'bus': [18.0, 28.0, 22.0, 19.0, 18.0, 19.0],
        'truck': [25.0, 35.0, 30.0, 27.0, 25.0, 27.0]
    }
    
    return pd.DataFrame(data)

# Main Execution
# Create a sample DataFrame
toll_rates_df = create_sample_toll_rates_df()

# Calculate time-based toll rates
result_df = calculate_time_based_toll_rates(toll_rates_df)

# Display the result DataFrame
print(result_df)


# In[ ]:




