# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:43:13 2024

@author: ditte
"""

import streamlit as st
import folium
from folium.plugins import MarkerCluster
import pandas as pd
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from geopy.distance import great_circle
import numpy as np
from scipy.spatial import cKDTree
from heapq import heappop, heappush
from vinc import v_direct
import os
from ipywidgets import interact, widgets
from branca.element import Template, MacroElement

#%%

# Load Excel files into a DataFrame
flight_1 = pd.read_excel('30Flight 1.xlsx')
flight_2 = pd.read_excel('30Flight 2.xlsx')
flight_3 = pd.read_excel('30Flight 3.xlsx')
flight_4 = pd.read_excel('30Flight 4.xlsx')
flight_5 = pd.read_excel('30Flight 5.xlsx')
flight_6 = pd.read_excel('30Flight 6.xlsx')
flight_7 = pd.read_excel('30Flight 7.xlsx')

# Creating a column for each flight with their unique flight number
flight_1['Flight'] = 1
flight_2['Flight'] = 2 
flight_3['Flight'] = 3
flight_4['Flight'] = 4 
flight_5['Flight'] = 5 
flight_6['Flight'] = 6 
flight_7['Flight'] = 7 

flight_1.dropna(inplace=True)
flight_2.dropna(inplace=True)
flight_3.dropna(inplace=True)
flight_4.dropna(inplace=True)
flight_5.dropna(inplace=True)
flight_6.dropna(inplace=True)
flight_7.dropna(inplace=True)

# Renaming some columns
new_column_names = {'Time (secs)' : 'timestamp', '[3d Latitude]' : 'latitude', '[3d Longitude]' : 'longitude'}
flight_1 = flight_1.rename(columns=new_column_names)
flight_2 = flight_2.rename(columns=new_column_names)
flight_3 = flight_3.rename(columns=new_column_names)
flight_4 = flight_4.rename(columns=new_column_names)
flight_5 = flight_5.rename(columns=new_column_names)
flight_6 = flight_6.rename(columns=new_column_names)
flight_7 = flight_7.rename(columns=new_column_names)

# Calculating the distance between waypoints
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).meters

# Calculating the total distance per flight
def calculate_cumulative_distance(flight):
    distances = [0]
    for i in range(1, len(flight)):
        coord1 = (flight['latitude'].iloc[i-1], flight['longitude'].iloc[i-1])
        coord2 = (flight['latitude'].iloc[i], flight['longitude'].iloc[i])
        distances.append(distances[-1] + calculate_distance(coord1, coord2))
    return distances

# Dividing the flights into 14 equidistant waypoints using total distance and an interval distance of total distance divided by 15
def determine_equidistant_waypoints(flight, num_waypoints=14):
    total_distance = calculate_cumulative_distance(flight)[-1]
    interval_distance = total_distance / (num_waypoints + 1)

    # Initializing an empty dictionary with waypoints and setting the cumulative distance to zero
    waypoints = []
    cumulative_distance = 0
    
    # Setting a target distance to reach each interval distance
    for i in range(num_waypoints):
        target_distance = (i + 1) * interval_distance
       # Update cumulative distance till it exceeds the target distance
        while cumulative_distance < target_distance:
            # Calculating the distance between the current and the next coordinates
            cumulative_distance += calculate_distance(
                (flight['latitude'].iloc[0], flight['longitude'].iloc[0]),
                (flight['latitude'].iloc[1], flight['longitude'].iloc[1])
            )
            # Removing the first row to move to the next coordinates
            flight = flight.iloc[1:]
           
        # Using the timestamp column to append the waypoints to the waypoint dictionary
        waypoints.append(flight['timestamp'].iloc[0])

    return waypoints

# Creating a list of flight DataFrame names
flight_names = ['flight_1', 'flight_2', 'flight_3', 'flight_4', 'flight_5', 'flight_6', 'flight_7']

# Iterate over flight DataFrames to create 14 equidistant waypoints for each flight excluding top of climb and top of descent
for flight_name in flight_names:
    current_flight = globals()[flight_name]
    equidistant_waypoints = determine_equidistant_waypoints(current_flight, num_waypoints=16)
    
    # Update the flight DataFrames with only their equidistant waypoint
    globals()[flight_name] = current_flight[current_flight['timestamp'].isin(equidistant_waypoints)]


# Merging all the flight data into one dataframe
all_flights = pd.concat([flight_1, flight_2, flight_3, flight_4, flight_5, flight_6, flight_7], ignore_index=True)

# Renaming columns 
all_flights = all_flights.rename(columns={'latitude': 'LAT', 'longitude': 'LON', '[3d Heading]': 'HEADING'})

#Creating a column to give every waypoint a number to create a sequence between the waypoints for each flight
sequence_length = 16
num_repeats = len(all_flights) // sequence_length + 1
all_flights['Waypoint'] = np.tile(np.arange(1, sequence_length + 1), num_repeats)[:len(all_flights)]


# Loading temperature data
TMPDT1 = pd.read_csv('TMP_date_1_alt_1.csv')
TMPDT2 = pd.read_csv('TMP_date_1_alt_2.csv')
TMPDT3 = pd.read_csv('TMP_date_1_alt_3.csv')
TMPDT4 = pd.read_csv('TMP_date_1_alt_4.csv')
TMPDT5 = pd.read_csv('TMP_date_1_alt_5.csv')
TMPDT6 = pd.read_csv('TMP_date_1_alt_6.csv')
TMPDT7 = pd.read_csv('TMP_date_1_alt_7.csv')


# Loading wind direction data
WDIRDT1 = pd.read_csv('WDIR_date_1_alt_1.csv')
WDIRDT2 = pd.read_csv('WDIR_date_1_alt_2.csv')
WDIRDT3 = pd.read_csv('WDIR_date_1_alt_3.csv')
WDIRDT4 = pd.read_csv('WDIR_date_1_alt_4.csv')
WDIRDT5 = pd.read_csv('WDIR_date_1_alt_5.csv')
WDIRDT6 = pd.read_csv('WDIR_date_1_alt_6.csv')
WDIRDT7 = pd.read_csv('WDIR_date_1_alt_7.csv')

# Loading wind speed data
WINDSPDT1 = pd.read_csv('WIND_date_1_alt_1.csv')
WINDSPDT2 = pd.read_csv('WIND_date_1_alt_2.csv')
WINDSPDT3 = pd.read_csv('WIND_date_1_alt_3.csv')
WINDSPDT4 = pd.read_csv('WIND_date_1_alt_4.csv')
WINDSPDT5 = pd.read_csv('WIND_date_1_alt_5.csv')
WINDSPDT6 = pd.read_csv('WIND_date_1_alt_6.csv')
WINDSPDT7 = pd.read_csv('WIND_date_1_alt_7.csv')

# Giving the temperature data column names
column_nameTEMP = ['long', 'lat', 'tempK']
TMPDT1.columns = column_nameTEMP
TMPDT2.columns = column_nameTEMP
TMPDT3.columns = column_nameTEMP
TMPDT4.columns = column_nameTEMP
TMPDT5.columns = column_nameTEMP
TMPDT6.columns = column_nameTEMP
TMPDT7.columns = column_nameTEMP

# Giving the wind direction data column names
column_nameWDIR = ['longWD', 'latWD', 'WDIR_degrees']
WDIRDT1.columns = column_nameWDIR
WDIRDT2.columns = column_nameWDIR
WDIRDT3.columns = column_nameWDIR
WDIRDT4.columns = column_nameWDIR
WDIRDT5.columns = column_nameWDIR
WDIRDT6.columns = column_nameWDIR
WDIRDT7.columns = column_nameWDIR

#Giving the wind speed data column names
column_nameWINDSP = ['longWINDSP', 'latWINDSP', 'WINDSP_kts']
WINDSPDT1.columns = column_nameWINDSP
WINDSPDT2.columns = column_nameWINDSP
WINDSPDT3.columns = column_nameWINDSP
WINDSPDT4.columns = column_nameWINDSP
WINDSPDT5.columns = column_nameWINDSP
WINDSPDT6.columns = column_nameWINDSP
WINDSPDT7.columns = column_nameWINDSP

# Assigning a unique identifier to each set of weather data
TMPDT1['MergeKey'] = range(len(TMPDT2))
WDIRDT1['MergeKey'] = range(len(WDIRDT2))
WINDSPDT1['MergeKey'] = range(len(WINDSPDT2))

TMPDT2['MergeKey'] = range(len(TMPDT2))
WDIRDT2['MergeKey'] = range(len(WDIRDT2))
WINDSPDT2['MergeKey'] = range(len(WINDSPDT2))

TMPDT3['MergeKey'] = range(len(TMPDT3))
WDIRDT3['MergeKey'] = range(len(WDIRDT3))
WINDSPDT3['MergeKey'] = range(len(WINDSPDT3))

TMPDT4['MergeKey'] = range(len(TMPDT4))
WDIRDT4['MergeKey'] = range(len(WDIRDT4))
WINDSPDT4['MergeKey'] = range(len(WINDSPDT4))

TMPDT5['MergeKey'] = range(len(TMPDT5))
WDIRDT5['MergeKey'] = range(len(WDIRDT5))
WINDSPDT5['MergeKey'] = range(len(WINDSPDT5))

TMPDT6['MergeKey'] = range(len(TMPDT6))
WDIRDT6['MergeKey'] = range(len(WDIRDT6))
WINDSPDT6['MergeKey'] = range(len(WINDSPDT6))

TMPDT7['MergeKey'] = range(len(TMPDT7))
WDIRDT7['MergeKey'] = range(len(WDIRDT7))
WINDSPDT7['MergeKey'] = range(len(WINDSPDT7))

#Merge the temperature, wind direction and wind speed dataframes with the mergekeys
weather1 = pd.merge(pd.merge(TMPDT1, WDIRDT1, on='MergeKey',
                   how='outer'), WINDSPDT1, on='MergeKey', how='outer')

weather2 = pd.merge(pd.merge(TMPDT2, WDIRDT2, on='MergeKey',
                   how='outer'), WINDSPDT2, on='MergeKey', how='outer')

weather3 = pd.merge(pd.merge(TMPDT3, WDIRDT3, on='MergeKey',
                   how='outer'), WINDSPDT3, on='MergeKey', how='outer')

weather4 = pd.merge(pd.merge(TMPDT4, WDIRDT4, on='MergeKey',
                   how='outer'), WINDSPDT4, on='MergeKey', how='outer')

weather5 = pd.merge(pd.merge(TMPDT5, WDIRDT5, on='MergeKey',
                   how='outer'), WINDSPDT5, on='MergeKey', how='outer')

weather6 = pd.merge(pd.merge(TMPDT6, WDIRDT6, on='MergeKey',
                   how='outer'), WINDSPDT6, on='MergeKey', how='outer')

weather7 = pd.merge(pd.merge(TMPDT7, WDIRDT7, on='MergeKey',
                   how='outer'), WINDSPDT7, on='MergeKey', how='outer')

# Merging all the weather data
weather = pd.concat([weather1, weather2, weather3, weather4, weather5, weather6, weather7], ignore_index=True)

    # Dropping columns
columns_to_drop = ['MergeKey', 'longWD', 'latWD', 'longWINDSP', 'latWINDSP']
weather.drop(columns=columns_to_drop, axis=1, inplace=True)

    # Renaming the columns to the same name as the all_flights dataframe
column_rename_dict = {'long': 'LON', 'lat': 'LAT', 'tempK': 'temperature'}
weather = weather.rename(columns=column_rename_dict)

# Using CKDtree to match the weather coordinates to the flight coordinates
weather_tree = cKDTree(weather[['LAT', 'LON']])

# Finding the closest points of weather for every waypoint in all_flights
all_flights['closest_point_index'] = [weather_tree.query(
    [lat, lon])[1] for lat, lon in zip(all_flights['LAT'], all_flights['LON'])]

# Merging the weather and flight data
all_flights = pd.merge(all_flights, weather, left_on='closest_point_index', right_index=True, how='left')

# Dropping duplicates coordinate and the closest point index and renaming longitude and latitude to their old names
all_flights.drop('closest_point_index', axis=1, inplace=True)
all_flights = all_flights.drop(columns=['LON_y', 'LAT_y'])
all_flights = all_flights.rename(columns={'LAT_x': 'LAT', 'LON_x': 'LON'})

# Defining a function for the ground speed with the windspeed, wind direction, heading and true airspeed
def calculate_ground_speed(row):
    wind_component = row['WINDSP_kts'] * \
        np.cos(np.radians(row['WDIR_degrees'] - row['HEADING']))
    ground_speed = row['TRUE AIRSPEED (derived)'] - wind_component
    return ground_speed

# Calling the function to calculate the groundspeed and converting the groundspeed to m/s
all_flights['Groundspeed in kts'] = all_flights.apply(calculate_ground_speed, axis=1)
all_flights['Groundspeed in m/s'] = all_flights['Groundspeed in kts'] * (1852/3600)

# Sort the DataFrame based on 'Flight' and 'waypoints' columns
all_flights.sort_values(by=['Flight', 'Waypoint'], inplace=True)

# Create a new column for the groundspeed of each next waypoint and initialize it with NaN
all_flights['groundspeed_to_next'] = float('nan')

# Iterate over flights and update groundspeed of the next waypoint for each waypoint
for Flight in all_flights['Flight'].unique():
    flight_mask = all_flights['Flight'] == Flight
    all_flights.loc[flight_mask, 'groundspeed_to_next'] = all_flights[flight_mask]['Groundspeed in m/s'].shift(-1)

# Fill the value for groundspeed next of the final waypoints with their own groundspeed
all_flights['groundspeed_to_next'].fillna(all_flights['Groundspeed in m/s'], inplace=True)

# Create a new column 'average_groundspeed' and calculate the average groundspeed between waypoints
all_flights['Groundspeed average'] = (all_flights['Groundspeed in m/s'] + all_flights['groundspeed_to_next']) / 2

# Defining a function to calculate the distance between waypoints
def geodesic_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers

# Calculate differences in latitude and longitude from each waypoint to their next waypoint within each flight group 
all_flights['Lat_diff'] = all_flights.groupby('Flight')['LAT'].diff(-1)
all_flights['Lon_diff'] = all_flights.groupby('Flight')['LON'].diff(-1)

# Calling the geodesic function to calculate the distance between waypoints
all_flights['Distance'] = all_flights.apply(
    lambda row: geodesic_distance(
        (row['LAT'], row['LON']),
        (row['LAT'] + row['Lat_diff'], row['LON'] + row['Lon_diff'])
    ) if row['Waypoint'] < all_flights.groupby('Flight')['Waypoint'].transform('max').iloc[0] else 0,
    axis=1
)

# Dropping the intermediate columns used for the distance calculation
all_flights = all_flights.drop(['Lat_diff', 'Lon_diff'], axis=1)

# Calcaulating the time using the groundspeed and distance
all_flights['Time in seconds'] = all_flights['Distance'] * 1000 / all_flights['Groundspeed average']

# Saving the dataframe as a csv file
all_flights.to_csv('AMS_BCN.csv', index=False, mode='w')

#%%

# Loading the flight data
csv_file_name = "AMS_BCN.csv"
trajectories = pd.read_csv(csv_file_name)

# Sorting the flight data by waypoints
trajectories = trajectories.sort_values(by='Waypoint')
trajectories = trajectories.reset_index(drop=True)

# Creating variables for number of flight paths number, of waypoints, total waypoints and range untill final waypoint
number_of_flight_paths = trajectories['Flight'].nunique()
number_of_waypoints = trajectories['Waypoint'].nunique()
total_waypoints = number_of_flight_paths * number_of_waypoints
range_untill_final_waypoints = total_waypoints - number_of_flight_paths

# Creating a function that calculates the distance between waypoints using geodistance and the longitude and latitude
def calculate_distance(current_waypoint, options):
    waypoints_distances = {}
    for x in options:
        distance = geodesic(
            (trajectories['LAT'][current_waypoint], trajectories['LON'][current_waypoint]),
            (trajectories['LAT'][x], trajectories['LON'][x])
        ).meters
        waypoints_distances[x] = distance
    return waypoints_distances

# Creating a function that stores all possible options for every waypoint in a dictionary called distance_dict
def get_all_options():
    distances_dict = {}
    waypoints = [list(range(i, i + number_of_flight_paths)) for i in range(0, total_waypoints, number_of_flight_paths)]
    for i in range(len(waypoints)):
        for j in range(len(waypoints[i])):
            if waypoints[i][j] <= (range_untill_final_waypoints - 1):
                distances_dict[waypoints[i][j]] = calculate_distance(waypoints[i][j], waypoints[i + 1])
    return distances_dict

#Creating a function that calculates the time between waypoints using the average groundspeed between points and the distance
def calculate_time(x, y, distance):
    ground_speed_x = trajectories.at[x, 'Groundspeed in m/s']
    ground_speed_y = trajectories.at[y, 'Groundspeed in m/s']
    average = (ground_speed_x + ground_speed_y) / 2
    return distance / average

# Creating a function that stores all the times for every waypoint to its possible options in a dictionary called times_dict
def get_all_times(all_options):
    times_dict = {}
    for x in all_options:
        times_dict[x] = {}
        for y in all_options[x]:
            times_dict[x][y] = calculate_time(x, y, all_options[x][y])
    return times_dict

# Creating a dijkstra function where times is set to infinity with a starting distance of zero
def dijkstra(all_options, starting_waypoint, ending_waypoint):
    times = {waypoint: float('infinity') for waypoint in all_options}
    times[starting_waypoint] = 0
   # A dictionary is initiated that stores the previous waypoints in the optimal path and a priority queue with the starting waypoint and its time
    previous = {waypoint: None for waypoint in all_options}
    priority_queue = [(0, starting_waypoint)]

    # Start of dijkstra's algorithm extracting the waypoint with the smallest known time
    while priority_queue:
        current_time, current_waypoint = heappop(priority_queue)
        # if a shorter path is found it will skip ahead to this path
        if current_time > times[current_waypoint]:
            continue

        # Checking neighboring waypoints if a shorter path is found and updated their distance if so
        if current_waypoint in all_options:
            for neighbor, time in all_options[current_waypoint].items():
                new_time = current_time + time
                if new_time < times.get(neighbor, float('inf')):
                    # Updating the time and previous waypoint 
                    times[neighbor] = new_time
                    previous[neighbor] = current_waypoint
                    # The neighbor is added to the priority queue with its updated time
                    heappush(priority_queue, (new_time, neighbor))

    # Initialize empty dictionary where optimal path will be stored
    path = []
    # When the final waypoint is reached the waypoints forming the optimal path will be stored in the path dictionary
    current_waypoint = ending_waypoint
    while current_waypoint is not None:
        path.insert(0, current_waypoint)
        current_waypoint = previous[current_waypoint]
    # The optimal path is returned
    return path

#Creating a function that calculates the starting and ending waypoints
def choose_median_waypoints():
    # Calculating the median longitude and latitude from the starting waypoints
    starting_waypoints = list(range(number_of_flight_paths))
    median_starting_longitude = np.median(trajectories.loc[starting_waypoints, 'LON'])
    median_starting_latitude = np.median(trajectories.loc[starting_waypoints, 'LAT'])
    median_starting_waypoint = min(starting_waypoints, key=lambda x: geodesic(
        (median_starting_latitude, median_starting_longitude),
        (trajectories.at[x, 'LAT'], trajectories.at[x, 'LON'])
    ).meters)

    # Calculating the median longitude and latitude from the ending waypoints 
    ending_waypoints = list(range(range_untill_final_waypoints, total_waypoints))
    median_ending_longitude = np.median(trajectories.loc[ending_waypoints, 'LON'])
    median_ending_latitude = np.median(trajectories.loc[ending_waypoints, 'LAT'])
    median_ending_waypoint = min(ending_waypoints, key=lambda x: geodesic(
        (median_ending_latitude, median_ending_longitude),
        (trajectories.at[x, 'LAT'], trajectories.at[x, 'LON'])
    ).meters)
    
    # Returning the starting and ending waypoint
    return median_starting_waypoint, median_ending_waypoint

# Calling the function to select a starting end ending waypoint
median_starting_point, median_ending_point = choose_median_waypoints()

# Calling the dijkstra function to find the optimal path with the get_all_times function and the selected starting and ending waypoints
path = dijkstra(get_all_times(get_all_options()), median_starting_point, median_ending_point)

# Creating a dataframe subset containing the points of the optimal path
df_subset = trajectories[(trajectories.index.isin(path))]

#%%

flight1=pd.read_excel('30Flight 1.xlsx')
flight2=pd.read_excel('30Flight 2.xlsx')
flight3=pd.read_excel('30Flight 3.xlsx')
flight4=pd.read_excel('30Flight 4.xlsx')
flight5=pd.read_excel('30Flight 5.xlsx')
flight6=pd.read_excel('30Flight 6.xlsx')
flight7=pd.read_excel('30Flight 7.xlsx')

# Assuming 'flight1', 'flight2', 'flight3', 'flight4', 'flight5', 'flight6', 'flight7' are your DataFrames
# Replace them with your actual DataFrames

# Drop NaN values from each DataFrame
flight1.dropna(inplace=True)
flight2.dropna(inplace=True)
flight3.dropna(inplace=True)
flight4.dropna(inplace=True)
flight5.dropna(inplace=True)
flight6.dropna(inplace=True)
flight7.dropna(inplace=True)

#%%

airports = pd.read_csv('airports-extended-clean.csv', sep=";")

schedule = pd.read_csv('schedule_airport.csv')

schedule=schedule.dropna()
print(schedule.isna().sum())

# List of airport codes
airport_codes=schedule['Org/Des'].unique()

airports = airports[airports['IATA'].isin(airport_codes) | airports['ICAO'].isin(airport_codes)]

org_des_counts = schedule['Org/Des'].value_counts().to_dict()

# Now, create a new column in airports dataframe called 'movements' and map the counts based on IATA code
airports['movements_IATA'] = airports['IATA'].map(org_des_counts)

# Now, create a new column in airports dataframe called 'movements' and map the counts based on ICAO code
airports['movements_ICAO'] = airports['ICAO'].map(org_des_counts)

# Fill NaN values with 0 in case an airport does not have any movement
airports['movements_IATA'].fillna(0, inplace=True)
airports['movements_ICAO'].fillna(0, inplace=True)

# Sum the movements from both IATA and ICAO into a single column 'movements'
airports['movements'] = airports['movements_IATA'] + airports['movements_ICAO']

# Drop intermediate columns if needed
airports.drop(['movements_IATA', 'movements_ICAO'], axis=1, inplace=True)

airports['Latitude'] = airports['Latitude'].apply(lambda x: float(x.replace(',', '.')))
airports['Longitude'] = airports['Longitude'].apply(lambda x: float(x.replace(',', '.')))

#%%

# Convert Latitude and Longitude columns to strings
airports['Latitude'] = airports['Latitude'].astype(str)
airports['Longitude'] = airports['Longitude'].astype(str)

# Replace comma with period in Latitude and Longitude columns
airports['Latitude'] = airports['Latitude'].str.replace(',', '.')
airports['Longitude'] = airports['Longitude'].str.replace(',', '.')

# Concatenate Latitude and Longitude into a new column called CORD1
airports['COR'] = airports['Latitude'] + ', ' + airports['Longitude']

# Coordinates of El Prat airport
zurich_coordinates = (47.4619, 8.5506)

# Function to calculate distance between two points
def calculate_distance(row):
    # Convert Latitude and Longitude to floats
    airport_coordinates = (float(row['Latitude']), float(row['Longitude']))
    distance = v_direct(airport_coordinates, zurich_coordinates)
    return distance

# Apply the function to calculate distance for each row and create a new column
airports['Distance_to_zurich'] = airports.apply(calculate_distance, axis=1)

# Format the distance column to display normal numbers
airports['Great Circle Distance Meters'] = airports['Distance_to_zurich'].apply(lambda x: '{:.2f}'.format(x))

# Conversion factor from meters to nautical miles
meter_to_nm_conversion = 0.000539957

# Function to convert distance in meters to nautical miles
def meters_to_nautical_miles(distance_in_meters):
    return distance_in_meters * meter_to_nm_conversion

# Apply the conversion function to create a new column with distances in nautical miles
airports['GCD_NM'] = airports['Distance_to_zurich'].astype(float).apply(meters_to_nautical_miles)

# Convert Latitude and Longitude columns to numeric values after replacing commas with periods
airports['Latitude'] = airports['Latitude'].str.replace(',', '.').astype(float)
airports['Longitude'] = airports['Longitude'].str.replace(',', '.').astype(float)

#%%


# Read airlines data
airlines = pd.read_csv("airlines.csv")



# Convert date column STD to datetime
schedule['STD'] = pd.to_datetime(schedule['STD'], format='%d/%m/%Y')

# Convert time column STA_STD_ltc to datetime, assuming it's in HH:MM:SS format
schedule['STA_STD_ltc'] = pd.to_datetime(schedule['STA_STD_ltc'])

# Convert time column ATA_ATD_ltc to datetime, assuming it's in HH:MM:SS format
schedule['ATA_ATD_ltc'] = pd.to_datetime(schedule['ATA_ATD_ltc'])

# Combine date from STD column with time from STA_STD_ltc column
schedule['STA_STD_ltc'] = schedule.apply(lambda row: row['STD'].replace(hour=row['STA_STD_ltc'].hour, minute=row['STA_STD_ltc'].minute, second=row['STA_STD_ltc'].second), axis=1)

# Combine date from STD column with time from ATA_ATD_ltc column
schedule['ATA_ATD_ltc'] = schedule.apply(lambda row: row['STD'].replace(hour=row['ATA_ATD_ltc'].hour, minute=row['ATA_ATD_ltc'].minute, second=row['ATA_ATD_ltc'].second), axis=1)

# Extract the data from the 'FLT' column and create a new column 'IATA code airline'
schedule['IATA code airline'] = schedule['FLT'].str[:2]

# Merge the two dataframes based on the common column 'IATA'
schedule_ext = pd.merge(schedule, airlines[['IATA', 'Name', 'Country']], left_on='IATA code airline', right_on='IATA', how='left')

# Rename the merged column to 'Airline Name'
schedule_ext.rename(columns={'Name': 'Airline Name'}, inplace=True)

# Drop IATA column
schedule_ext.drop('IATA', axis=1, inplace=True)

# Select relevant columns
schedule2 = schedule_ext[['ATA_ATD_ltc', 'LSV']]

# Sort schedule2 by ATA_ATD_ltc
schedule2_sorted = schedule2.sort_values(by='ATA_ATD_ltc')

# Reset index
schedule2_sorted = schedule2_sorted.reset_index(drop=True)

# Create a new DataFrame to store the count of planes at each time point
plane_count = pd.DataFrame()

# Calculate the net change in the number of planes at each time point
plane_count['Time'] = schedule2_sorted['ATA_ATD_ltc']
plane_count['Change'] = schedule2_sorted['LSV'].apply(lambda x: 1 if x == 'L' else -1)

# Initialize the Total column with a starting value
start_value = 300  # Change this to your desired starting value
plane_count['Total'] = start_value

# Aggregate the changes to get the total count of planes at each time point
plane_count['Total'] += plane_count['Change'].cumsum()

# Ensure that the number of planes doesn't go below 0
plane_count['Total'] = plane_count['Total'].clip(lower=0)

# Display the plane_count DataFrame
print(plane_count)




#%%

st. set_page_config(layout="wide")

st.title("Flights & Delays :)")

tab1, tab2, tab3, tab4 = st.tabs(["Home", "BCN", "ZRH", "Delay Predictions"])

with tab1:
    st.write("This interactive dashboard presents insights and predictions on bird strikes in the United States from 2003 to 2023, juxtaposed with the total number of flights during the same period. Explore trends, analyze data, and uncover insights to understand the dynamics of bird strikes in the United States.")

    st.header("User instructions:")   
    st.write("- Use the tabs to navigate between different pages.")
    st.write("- Hover over the visualizations to display extra information.")
    st.write("- Use the checkboxes, sliders and drop-down menus to filter the data shown in the visualizations.")
    
    st.write("The data was found on Kaggle (Aircraft Wildlife Strikes 1990-2023, U.S. Airline Traffic Data (2003-2023)) and is based on the Federal Aviation Administration (FAA) Wildlife Strike Database and the U.S. Department of Transportation’s (DOT) Bureau of Transportation Statistics.")

    st.write("This dashboard was created by Yannick Groot, Roosmarijn de Haan, Ditte de Lange and Shayan Mairuf.")
    
with tab2:
    st.write("Test")

    
    col1, col2 = st.columns(2)
    
    with col1:
    
# Adjust the width of the plot
        fig=plt.figure(figsize=(10, 6))  # Set the width to 10 inches and height to 6 inches

# Plot altitude in feet for each flight
        for flight_name, flight_df in zip(['Flight 1', 'Flight 2', 'Flight 3', 'Flight 4', 'Flight 5', 'Flight 6', 'Flight 7'],
                                          [flight1, flight2, flight3, flight4, flight5, flight6, flight7]):
            plt.plot(flight_df['Time (secs)'], flight_df['[3d Altitude Ft]'], label=f"{flight_name}")

        plt.xlabel('Time (secs)')
        plt.ylabel('Altitude (ft)')
        plt.title('Flight Profiles (Feet)')
        plt.legend()
        plt.grid(False)

# Show the plot
        plt.tight_layout()
        st.pyplot(fig)
    
    # Get the last value and max value of Time (secs) for each flight
    last_time_values = [flight['Time (secs)'].iloc[-1] for flight in [flight1, flight2, flight3, flight4, flight5, flight6, flight7]]
    max_time_values = [flight['Time (secs)'].max() for flight in [flight1, flight2, flight3, flight4, flight5, flight6, flight7]]

    with col2:

# Create a bar plot
        fig2=plt.figure(figsize=(10, 6))
        bars = plt.bar(range(1, 8), last_time_values, tick_label=['Flight 1', 'Flight 2', 'Flight 3', 'Flight 4', 'Flight 5', 'Flight 6', 'Flight 7'])

# Add annotations for max values
        for bar, max_time in zip(bars, max_time_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"Max: {max_time}", ha='center', va='bottom')

        plt.xlabel('Flight')
        plt.ylabel('Time (secs)')
        plt.title('Last Time (secs) and Max Time (secs) for Each Flight')

        st.pyplot(fig2)
        
    # Create a Folium map
# Create the first map
    map1 = folium.Map(location=[47, 5], zoom_start=5)

# Create a feature group for each flight in map1
    for flight, name, color in zip([flight1, flight2, flight3, flight4, flight5, flight6, flight7],
                                ['Flight 1', 'Flight 2', 'Flight 3', 'Flight 4', 'Flight 5', 'Flight 6', 'Flight 7'],
                                ['silver', 'cyan', 'peru', 'cadetblue', 'orchid', 'lawngreen', 'coral']):
        flight_path = list(zip(flight['[3d Latitude]'], flight['[3d Longitude]']))
        markers = [folium.CircleMarker(location=[lat, lon], radius=2, tooltip=f"Altitude: {alt} ft<br>Heading: {heading}<br>Speed: {speed} kts", fill=True, fill_color=color, fill_opacity=1, color=color) for lat, lon, alt, heading, speed in zip(flight['[3d Latitude]'], flight['[3d Longitude]'], flight['[3d Altitude Ft]'], flight['[3d Heading]'], flight['TRUE AIRSPEED (derived)'])]
        for marker in markers:
            marker.add_to(map1)
            folium.PolyLine(flight_path, color=color, weight=2.5, opacity=1, tooltip=name).add_to(map1)

# Create a feature group for map2
    feature_group_map2 = folium.FeatureGroup(name='Optimal Path')

# Plotting the optimal path as a red line on top of other paths in map2
    folium.PolyLine(locations=df_subset[['LAT', 'LON']].values, color='red').add_to(feature_group_map2)

# Plotting the optimal path waypoints as red circles in map2
    for index, row in df_subset.iterrows():
        folium.CircleMarker(location=[row['LAT'], row['LON']], radius=5, color='red', fill=True, fill_color='red', fill_opacity=1).add_to(feature_group_map2)

# Add the feature group for map2 to map1
    feature_group_map2.add_to(map1)

# Add a LayerControl to toggle the visibility of each flight path and map2
    folium.LayerControl().add_to(map1)

# Display the combined map
    st_data = st_folium(map1, height=500, width=1000)

    
   
with tab3:
    st.write("Zurich data")
    
# Convert Timestamps to numerical representation
    min_date = plane_count['Time'].min().timestamp()
    max_date = plane_count['Time'].max().timestamp()

# Create a Streamlit range slider for selecting the date range
    start_date, end_date = st.slider('Select Date Range',
                                     min_value=min_date,
                                     max_value=max_date,
                                     value=(min_date, max_date),
                                     format='YYYY-MM')

# Convert back to Timestamps
    start_date = pd.Timestamp(start_date, unit='s')
    end_date = pd.Timestamp(end_date, unit='s')

# Filter the DataFrame based on the selected date range
    filtered_data = plane_count[(plane_count['Time'] >= start_date) & (plane_count['Time'] <= end_date)]

# Plot the filtered data
    fig3 = plt.figure(figsize=(10, 6))
    plt.plot(filtered_data['Time'], filtered_data['Total'], color='blue', marker='o', markersize=4)
    plt.xlabel('Time')
    plt.ylabel('Total Count of Planes')
    plt.title('Total Count of Planes Over Time')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()

# Display the plot
    st.pyplot(fig3)
    
    # Define coordinates of Zurich airport
    zurich_coordinates = (47.4619, 8.5506)
  

# Function to create Map A
    def create_map_A():
    
    # Create a folium map centered around the mean latitude and longitude
        m = folium.Map(location=[airports['Latitude'].mean(), airports['Longitude'].mean()], zoom_start=2)

    # Add BCN marker to the map
        folium.Marker(location=zurich_coordinates, popup='ZRH').add_to(m)

    # Iterate over each airport to draw connections and add markers
        for index, row in airports.iterrows():
            airport_coordinates = (row['Latitude'], row['Longitude'])
        # Draw a line connecting Zurich airport and the current airport
            line = folium.PolyLine(locations=[zurich_coordinates, airport_coordinates], color='blue', weight=3, opacity=0.3)
            line.add_to(m)
        
        # Check if the airport is BCN (Barcelona–El Prat Airport)
            if row['IATA'] == 'ZRH':
                continue  # Skip BCN airport
        
        # Calculate great circle distance
            gcd = great_circle(airport_coordinates, zurich_coordinates).nautical
        # Create tooltip content
            tooltip_content = f"{row['IATA']} - ZRH ({gcd:.2f} NM)"
            line.add_child(folium.Tooltip(tooltip_content))  # Add tooltip to the line

        # Check if the airport is BCN (Barcelona–El Prat Airport)
            if row['IATA'] == 'ZRH':
            # Add a marker for BCN airport
                folium.Marker(location=airport_coordinates, 
                              popup=f"{row['Name']}", 
                              icon=folium.Icon(color='darkblue', icon='plane', angle=45)).add_to(m)
            else:
            # Add a tiny dot marker for other airports
                folium.CircleMarker(location=airport_coordinates, 
                                    radius=3, 
                                    color='red', 
                                    fill=True, 
                                    fill_color='red', 
                                    tooltip=f"{row['City']}, {row['Country']} ({row['IATA']}/{row['ICAO']})", 
                                    popup=f"{row['Name']}", 
                                    fill_opacity=0.6).add_to(m)

        return m

# Function to create Map B
    def create_map_B():
    # Create a folium map centered around the mean latitude and longitude
        m = folium.Map(location=[airports['Latitude'].mean(), airports['Longitude'].mean()], zoom_start=2)

# Find maximum and minimum movement values
        max_movements = airports['movements'].max()
        min_movements = airports['movements'].min()

# Add markers for each airport
        for index, row in airports.iterrows():
    # Calculate relative size based on movement values
            relative_size = (row['movements'] - min_movements) / (max_movements - min_movements) * 10
    
            folium.CircleMarker([row['Latitude'], row['Longitude']], 
                                radius=2 + relative_size,
                                tooltip=f"{row['City']}, {row['Country']} ({row['IATA']}/{row['ICAO']})",
                                popup=f"{row['Name']}, {row['City']}, {row['Country']}",
                                color='darkblue',
                                fill=True,
                                fill_color='darkblue',
                                fill_opacity=0.6).add_to(m)

        return m
    
    # Function to create Map C
    # Function to create Map C
    def create_map_C(selected_continent):
    # Create a folium map centered around the mean latitude and longitude
        m = folium.Map(location=[airports['Latitude'].mean(), airports['Longitude'].mean()], zoom_start=2)

    # Define color scheme for the ranges based on 'movements' column
        min_movements = airports['movements'].min()
        max_movements = airports['movements'].max()

    # Define the ranges
        ranges = [(1, 250), (250, 500), (500, 750), (750, 1000), (1000, 1500), 
              (1500, 2000), (2000, 2500), (2500, 3000), (3000, 5000), (5000, 10000), (10000, max_movements)]

    # Define color for each range
        color_scheme = {
            (1, 250): 'lightblue',
            (250, 500): 'lightgreen',
            (500, 750): 'green',
            (750, 1000): 'lightgreen',
            (1000, 1500): 'yellow',
            (1500, 2000): 'orange',
            (2000, 2500): 'red',
            (2500, 3000): 'darkred',
            (3000, 5000): 'purple',
            (5000, 10000): 'darkblue',
            (10000, max_movements): 'black'
            }

    # Dictionary to store markers for each range
        markers_by_range = {range_: [] for range_ in ranges}

    # Function to update the map based on the selected continent
        def update_map(continent):
            latitude, longitude, zoom = 20, 0, 2  # Default values for 'World'
            if continent == 'Africa':
                latitude, longitude, zoom = 0, 21.0938, 3
            elif continent == 'Asia':
                latitude, longitude, zoom = 25, 110.6197, 4
            elif continent == 'Europe':
                latitude, longitude, zoom = 54.5260, 15.2551, 4
            elif continent == 'North America':
                latitude, longitude, zoom = 54.5260, -105.2551, 3
            elif continent == 'Oceania':
                latitude, longitude, zoom = -30.2744, 140.7751, 4
            elif continent == 'South America':
                latitude, longitude, zoom = -20.2350, -51.9253, 3

        # Clear previous map and create a new one
            m = folium.Map(location=[latitude, longitude], zoom_start=zoom)

        # Add markers for each airport based on movement ranges
            for index, row in airports.iterrows():
                movements = row['movements']
                for range_ in ranges:
                    if range_[0] <= movements < range_[1]:
                        color = color_scheme[range_]
                        marker = folium.Marker([row['Latitude'], row['Longitude']],
                                               tooltip=f"{row['City']}, {row['Country']} ({row['IATA']}/{row['ICAO']})",
                                               popup=f"{row['Name']}, {row['City']}, {row['Country']}<br>Total movements: {movements}",
                                               icon=folium.Icon(icon='plane', color=color, angle=45))
                        markers_by_range[range_].append(marker)  # Add marker to the corresponding range
                        break

        # Add markers for each range to the map and to corresponding layers
            for range_, markers in markers_by_range.items():
                layer = folium.FeatureGroup(name=f'{range_[0]}-{range_[1]}', show=True)
                for marker in markers:
                    layer.add_child(marker)
                    m.add_child(layer)

        # Add layer control to the map
            folium.LayerControl().add_to(m)

            return m


    # Call update_map function with selected continent
        m = update_map(selected_continent)

    # Display the map
        st_folium(m, height=500, width=1000)



# Streamlit app
    
# Streamlit app
# Create a Streamlit selectbox
    selected_map = st.selectbox('Select Map', ['Route Network', 'Airport Movements Size', 'Movements by Continent'])

# Dictionary containing map creation functions
    map_functions = {
        'Route Network': create_map_A,
        'Airport Movements Size': create_map_B,
        'Movements by Continent': create_map_C
        }

# Call the selected map creation function
    selected_map_func = map_functions[selected_map]

    if selected_map == 'Movements by Continent':
        selected_continent = st.selectbox('Select Continent', ['World', 'Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America'], key='continent_select')
        # Call create_map_C() with the selected continent
        m = create_map_C(selected_continent)
        # Display the map
        st_folium(m, height=500, width=1000)
    else:
        m = selected_map_func()
    # Display the map
        st_folium(m, height=500, width=1000)
        
with tab4:
    st.write("Predictions")