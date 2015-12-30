import pandas as pd
from bs4 import BeautifulSoup
import requests

###Get a list of all tracks from previous step (Extract_Lap_Times_Data.py)
tracks_name = pd.read_csv('Tracks_list_data_final.csv')['Track']

####Extract a unique list of cars that have lap times
cars_list = pd.DataFrame()
for track in tracks_name:
    track_url = "http://fastestlaps.com/tracks/" + track
    r = requests.get(track_url)
    #Extract the HTML page
    soup = BeautifulSoup(r.text)
    #Extract one main table within the HMTL page
    table = soup.find('table', attrs={'class':"table table-striped fl-laptimes-trackpage"})
    try: # Skip the track that has no lap time data
        #Extract all rows in the table, skip the header row
        rows = table.find_all('tr')[1:]
        #Extract all data from each row
        cars, cars_url = [], []
        for row in rows:
            #car is in the second 'td' column
            cars.append(row.find_all('td')[1].find('a').string)
            #car url is inside <a href=url> car </a>
            car_url = 'www.fastestlaps.com' + row.find_all('td')[1].find('a')['href']
            cars_url.append(car_url)

        #Save all cars from each track (inside each track loop)
        cars_list_per_track = pd.DataFrame ({'Car':cars, 'Car URL': cars_url})
        #Consolidate all cars into a dataframe
        cars_list = pd.concat([cars_list, cars_list_per_track], ignore_index = True)
    except: pass

#De-duplicate the cars_list
unique_cars_list = cars_list.drop_duplicates(inplace = False)
#Save this unique cars list into a csv file
unique_cars_list.to_csv('unique_cars_list.csv', index = False, encoding = 'UTF-8')

