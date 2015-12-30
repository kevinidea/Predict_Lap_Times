import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
from slugify import slugify #To replace all Latin letters with standard ASCII letters

###Get a list of all tracks from a track list page
url = "http://fastestlaps.com/tracks"
r = requests.get(url)
soup = BeautifulSoup(r.text)
#Extract a list of alphabets HTML
track_list_by_alphabet = soup.find_all('ul', attrs={'class':"fl-indexlist"})
tracks_name, tracks_length = [], []
#Extract a subset of tracks per alphabet
for alphabet in track_list_by_alphabet:
    tracks_list = alphabet.find_all('li')
    #Extract each track name
    for track in tracks_list:
        track_name = track.find('a').string.strip()
        #Modify each track name to match with their url using regular expression
        #Reno-Fernley Raceway Conf 2 (1.7m) => Reno-Fernley-Raceway-Conf-2-1-7m
        track_name = re.sub(r"\(|\)|\"",'',track_name) #Delete (,),"
        track_name = re.sub(r"\.| |\/",'-', track_name) #Replace .,space, / with -
        track_name = re.sub(r"\-+", '-', track_name) #Replace any multiple -- with -
        try: track_name.decode('ascii') #Replace Latin letters with standard ASCII letters if present
        except UnicodeEncodeError: track_name = slugify(track_name).title() #Capitalize each word
        #Rename 3 exceptional Nurburgring tracks to match their URLs
        if "Nurburgring-Nordschleife" in track_name:
            track_name = track_name.replace("Nurburgring-Nordschleife", 'Nordschleife')
        #Compile all the tracks into a list
        tracks_name.append(track_name)
        #Extract each track length if available
        try: tracks_length.append(track.contents[1])
        except: tracks_length.append('Unknown')

#Compile tracks list dataset
tracks_list_data = pd.DataFrame({'Track': tracks_name, 'Track Length': tracks_length})
#Save into a csv file
tracks_list_data.to_csv('Tracks_list_data.csv', index = False, encoding = 'UTF-8')


###Extract lap times for all tracks
lap_times_data = pd.DataFrame()
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
        position, car, driver, lap_time, hp, weight = [], [], [], [], [], []
        for row in rows:
            element = row.find_all('td')
            position.append(element[0].string)
            car.append(element[1].find('a').string)
            driver.append(element[2].string)
            lap_time.append(element[3].find('a').string)
            #split hp / kg = 312 / - to hp and kg plus replacing - with 0
            hp.append(element[4].string.split("/")[0].strip().replace('-','0'))
            weight.append(element[4].string.split("/")[1].strip().replace('-','0'))
        #Add a track name column
        tracks = [track]*len(rows)
        #Save all lap times per track
        track_lap_times_data = pd.DataFrame \
        ({"Position":position, "Car":car, "Driver":driver, "Lap Time":lap_time, "HP":hp, "Weight":weight, "Track":tracks})
        #Consolidate all lap times data into a dataframe
        lap_times_data = pd.concat([lap_times_data, track_lap_times_data], ignore_index = True )
    except: pass

#Save this dataframe into a CSV file
lap_times_data.to_csv('lap_times_data.csv', index=False, encoding ='UTF-8')
