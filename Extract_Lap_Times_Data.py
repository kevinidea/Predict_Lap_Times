import pandas as pd
from bs4 import BeautifulSoup
import requests

#Extract Nürburgring Nordschleife lap times
url = "http://fastestlaps.com/tracks/nordschleife"
r = requests.get(url)
#Extract the HTML page
soup = BeautifulSoup(r.text)
#Extract the table within the HMTL page
table = soup.find('table', attrs={'class':"table table-striped fl-laptimes-trackpage"})
#Extract all rows in the table, skip the header row
rows = table.find_all('tr')[1:]
#Extract all data from each row
position, car, driver, lap_time, hp_kg = [], [], [], [], []
for row in rows:
    element = row.find_all('td')
    position = position.append(element[0].string)
    car = car.append(element[1].find('a').string)
    driver = driver.append(element[2].string)
    lap_time = lap_time.append(element[3].find('a').string)
    hp_kg = hp_kg.append(element[4].string)

#Add some more variables
number_of_rows = len(rows)
track = ['nordschleife']*number_of_rows
number_of_corners = [154]*number_of_rows
