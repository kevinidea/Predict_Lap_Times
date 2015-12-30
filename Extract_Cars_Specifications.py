import pandas as pd
from bs4 import BeautifulSoup
import requests
from collections import OrderedDict

###Get a list of all unique cars from previous step (Extract_Unique_Cars_List.py)
cars = pd.read_csv('unique_cars_list_final.csv')


###Extract specifications for all unique cars
search_texts = ['Car', 'Power', 'Torque', 'Car type', 'Curb weight', 'Dimensions', 'Wheelbase', \
"Power / weight", 'Introduced', 'Origin country', 'Engine type', 'Displacement', \
"Power / liter", 'Transmission', 'Layout', 'Top speed', "1/4 mile", 'car URL']

#Make an ordered list of tuples to create a dictionary with ordered keys
car_variables = [ ('Car',[]), ('Power', []), ('Torque', []), ('Type',[]), ('Weight', []), ('Dimensions', []), \
('Wheelbase',[]), ('Power Per Weight', []), ('Year Model', []), ('Country', []), ('Engine Type', []), ('Displacement', []), \
('Power Per Liter', []), ('Transmission', []), ('Layout', []), ('Top Speed', []), ('Quarter Mile Time', []), ('Car URL', [])]
car_specs = OrderedDict(car_variables)

for i in cars.index.values:
    #Go to each car specs HTML page
    car_url = "http://" + cars.iloc[i, 1]
    r = requests.get(car_url)
    soup = BeautifulSoup(r.text)
    #Extract each car specs
    car_specs['Car'].append(cars.iloc[i,0])
    car_specs['Car URL'].append(car_url)
    for idx in range(1,3): #Extract power and torque info
        try: car_specs[car_specs.keys()[idx]].append(soup.find(text=search_texts[idx]).findNext('td').find('a').string)
        except: car_specs[car_specs.keys()[idx]].append(None) #Return null if blank
    for idx in range(3, len(car_variables)-1): #Extract info for everything else
        try: car_specs[car_specs.keys()[idx]].append(soup.find(text=search_texts[idx]).findNext('td').string)
        except: car_specs[car_specs.keys()[idx]].append(None) #Return null if blank

#Transform the dictionary into a dataframe
car_specs = pd.DataFrame(car_specs)
#Save the dataframe into a CSV file
car_specs.to_csv('cars_specifications.csv', index=False, encoding = 'UTF-8')