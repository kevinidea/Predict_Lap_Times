import pandas as pd
from bs4 import BeautifulSoup
import requests

###Get all car manufacturer names from the main website
#Go to the main website
url = 'http://www.myquartermile.com/'
#Scrape the HTML page
r = requests.get(url)
#Turn HTML into BeautifulSoup object
soup = BeautifulSoup(r.text)
car_brands_html = soup.find_all('a', attrs={'class':'readmore'})
car_brands = [car.string for car in car_brands_html]
#exclude the last 2 missing names
car_brands = car_brands[:-2]


###Get quarter mile time data from all car manufacturers and save it into a dataframe and output to CSV file
quarter_mile_data = pd.DataFrame()
#get all URLs for car manufacturers
car_brands_url = ['http://myquartermile.com/qtrmile.php?make=' + car_brand for car_brand in car_brands]
#loop through each HTML page per car brand
for car_brand_url in car_brands_url:
    r2 = requests.get(car_brand_url)
    html_page = BeautifulSoup(r2.text)
    #examine the HTML closely and find pattern to extract the right table
    table = html_page.find('table', attrs={'width':'100%', 'border':'0','align':'center'})
    #print table.prettify() 
    rows = table.find_all('tr')[5:-1] #include only the relevant tr rows
    year, make, model, time0to60, quarter_mile_time, comment =[], [], [], [], [], []
    #extract all data from each row
    for row in rows:
        #print row.prettify() #prettify works on each element only
        #extract single data from each column
        column = row.find_all('td')
        #try & except handle null value
        try: year.append(column[0].string.strip())
        except: year.append(None)
        try: make.append(column[1].string.strip())
        except: make.append(None)
        try: model.append(column[2].string.strip())
        except: model.append(None)
        try: time0to60.append(column[3].string.strip())
        except: time0to60.append(None)
        try: quarter_mile_time.append(column[4].string.strip())
        except: quarter_mile_time.append(None)
        try: comment.append(column[5].string.strip()) 
        except: comment.append(None)
        
    #Save 5 lists of data from each car brand into a dataframe using dictionary
    each_quarter_mile_data = pd.DataFrame \
    ({'Year':year, 'Make':make, 'Model':model, "0-60 Time":time0to60, "1/4 Mile Time":quarter_mile_time, 'Comment':comment})
    #Continue appending new data into a consolidated dataframe, ignoring index to avoid duplicate index
    quarter_mile_data = quarter_mile_data.append(each_quarter_mile_data, ignore_index=True)
    
#Save all data into a csv file
quarter_mile_data.to_csv('quarter_mile_data.csv', index=False)



