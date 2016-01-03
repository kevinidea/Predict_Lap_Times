import pandas as pd

cars = pd.read_csv('cars_specifications_final.csv')

#Get HP
pattern = r"\d+\s*\w+\s*\((\d+).*"
#search for rows that have the pattern and not null
#condition = (cars['Power'].str.contains(pattern)) & ((cars['Power']).notnull())
#Get the horse power bhp inside (\d+) group
#str.extract return none for non-matching pattern or null already so no need to filter row
cars['HP']= cars['Power'].str.extract(pattern)

#Get Torque
pattern = r"\d+\s*\w+\s*\(([0-9]+).*"
#condition = (cars['Torque'].str.contains(pattern)) & ((cars['Torque']).notnull())
#Get the Torque in lb-ft inside (\[0-9]+)
cars["Torque (lb-ft)"] = cars['Torque'].str.extract(pattern)

#Fill missing car type with others
cars['Car Type'] = cars['Type'].fillna('Others')

#Get Weight
pattern = r"[0-9]+\s*[kKgG]+\s*\(([0-9]+)\s*\D+\)" #complete pattern with 1 group
#condition = (cars['Weight'].str.contains(pattern)) & ((cars['Weight']).notnull())
#Get the Weight in lbs inside ([0-9]+) group
cars["Weight (lbs)"]= cars['Weight'].str.extract(pattern)

#Get length, width, height from dimensions using 3 regular expression groups
pattern = r"\d+\.\d+\s*m\s*\((\d+)\s*in.+long.+\((\d+)\s*in.+wide.+\((\d+).+high.*"
#to be safe, condition is made to ensure the correct corresponding rows order
condition = (cars['Dimensions'].str.contains(pattern)) &((cars['Dimensions']).notnull())
#Get length, width, and height
cars.loc[condition, "Length (in)"] = cars['Dimensions'].str.extract(pattern)[0]
cars.loc[condition, "Width (in)"] = cars['Dimensions'].str.extract(pattern)[1]
cars.loc[condition, "Height (in)"] = cars['Dimensions'].str.extract(pattern)[2]

#Get Wheelbase
pattern = r".+\((\d+).+" #using the most simple pattern
#condition = (cars['Wheelbase'].str.contains(pattern)) & ((cars['Wheelbase']).notnull())
cars["Wheelbase (in)"] = cars['Wheelbase'].str.extract(pattern)

#Get insights from engine types
#Turbocharged or not
not_null = cars['Engine Type'].notnull()
#condition is required here to filter the rows with turbocharged engine
condition = (cars['Engine Type'].str.contains(r"([tT]urbo)|(charged)")) & not_null
cars.loc[condition,'Turbocharged']=1
cars['Turbocharged'] = cars['Turbocharged'].fillna(0)
#Diesel fuel or not
condition = (cars['Engine Type'].str.contains(r"[dD]iesel")) & not_null
cars.loc[condition, 'Diesel'] = 1
cars['Diesel'] = cars['Diesel'].fillna(0)
#Characterize engine types into 4 major categories
pattern_v_type = r"[vV]\s*\-*(\d+)"
pattern_inline_or_straight = r"[iI][nN][lL][iI][nN[eE]\s*\-*(\d+)|[iI]n\s*line\s*\-*(\d+)|[iI]\s*\-*(\d+)|[sS]\s*\-*(\d+)|[sS]traight\s*\-*(\d*)"
pattern_boxer_or_flat = r"([bB]oxer)|[fF]lat\s*\-*(\d+)|F\s*(\d+)"
pattern_rotary_or_wankel = r"R\s*(\d+)|[rR]otory|W(\d+)|[wW]ankel"
#Set 4 different conditions
v_type = (cars['Engine Type'].str.contains(pattern_v_type)) & not_null
inline_or_straight = (cars['Engine Type'].str.contains(pattern_inline_or_straight)) & not_null
boxer_or_flat = (cars['Engine Type'].str.contains(pattern_boxer_or_flat)) & not_null
rotary_or_wankel = (cars['Engine Type'].str.contains(pattern_rotary_or_wankel)) & not_null
#Consolidate these info into Engine column
cars.loc[not_null, 'Engine'] = 'Other Engine'
cars.loc[v_type,'Engine'] = "V-Type Engine"
cars.loc[inline_or_straight, 'Engine'] = "Inline/Straight Engine"
cars.loc[boxer_or_flat, 'Engine'] = "Boxer/Flat Engine"
cars.loc[rotary_or_wankel, 'Engine'] = "Wankel/Rotary Engine"

#Get number of gears from transmission
pattern1 = r"^([0-9])"
condition1 = (cars['Transmission'].str.contains(pattern1)) & (cars['Transmission'].notnull())
cars.loc[condition1, 'Gears'] = cars['Transmission'].str.extract(pattern1)
pattern2 = r"\D+([0-9])\s*\-*[sSgG][pe]"
condition2 = (cars['Transmission'].str.contains(pattern2)) & (cars['Transmission'].notnull())
cars.loc[condition2, 'Gears'] = cars['Transmission'].str.extract(pattern2)

#Automatic or Manual transmission
automatic = (cars['Transmission'].str.contains(r"[aA][uU][tT][oO]")) & (cars['Transmission'].notnull())
manual = (cars['Transmission'].str.contains(r"[mM]anual|[cC]lutch")) & (cars['Transmission'].notnull())
cars.loc[automatic, 'Transmission Type'] = 'Automatic'
cars.loc[manual, 'Transmission Type'] = 'Manual'

#Get engine displacement
pattern = r"(\d\.?\d?)\s*[lL]"
#condition = (cars['Displacement'].str.contains(pattern)) & (cars['Displacement'].notnull())
cars['Engine Displacement'] = cars['Displacement'].str.extract(pattern)

#Get Power to Liter ratio
pattern = r".+\((\d+)\s*"
#condition = (cars['Power Per Liter'].str.contains(pattern)) & (cars['Power Per Liter'].notnull())
cars['HP Per Liter'] = cars['Power Per Liter'].str.extract(pattern)

#Get Power to weight ratio
pattern = r".+\((\d+)\s*[bB][hH][pP]"
#condition = (cars['Power Per Weight'].str.contains(pattern)) & (cars['Power Per Weight'].notnull())
cars['HP Per Ton'] = cars['Power Per Weight'].str.extract(pattern)

#Get top speed
pattern = r".+\(([0-9]+)\s*[mM][pP][hH]"
#condition = (cars['Top Speed'].str.contains(pattern)) & (cars['Top Speed'].notnull())
cars['Top Speed (mph)'] = cars['Top Speed'].str.extract(pattern)

#Get quarter mile time
pattern = r"(\d+\.?\d*)" #minimal pattern
cars['Quarter Mile'] = cars['Quarter Mile Time'].str.extract(pattern)

cars.to_csv('cars_specifications_clean_v1.csv', index = False, encoding = 'UTF-8')







