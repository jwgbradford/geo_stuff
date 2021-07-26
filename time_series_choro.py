#%% vscode notebook cell delim
'''
References:
https://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/TimeSliderChoropleth.ipynb
https://geohackweek.github.io/ghw2018_web_portal_inlandwater_co2/InteractiveTimeSeries.html
'''
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime 
import numpy as np
#%% 'Study_midyear' to timestamp
def midyear_to_timestamp(key3):
    sec_in_year = (365*24*60*60)

    # 'Study_midyear' e.g. 2010.5
    yr = int(float(key3))
    mth_secs = (float(key3) - yr) * sec_in_year
    dt = datetime(yr, 1, 1)
    timestamp = dt.timestamp() + mth_secs
    dt = datetime.fromtimestamp(timestamp)
    timestamp = dt.timestamp()
    return timestamp

def timestamp_date(timestamp):
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime('%d/%m/%Y')
# %% initial df from csv file with 'date' col
#--------------------------------------------
group_cols = ['Study_number', 'Latitude', 'Longitude'] # site
csv_cols = list((*group_cols, 'Study_midyear', 'Rs_annual'))
df_base = pd.read_csv('srdb-data-V5.csv', usecols=csv_cols)[csv_cols].dropna().astype(str)

#%% convert 'Study_midyear' to datetime
df_base['timestamp'] =  df_base['Study_midyear'].apply(midyear_to_timestamp)
df_base['date'] =  df_base['timestamp'].apply(timestamp_date)

df_base['Rs_annual'] = df_base['Rs_annual'].astype('float64')
df_base['timestamp'] = df_base['timestamp'].astype('int64')
df_base['date'] = pd.to_datetime(df_base['date'])

#%% Sort by site and date
sort_cols = list((*group_cols, 'date'))
df_base.sort_values(sort_cols, inplace=True, ignore_index=True)
df_base.head(10)



#%%
#--------------------------------------------------------
grouped = df_base.groupby('Study_number')
#len(grouped)
time_series_data = {}
max_study_midyears = [None, None]
for idx, (name, group) in enumerate(grouped):
    # dump first and last
    if False and (idx == 0 or idx == (len(grouped) - 1)):
        print(f"idx={idx}, name={name}")
    
    grouped2 = group.groupby(['Latitude', 'Longitude'])
    #print(len(grouped2))
    coords = {}
    for idx2, (name2, group2) in enumerate(grouped2):
        if False and (idx2 == 0 or (idx2 > 0 and idx2 == (len(grouped2) - 1))):
            print(f"\tidx2={idx2}, name2={name2}")

        grouped3= group.groupby('Study_midyear')
        #print(len(grouped3))
        study_midyears = {}
        for idx3, (name3, group3) in enumerate(grouped3):
            if False and (idx3 == 0 or (idx3 > 0 and idx3 == (len(grouped3) - 1))):
                print(f"\t\tidx3={idx3}, name3={name3}, group3['Rs_annual']={group3['Rs_annual'].tolist()} ")
            study_midyears[name3] = group3['Rs_annual'].tolist()
        coords[name2] = study_midyears
        if max_study_midyears[0] == None or \
            (len(study_midyears) > len(time_series_data[max_study_midyears[0]][max_study_midyears[1]])):
            max_study_midyears = [name, name2]
    time_series_data[name] = coords

# %%
if True:
    for idx, key in enumerate(time_series_data):
        print(f"{key}")

        for idx2, key2 in enumerate(time_series_data[key]):
            print(f"\t{key2}")

            for idx3, key3 in enumerate(time_series_data[key][key2]):
                print(f"\t\t{key3}")

                print(f"\t\ttime_series_data[{key}][{key2}][{key3}] = {time_series_data[key][key2][key3]}")

#%% 
def coord_to_polygon(lon, lat):
    # create polygon (square) from lon, lat string params
    lon1 = str(float(lon)+0.2)
    lat1 = str(float(lat)+0.1)
    res = [
        [lon,   lat], # bottom left
        [lon1, lat], # bottom right
        [lon1, lat1], # top right
        [lon,   lat1],    # top left
        [lon,   lat] # back to bottom left
    ]
    return res
#%%
# https://geoffboeing.com/2015/10/exporting-python-data-geojson/
def df_to_geojson(df, properties, lat='Latitude', lon='Longitude'):
    geojson = {'type':'FeatureCollection', 'features':[]}
    for index, row in df.iterrows():
        feature = {'id':index,
                   'type':'Feature',
                   'properties':{},
                   'geometry':{'type':'Polygon',
                               'coordinates':[]}}
        feature['geometry']['coordinates'] = [
            coord_to_polygon(row[lon],row[lat])
        ]
        for prop in properties:
            feature['properties'][prop] = row[prop]
        geojson['features'].append(feature)
    return geojson

# %%
# Find study_number that has one (lat, lng), foreach date use first value
from datetime import datetime 
vals = []
study_series_data = {}
sec_in_year = (365*24*60*60)
gdf_cols = ['Study_number', 'Latitude', 'Longitude']
gdf = pd.DataFrame(columns=gdf_cols)
#for key in time_series_data:
keys = list(time_series_data.keys())
keys = [max_study_midyears[0]]  ## just specific study_number
for key in keys:

    # 'Study_number'
    key2s = list(time_series_data[key]) # ['Latitude', 'Longitude']
    key2s = [max_study_midyears[1]] # just specific ['Latitude', 'Longitude']

    key2 = key2s[0] ## first

    study_midyears = {}
    for key3 in time_series_data[key][key2]:
        # 'Study_midyear' e.g. 2010.5
        yr = int(float(key3))
        mth_secs = (float(key3) - yr) * sec_in_year
        dt = datetime(yr, 1, 1)
        timestamp = dt.timestamp() + mth_secs
        dt = datetime.fromtimestamp(timestamp)
        timestamp = dt.timestamp()

        val = float(time_series_data[key][key2][key3][0]) ## first
        vals.append(val)
        study_midyears[str(timestamp)] = val
        #print(f"\t\ttime_series_data[{key}][{key2}][{key3} {dt.strftime('%d/%m/%Y')}][0] = {val} ")
    
    row = [key, key2[0], key2[1]]
    gdf.loc[len(gdf)] = row
    study_series_data[key] = study_midyears

#%%
gdf2 = gdf.set_index('Study_number')
geo_json = df_to_geojson(gdf2, [])
#%%
#print(study_series_data)
gdf2.head()




#%%
print(geo_json['features'][0])
#%%
#https://stackoverflow.com/questions/20011494/plot-normal-distribution-with-matplotlib

import scipy.stats as stats
from statistics import mode
import math

vals.sort()
print(f"len={len(vals)}, min={vals[0]}, max={vals[-1]}, mean={np.mean(vals)}, median={np.median(vals)}") #  mode={mode(vals)}, 

#%%
h = vals
h.sort()
hmean = np.mean(h)
hstd = np.std(h)
pdf = stats.norm.pdf(h, hmean, hstd)
plt.plot(h, pdf) # including h here is crucial

#%%
if False:
    # https://stackoverflow.com/questions/33203645/how-to-plot-a-histogram-using-matplotlib-in-python-with-a-list-of-data
    x = np.array(vals)
    q25, q75 = np.percentile(x,[.25,.75])
    bin_width = 2*(q75 - q25)*len(x)**(-1/3)
    bins = round((x.max() - x.min())/bin_width)
    #print("Freedman–Diaconis number of bins:", bins)
    plt.hist(x, bins = bins)

# %%
from branca.colormap import linear

cmap = linear.PuRd_09.scale(vals[0], vals[-1])

style_dict = {}
for key in study_series_data.keys():
    # 'Study_number'
    timestamp_dict = {}
    for key2 in study_series_data[key]:
        # timestamp
        color_dict = {}
        val = study_series_data[key][key2]
        color_dict['color'] = cmap(val)
        color_dict['opacity'] = 1.0
        timestamp_dict[key2] = color_dict
    style_dict[key] = timestamp_dict

# %%
import folium
from folium.plugins import TimeSliderChoropleth


m = folium.Map([0, 0], tiles="Stamen Toner", zoom_start=2)

g = TimeSliderChoropleth(
    geo_json,
    styledict=style_dict,
).add_to(m)

m.save('time_series_choro.html')

m

# %%
