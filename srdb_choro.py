#%% vscode notebook cell delim
'''
References:
https://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/TimeSliderChoropleth.ipynb
https://geohackweek.github.io/ghw2018_web_portal_inlandwater_co2/InteractiveTimeSeries.html
'''
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime 

#%matplotlib inline
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

#%% group on site ('Study_number', 'Latitude', 'Longitude')
''' Some 'Study_number' have multiple very close sites with 
consequtive 'Study_midyear' values so are probably intended to
be part of the same site series. Ignore for now.
'''
grouped = df_base.groupby(group_cols) # e.g. index ('4391', '2.97', '102.3')

#%% Display groups to eyeball look ok
group_keys = list(grouped.groups)
print(f"group_keys len={len(group_keys)}, first={group_keys[0]}, last={group_keys[-1]}") 

#%%
grouped.get_group(group_keys[0])
#%%
grouped.get_group(group_keys[-1])

#%% Create df2_grouped from groups and sort datestamps within groups 
''' For now, simply ignore groups that have repeated dates
'''
df2_grouped = None
for idx, (name, group) in enumerate(grouped):
    midyears = group['Study_midyear']
    uniq = midyears.sort_values().unique()
    if len(uniq) == len(midyears):
        # all midyears unique
        #print(f"uniq len={len(uniq)}, {uniq}")
        group_copied = group.copy()

        group_copied['timestamp'] =  group_copied['Study_midyear'].apply(midyear_to_timestamp)

        group_copied['date'] =  group_copied['timestamp'].apply(timestamp_date)
        group_copied['date'] = pd.to_datetime(group_copied['date'])

        group_copied.sort_values(by=['timestamp'], inplace=True, ignore_index=True)
        if df2_grouped is None:
            df2_grouped = group_copied
        else:
            df2_grouped = df2_grouped.append(group_copied)
#%% df2_grouped index is tuple ('Study_number', 'Latitude', 'Longitude')
df2_grouped.reset_index(drop=True, inplace=True)
df2_grouped.sort_values(group_cols, inplace=True, ignore_index=True)

#df2_grouped.sort_values(group_cols, inplace=True, ignore_index=True)
#df2_grouped.set_index(group_cols, inplace=True)

#%%
df2_grouped.head(10)

#%%
records_init = len(df_base)
records_remain = len(df2_grouped)
records_discarded = records_init - records_remain
print(f"discarded {records_discarded} records due to repeated 'Study_midyear' in group, " + 
            f"{records_remain} ({(float(records_remain) / float(records_init))*100:.1f} %) records remain")

#%% 'Rs_annual' ranges 
#---------------------
grouped = df2_grouped.groupby(group_cols)
df3_val = grouped.agg({'Rs_annual': ['min', 'max', 'count']})
df3_val.columns = [col[1] for col in df3_val.columns]  # remove 'Rs_annual' outer level col
df3_val.sort_values('count', inplace=True, ascending=False)
df3_val.head()

#%%
''' only sites those with multiple timestamp count. This constricts number of 
map markers to those with plenty of history
'''
group_count_min = 7
if True:
    df3_val = df3_val[ df3_val['count'] > group_count_min ]
    print(f"Number of sites with timestamp series length greater than " +
                f"{group_count_min} = {len(df3_val)}")

#%% # restrict number sites when DBG for simplicity
vals_len = len(df3_val)
vals_len = 1    # restrict number of sites

df3_val = df3_val.iloc[0:vals_len]
df3_val.head()

#%% 
# By now we df3_val for the index of groups to map, and df2_grouped for the 
# values to use
#------------------------

# %% Colour map
from branca.colormap import linear

val_max = df3_val['max'].max()
val_min = df3_val['min'].min()
cmap = linear.PuRd_09.scale(val_min, val_max)

print(f"val_max={val_max}, val_min={val_min}")

#%% Create df of all groups matching df3_val index

df4_filtered = pd.DataFrame()
grouped = df2_grouped.groupby(group_cols)

for i, (index, row) in enumerate(df3_val.iterrows()):
    df_group = grouped.get_group(index).copy()
    df_group['site'] = '_'.join(index)  # string from tuple, e.g. '3053_36.13_137.42'
    df4_filtered = df4_filtered.append(df_group)

df4_filtered['colour'] = df4_filtered['Rs_annual'].apply(cmap)
df4_filtered.reset_index(drop=True, inplace=True)

#%%
df4_filtered.head()

#%%
records_remain2 = len(df4_filtered)    
records_discarded = records_remain - records_remain2
print(f"discarded {records_discarded} records due to group_count={group_count_min} " + 
        f"and vals_len={vals_len} filtering")
print(f"remaining number of records = {records_remain2} ({(float(records_remain2) / float(records_init))*100:.1f}%)")

#%%
# Plot 'Rs_annual' of first group to check color map

grouped = df4_filtered.groupby(group_cols)
for idx, (group_name, df_group) in enumerate(grouped):
    if idx > 0:
        break # just do first one
    title = '_'.join(group_name) # create string from tuple
    colours = df_group['colour'].tolist()
    df_group[['date', 'Rs_annual']].plot(x='date', y='Rs_annual', kind='bar', 
                title=title, color=colours)

#%% df to geojson, df4_filtered contains all the sites to display
''' 
folium.TimeSliderChoropleth() does not work with geojson 'Point' geometry.
Instead, create a (square) 'Polygon', see: 
https://geoffboeing.com/2015/10/exporting-python-data-geojson/
'''
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
def df_to_geojson(df, properties=[], lat_col='Latitude', 
                                        lon_col='Longitude'):
    geojson = {'type':'FeatureCollection', 'features':[]}
    for index, row in df.iterrows():
        feature = {'id':index,
                   'type':'Feature',
                   'properties':{},
                   'geometry':{'type':'Polygon',
                               'coordinates':[]}}
        feature['geometry']['coordinates'] = [
            coord_to_polygon(row[lon_col],row[lat_col])
        ]
        for prop in properties:
            feature['properties'][prop] = row[prop]
        geojson['features'].append(feature)
    return geojson

# %% df from groupby
'''
Create a df indexed by 'site' for use by geojson
'''
site_cols = list((*group_cols, 'site'))
df5_geojson = df4_filtered[site_cols].copy()

df5_geojson['count'] = 1 # temp val
df5_geojson = df5_geojson.groupby(site_cols).count() # count will be len of each col

df5_geojson.reset_index(inplace=True)
df5_geojson.set_index('site', inplace=True)

df5_geojson.head()

#%% Convert df to json
geo_json = df_to_geojson(df5_geojson, ['Study_number'])
features_len = len(geo_json['features'])
print(f"geo_json features len={features_len}, last feature={geo_json['features'][-1]}")

#%%
'''
Format of styledict:
'site' col used as index in df5_geojson which is then assigned 
to geojson 'features'.'id'
{<site> : {
    <timestamp> : {
        color : <cmap derived from Rs_annual>,
        opacity : 0.5
    }
},

'''
style_dict = {}
grouped = df4_filtered.groupby('site')

for group_name, df_group in grouped:
    timestamp_dict = {}
    for index, row in df_group.iterrows():
        color_dict = {}
        color_dict['color'] = row['colour']
        color_dict['opacity'] = 0.7
        timestamp_dict[f"{row['timestamp']:.0f}"] = color_dict
    style_dict[group_name] = timestamp_dict

for idx, key in enumerate(style_dict):
    if idx > 0:
        break   # just display first one
    for idx2, key2 in enumerate(style_dict[key]): 
        if idx2 > 0:
            break   # just display first one
        print(f"style_dict[{key}][{key2}] = {style_dict[key][key2]}")
# %%
import folium
from folium.plugins import TimeSliderChoropleth

m = folium.Map([0, 0], tiles="Stamen Toner", zoom_start=2)
g = TimeSliderChoropleth(
    geo_json,
    styledict=style_dict,
).add_to(m)

m.save('srdb_choro.html')
m

# %%
