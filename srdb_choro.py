#%% vscode notebook cell delim
'''
References:
https://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/TimeSliderChoropleth.ipynb
https://geohackweek.github.io/ghw2018_web_portal_inlandwater_co2/InteractiveTimeSeries.html
'''

#%matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from datetime import datetime 

#%%
DBG = False # set True for debug
vals_len_dbg = None  # set to count of sites to map, if not set then all sites
group_count_min = 4 # set to minimun number of records per site (current max is 13)
outfile='srdb_choro_out' # file prefix for outputs

if DBG:
    vals_len_dbg = 1

#%% Test if run in notebook
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

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
csv_cols = group_cols + ['Country', 'Region', 'Study_midyear', 'Rs_annual']

df_base = pd.read_csv('srdb-data-V5.csv', usecols=csv_cols)[csv_cols].dropna().astype(str)

#%% convert 'Study_midyear' to datetime
df_base['timestamp'] =  df_base['Study_midyear'].apply(midyear_to_timestamp)
df_base['date'] =  df_base['timestamp'].apply(timestamp_date)

df_base['Rs_annual'] = df_base['Rs_annual'].astype('float64')
df_base['timestamp'] = df_base['timestamp'].astype('int64')
df_base['date'] = pd.to_datetime(df_base['date'])

#%% sort within groups on date
df_base.sort_values(group_cols + ['date'], inplace=True, ignore_index=True)
df_base.head(5)

#%% group on site ('Study_number', 'Latitude', 'Longitude')
''' Some 'Study_number' have multiple very close sites with 
consequtive 'Study_midyear' values so are probably intended to
be part of the same site series. Ignore for now.
'''
grouped = df_base.groupby(group_cols) # e.g. index ('4391', '2.97', '102.3')

#%% Display groups to eyeball look ok
if DBG:
    group_keys = list(grouped.groups)
    print(f"group_keys len={len(group_keys)}, first={group_keys[0]}, last={group_keys[-1]}") 
    print(f"first get_group({group_keys[0]}) = {grouped.get_group(group_keys[0])}")
    print(f"last get_group({group_keys[-1]}) = {grouped.get_group(group_keys[-1])}")

#%% Rs_annual aggregate
df_agg = grouped.agg({'Rs_annual': ['min', 'max', 'mean'], \
            'date' : ['min', 'max', 'count', 'nunique']})
df_agg.head()

#%% For now, simply ignore groups that have repeated dates
df_agg = df_agg[ df_agg[('date', 'count')] == df_agg[('date', 'nunique')] ]

#%% Filter out those sites without sufficient history
df_agg = df_agg[ df_agg[('date', 'count')] >= group_count_min ]

#%% remove outliers, those with a mean that has a standard deviation greater than 3sigma 
# https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
df_outliers = df_agg[ (np.abs(stats.zscore(df_agg[('Rs_annual', 'mean')])) > 3) ]
df_outliers

print(f"remove outliers {df_agg[ (np.abs(stats.zscore(df_agg[('Rs_annual', 'mean')])) >= 3) ].index}")

df_agg = df_agg.drop(index=df_outliers.index)

#%% Sort groups by count
df_agg.sort_values(('date', 'count'), inplace=True, ascending=False)

#%% Possibly reduce number of sites if DBG
if vals_len_dbg != None: 
    # restrict number sites when DBG for simplicity
    if vals_len_dbg < len(df_agg):
        df_agg = df_agg.iloc[0:vals_len_dbg]    # first n rows
df_agg.head()

#%%
val_max = df_agg[('Rs_annual', 'max')].max()
val_min = df_agg[('Rs_annual', 'min')].min()
print(f"val_max={val_max}, val_min={val_min}")

#%%
records_init = len(df_base)
records_remain = df_agg[('date', 'count')].sum()
records_discarded = records_init - records_remain
print(f"discarded {records_discarded} records, sites remain={len(df_agg)}" + \
        f", records remain={records_remain}" + \
        f" ({(float(records_remain) / float(records_init))*100:.1f} %)")

#%% Obsolete code to remove groups with repeated date
'''
#----------------------------------------------------------
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
#----------------------------------------
'''

#%% Use df_agg index to filter original df
df4_filtered = df_base.copy()
df4_filtered.set_index(group_cols, inplace=True)
df4_filtered = df4_filtered.loc[df_agg.index] # use df_agg index which is sorted by 'count'
df4_filtered.reset_index(inplace=True)

df4_filtered['site'] = df4_filtered['Study_number'] + '_' + \
                    	df4_filtered['Latitude'] + '_' + \
                        df4_filtered['Longitude']
df4_filtered

#%% histogram (value frequency) to decide colourmap step
df4_filtered[['Rs_annual']].hist()

#%% 
'''
# By now we have df_agg for the index of groups to map and stats of vals, 
# and df4_filtered for the list of vals to use
# ------------------------
'''
# %% Colour map
from branca.colormap import linear

# Tried PuRd_09 range but low values indistinct. Instead use a rainbow range 
cmap = linear.Spectral_11.scale(val_min, val_max) # PuRd_09

#%% Avoid the few very large values skewing the linear colourmap
# Use quantiles to equalise the number of values in each colour step
cmap_step=11
cmap = cmap.to_step(data=df4_filtered['Rs_annual'], n=cmap_step) # , method='quantiles')
cmap

#%%
df4_filtered['colour'] = df4_filtered['Rs_annual'].apply(cmap)

#%%
# Plot 'Rs_annual' of first group to check color map

grouped = df4_filtered.groupby(group_cols)
df_group = grouped.get_group(df_agg.index[0])  
colours = df_group['colour'].tolist()
site = df_group['site'].tolist()[0]
df_group[['date', 'Rs_annual']].plot(x='date', y='Rs_annual', kind='bar', 
                title=site, color=colours)

#%% Plot a subset of groups against a colourmap background for check
fig, ax = plt.subplots()

for idx, (name, group) in enumerate(grouped):
    if idx > 5:
        break   # plot first 5
    site = group['site'].tolist()[0]
    plt.plot(group['date'], group['Rs_annual'], 'x-', label=site)

plt.legend(title='site', 
            bbox_to_anchor=(1, 1),
            bbox_transform=plt.gcf().transFigure)

cmap_bucket = (val_max - val_min)/cmap_step
for i in range(int(val_min), int(val_max), int(cmap_bucket)):
    ax.axhspan(i, i+cmap_bucket, facecolor=cmap(i+0.5*cmap_bucket), alpha=0.7)
plt.title('Rs_annual against colourmap for subset sites')

if is_interactive():
    plt.show()
else:
    outf = f"{outfile}_colours.png"
    print(f"Saving file {outf}")
    plt.savefig(outf)

#%% df to geojson, df4_filtered contains all the sites to display
''' 
folium.TimeSliderChoropleth() does not work with geojson 'Point' geometry.
Instead, create a (square) 'Polygon', see: 
https://geoffboeing.com/2015/10/exporting-python-data-geojson/
'''
def coord_to_geometry(lon, lat):
    # create polygon (square) from lon, lat centre
    top_right = [str(float(lon)+0.1), str(float(lat)+0.05)]
    bottom_left = [str(float(lon)-0.1), str(float(lat)-0.05)]

    coords = [
        bottom_left,
        [top_right[0], bottom_left[1]], # bottom right
        top_right,
        [bottom_left[0], top_right[1]],    # top left
        bottom_left # back to bottom left
    ]
    res={}
    if False:
        res = {'type' : 'Point', 'coordinates': coords[0]}   # DBG
    else:
        res = {'type' : 'Polygon', 'coordinates': [coords]}
    return res
#%% 
def df_to_geojson(df, properties=[], lat_col='Latitude', 
                                        lon_col='Longitude'):
    geojson = {'type':'FeatureCollection', 'features':[]}
    for index, row in df.iterrows():
        feature = {'id':index,
                   'type':'Feature',
                   'properties':{},
                   'geometry': coord_to_geometry(row[lon_col],row[lat_col])}
        for prop in properties:
            feature['properties'][prop] = row[prop]
        geojson['features'].append(feature)
    return geojson

# %% Create a geojson df indexed by grouby order
properties_cols = ['site', 'Country', 'Region']  # extra info in geojson
df5_geojson = df4_filtered[group_cols + properties_cols].copy()

df5_geojson = df5_geojson.groupby(group_cols).first()
df5_geojson = df5_geojson.loc[df_agg.index] # reorder to match df_agg

val_cols = [('Rs_annual', 'min'), ('Rs_annual', 'max'), 
            ('Rs_annual', 'mean'),  ('date', 'count'),
            ('date', 'min'), ('date', 'max'), ]
prop_val_cols = ['min',  'max', 
            'mean', 'count',
            'start_date', 'end_date']

df5_geojson[prop_val_cols] = df_agg[val_cols].astype(str)
df5_geojson['mean'] = df5_geojson['mean'].astype('float64').apply(lambda x: f"{x:.1f}") # one decimal place
properties_cols = properties_cols + prop_val_cols

df5_geojson.reset_index(inplace=True)
df5_geojson.head()

#%% Convert to json
geo_json = df_to_geojson(df5_geojson, properties_cols)

#%% sanity check geo_json
if DBG:
    features_len = len(geo_json['features'])
    print(f"geo_json features len={features_len}, features[0]={geo_json['features'][0]}")
    for idx, feature in enumerate (geo_json['features']):
        p = geo_json['features'][idx]['properties']
        print(f"features id={feature['id']}, site={p['site']}, location={p['Region']}, {p['Country']}")

#%% stle_dict is site colour history
#---------------------------------------------
'''
Format of styledict:
<idx> is group index in df5_geojson which is assigned 
to geojson 'features'.'id' key
{<idx> : {
    <timestamp> : {
        color : <cmap derived from Rs_annual>,
        opacity : 0.5
    }
},
Note: choropleth timeslider is out of order if timestamp is string type. It is
the correct order if numeric type, however the map does not show the site
'''
style_dict = {}
grouped = df4_filtered.groupby(group_cols)

for idx, (group_name, df_group) in enumerate(grouped):
    timestamp_dict = {}
    for row_index, row in df_group.iterrows():
        color_dict = {}
        color_dict['color'] = row['colour']
        color_dict['opacity'] = 0.7
        timestamp_dict[f"{row['timestamp']:.0f}"] = color_dict
        #timestamp_dict[row['timestamp']] = color_dict # DBG
    style_dict[idx] = timestamp_dict

if DBG:
    for idx, key in enumerate(style_dict):
        if idx > 0:
            break   # just display first one
        for idx2, key2 in enumerate(style_dict[key]): 
            if idx2 > 0:
                break   # just display first one
            print(f"style_dict[{key}][{key2}] = {style_dict[key][key2]}")

# %% ref: https://stackoverflow.com/questions/53898733/how-to-customize-the-layercontrol-in-folium
import folium
from folium.plugins import TimeSliderChoropleth, MarkerCluster

#%% map marker popup text using html <br> for embedded newline
def map_markers(df, m, prop_cols=[]):
    for index, row in df.iterrows():
        props = ',<br> '.join((f"{i} = {row[i]}" for i in prop_cols))
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"id = {index},<br>{props}").add_to(m)


#%% Generate map with multiple layers
first_row = df5_geojson.iloc[0]
map = folium.Map([first_row['Latitude'], first_row['Longitude']], 
                                zoom_start=2)
folium.TileLayer('openstreetmap').add_to(map)
folium.TileLayer('Stamen Terrain').add_to(map) # terrain
folium.TileLayer('Stamen Toner').add_to(map)  # plain map good for showing colourmap
tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(map)    # Satetlite

fg=folium.FeatureGroup(name='Rs_annual choropleth', show=True)
map.add_child(fg)
choro_cluster = MarkerCluster().add_to(fg)
g = TimeSliderChoropleth(
    geo_json,
    styledict=style_dict,
).add_to(choro_cluster)

fg2=folium.FeatureGroup(name='Rs_annual site', show=True)
map.add_child(fg2)
marker_cluster = MarkerCluster().add_to(fg2)
map_markers(df5_geojson, marker_cluster, properties_cols)

cmap.add_to(map) # Add colour map 

folium.LayerControl().add_to(map)

outf=f"{outfile}.html"
map.save(outf)
map

print(f"Created file {outf} with {len(grouped)} sites")

# %%
