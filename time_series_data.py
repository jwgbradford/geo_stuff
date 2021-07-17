import pandas, json
from datetime import datetime

''' this is the dictionary structure we want to end up with
{"study_number" : {
    mid_year : rs,
    mid_year : rs
},
"study_number" : {
    mid_year : rs,
    mid_year : rs
}
}
'''

df = pandas.read_csv('srdb-data-V5.csv')[['Latitude', 'Longitude', 'Study_midyear', 'Study_number', 'Rs_annual']].dropna().astype(str)

''' this soltion is almost teh same code as in the example
https://www.analyticsvidhya.com/blog/2020/06/guide-geospatial-analysis-folium-python/
this is his code

covid_dict={}
for i in df_covid['state_id'].unique():
    covid_dict[i]={}
    for j in df_covid[df_covid['state_id']==i].set_index(['state_id']).values:   
        covid_dict[i][j[0]]={'color':j[1],'opacity':0.7}
'''

''' note for this example, I don't convert mid-year into millisecond
also drawing on this example
https://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/TimeSliderChoropleth.ipynb
'''

time_series_data = {}
for study_id in df['Study_number'].unique():
    ''' 
    This is the first difference, instead of doing everything in the one dictionary.
    When I tried that it didn't add new entries to the study_number : {dict} it just overwrote
    and left me with a single entry each time. This could be because our dictionary is actually a
    little simpler than the example above.
    To solve this I create another working dictionary that get's reset each time we pick up a new
    unique Study_number
    '''
    time_series = {}
    # This next line is grouping the rows into a set, using study_number as the index
    # it then returns the values of the row as a list = study_data
    for study_data in df[df['Study_number']==study_id].set_index(['Study_number']).values:
        # the data is now in list, so we can't use our column names to index
        # in the list, mid_year is index 2 and rs_annual is index 3
        # as we're using study_number as the set index, it doesn't get returned in study_data
        dt_time = study_data[2][:4] + '-06-01'
        dt_obj = datetime.strptime(dt_time, '%Y-%m-%d')
        millisec = dt_obj.timestamp()
        dt_index = str(millisec)
        time_series[dt_index] = study_data[3]
    # once we have picked up all the values for a unique study_number
    # we can save it in our main dictionary
    time_series_data[study_id] = time_series
print(time_series_data)
filename = 'time_data.csv'
with open(filename, 'w') as file:
    json.dump(time_series_data, file, indent=2)
file.close