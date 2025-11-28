import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
#import plotly.express as px

files = st.file_uploader('Upload Files', accept_multiple_files = True, type='csv', help = 'When data is not uploaded, we use test_data to  demonstrate how / what graph appear.')

data ={
'Job ID':['job_1','job_2','job_3','job_4','job_5','job_6','job_7','job_8','job_9','job_10'],
'User':['me@gmail.com', 'bot@pipeline-looker.iam.gserviceaccount.com', 'me@gmail.com', 'bot@pipeline-looker.iam.gserviceaccount.com', 'me@gmail.com', 'bot@pipeline-looker.iam.gserviceaccount.com', 'me@gmail.com', 'bot@pipeline-looker.iam.gserviceaccount.com', 'me@gmail.com', 'bot@pipeline-looker.iam.gserviceaccount.com' ],
'Create Time':['2025-01-01T17:58:05.287000', '2025-01-02T19:58:05.287000', '2025-01-03T14:58:05.287000', '2025-01-04T14:58:05.287000', '2025-01-05T13:58:05.287000', '2025-01-06T18:58:05.287000', '2025-01-07T17:58:05.287000', '2025-01-08T16:58:05.287000', '2025-01-09T14:58:05.287000', '2025-01-10T13:58:05.287000' ],
'End Time':['2025-01-01T18:58:05.287000', '2025-01-02T18:58:05.287000', '2025-01-03T18:58:05.287000', '2025-01-04T18:58:05.287000', '2025-01-05T18:58:05.287000', '2025-01-06T18:58:05.287000', '2025-01-07T18:58:05.287000', '2025-01-08T18:58:05.287000', '2025-01-09T18:58:05.287000', '2025-01-10T18:58:05.287000'],
'Duration':[ 0.345, 0.115, 0.235, 0.655, 0.285, 0.985, 0.545, 0.235, 0.765, 0.875 ],
'GB Processed':[0.05, 0.15, 0.025, 0.095, 0.005, 0.065, 0.005, 0.00, 0.085, 0.015],
'Job Type':['QUERY', 'QUERY', 'QUERY', 'QUERY', 'QUERY', 'QUERY','QUERY', 'QUERY','QUERY', 'QUERY'],
'Reference':['looker_', 'stores_', 'ads_', 'looker_', 'stores_', 'ads_', 'looker_', 'stores_', 'ads_', 'stores_'],
'Time':['18:58:00', '18:58:00', '18:58:00', '18:58:00', '18:58:00', '18:58:00', '18:58:00', '18:58:00', '18:58:00', '18:58:00' ],
'Date':['2025-11-1', '2025-11-1', '2025-11-1', '2025-11-1', '2025-11-1', '2025-11-1', '2025-11-1', '2025-11-1', '2025-11-1', '2025-11-1' ],
'Day':['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday']} 

test_data = pd.DataFrame(data)

if files:
    tabs = st.tabs([f"File {i+1}" for i in range(len(files))])
    for i, files in enumerate(files):
        with tabs[i]:
            df = pd.read_csv(files)
            
else:
    st.write('Test Data Visualization')
    tabs = st.tabs(['Test Data'])
    with tabs[0]:
        df = test_data
i = 0
#df = pd.read_csv('jobs.csv')
#only in case table columns name not right

df = df.rename(columns = {'job_id':'Job ID', 'create_time':'Create Time', 'end_time':'End Time', 'duration':'Duration', 'gb_processed':'GB Processed', 'job_type':'Job Type', 'user_email':'User', 'referenced_tables': 'Referenced'})

#df = df.drop(['Unnamed: 11','Unnamed: 12'], axis =1)

df['Create Time'] = pd.to_datetime(df['Create Time'], format='ISO8601')
df['Date'] = df['Create Time'].dt.date

head_adjust = pd.Timedelta(minutes = 30)
df['Time'] = df['Create Time'] + head_adjust
df['Time'] = df['Time'].dt.time.astype('string')
df['Head'] = df['Time'].str[:2]

df['Time'] = df['Create Time'].dt.time.astype('string')
df['Time'] = df['Time'].str[:5]

df = df.set_index('Job ID')
#AFTER DATA CLEANING
################################################################################################################################################################
#EXTEND TABLE
map1 = pd.DataFrame({
    'key': ['ads_', 'stores_', 'looker_', 'TT', 'SHP', 'Organic TT', 'Organic SHP', 'sku712', 'mhs', 'tt performance', 'shp performance', 'Orders'],
    'destination': ['Ads Performance', 'SHP X TT (EMC, SS, PKR) 2.0', 'Livestream Analytic Report', 'Ads Schedule', 'Ads Schedule', 'Ads Schedule', 'Ads Schedule', 'Live Schedule', 'Live Schedule', 'Live Schedule', 'Live Schedule', 'SHP X TT (EMC, SS, PKR) 2.0']})

def map_dashboard(ref):
    for key, dash in zip(map1['key'], map1['destination']):
        if key in ref:
            return dash
    return 'Undetected'
################################################################################################################################################################
# GRAPH TO SEE THE WHOLE STORY ETC
df = df.reset_index() #key before jumping to graph
    
################################################################################################################################################################
epsilon = df.groupby(['Date','Time', 'Head', 'User', 'Reference', 'Day']).agg(**{
    'GB Processed':('GB Processed','sum'),
    'Duration':('Duration','sum')})

epsilon = epsilon.reset_index()
epsilon['Count'] = 1
epsilon['Destination'] = epsilon['Reference'].apply(map_dashboard)
#epsilon['Create Time'] = epsilon['Date'].astype('string') + ' ' + epsilon['Time']

epsilon = epsilon.set_index(['Date','Time', 'Head', 'User', 'Reference', 'Day', 'Destination'] )

with tabs[i]:
    epsilon
    byte , count , duration , avg_byte  = st.columns(4)
    with byte:
        total1 = epsilon['GB Processed'].sum()
        st.header('**Processed GB**', divider= 'grey')
        st.subheader(f'{total1 :,.2f} GB')
        st.caption(f'{(total1 / 1000) * 100 :,.2f} % of capacity limit per month (1 TB per month).')
    with count : 
        total2 = epsilon['Count'].sum()
        st.header('**Unique Jobs**', divider= 'grey')
        st.subheader(f'{total2 :,.0f} Jobs')
        st.caption(f'{total2 :,.0f} Jobs from all sources (BigQuery, Looker).')
    with duration :
        total3 = epsilon['Duration'].mean()
        st.header('**Average Duration**', divider= 'grey')
        st.subheader(f'{total3  :,.2f} ms')
        st.caption(f'{(total3 /  1000) :,.2f} seconds per unique jobs from all sources.')
    with avg_byte : 
        total4 = epsilon['GB Processed'].mean()
        st.header('**Average GB**', divider= 'grey')
        st.subheader(f'{total4 :,.4f} GB')
        st.caption(f'Each job inserted cost estimately {total4*1000 :,.2f} MB')

with tabs[i]:
    st.divider(width='stretch')

epsilon = epsilon.reset_index()
epsilon['Create Time'] = epsilon['Date'].astype('string') + ' ' + epsilon['Time']
timeline = epsilon.groupby('Create Time', as_index = False).agg({'GB Processed':'sum', 'Duration':'sum'})
    
bar = (
    alt.Chart(timeline).mark_line().encode(
        x=alt.X("Create Time:O", title="Date Time", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("GB Processed:Q", title="GB Processed"),
        tooltip=["Create Time", "GB Processed", "Duration"]))

line = (
    alt.Chart(timeline).mark_line().encode(
        x=alt.X("Create Time:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Duration:Q", title="Count", axis=alt.Axis(titleColor="Black")), 
        color=alt.value("green")))

chart = (alt.layer(bar, line).resolve_scale(y="independent").properties(width=700, height=400, title=alt.Title("GB Processed & Duration Overtime", anchor = 'middle', fontSize=20)))

with tabs[i]:
    st.altair_chart(chart)

################################################################################################################################################################
#define tabs / text / function / def

min_date = df['Date'].min()
min_date = pd.to_datetime(min_date).strftime("%d %B")
max_date = df['Date'].max()
max_date = pd.to_datetime(max_date).strftime("%d %B %Y")

tab1, tab2 = st.tabs(['## Summary', '## Dev Tab'])

map2 = pd.DataFrame({
    'key': ['@gmail.com', 'gserviceaccount.com'],
    'define': ['active-user', 'bot']})

def map_user(ref):
    for key, define in zip(map2['key'], map2['define']):
        if key in ref:
            return define
    return 'Undetected'
################################################################################################################################################################
#TAB 1 START HERE


df_pie = epsilon.groupby('User', as_index = False).agg({'GB Processed' : 'sum'})

base = alt.Chart(df_pie).encode(
    theta=alt.Theta('User:N', stack=True),
    radius=alt.Radius('GB Processed:Q', scale=alt.Scale(type='sqrt', zero=True, rangeMin=20)),
    color= alt.Color('User:N', legend = alt.Legend(orient='bottom', direction='horizontal', columns=4, title='User', offset = 0)),
    tooltip = [alt.Tooltip('User:N'), alt.Tooltip('GB Processed:Q', format = ',.2f')]
).properties(title = alt.Title(f'Usage from {min_date} to {max_date} per User', anchor='middle', fontSize = 20), width=600, height=600)

#alt.Legend(orient='top', direction='horizontal', columns=3, title='Reference', anchor = 'middle', offset = 10))

c1 = base.mark_arc(innerRadius=0, stroke=None)

c2 = base.mark_text(radiusOffset=20).encode(text=alt.Text('GB Processed:Q', format = ',.2f'))

with tabs[i]:
    with tab1:
        st.altair_chart(c1 + c2)

################################################################################################################################################################
#SCATTER WITH BRUSH

kappal = epsilon.groupby(['Date', 'Head', 'Destination'], as_index = False).agg({'GB Processed':'sum', 'Count':'sum'})
kappal['Average GB'] = kappal['GB Processed'] / kappal['Count']

color = alt.Color('Destination:N').scale(scheme = 'category20')
brush = alt.selection_interval(encodings=['x'])
click = alt.selection_point(encodings=['y'])

points = alt.Chart(kappal).transform_filter(click).mark_circle().encode(
    x=alt.X('Head:N', title='Time'),
    y=alt.Y('GB Processed:Q', title='GB Processed'),
    size=alt.Size('Count:Q'),
    color=alt.when(brush).then(color).otherwise(alt.value('black')),
    tooltip=[alt.Tooltip('Destination:N'), alt.Tooltip('GB Processed:Q', format = ',.2f'), alt.Tooltip('Count:Q', format = ',.0f')] 
    ).add_params(brush).properties(width=550, height=300)

bars = alt.Chart(kappal).mark_bar().encode(
    x='Count:Q', y='Destination:N',
    color=alt.when(brush).then(color).otherwise(alt.value('black')),
    tooltip=[alt.Tooltip('Destination:N'), alt.Tooltip('Count:Q', format = ',.0f')]
).transform_filter(brush).add_params(click).properties(width=550)

heat = alt.vconcat(points, bars).properties(title=alt.Title('Traffic per Time', anchor = 'middle'))
with tabs[i]:
    with tab1:
        st.altair_chart(heat, use_container_width=True)
###############################################################################################################################################################
#scatter as box and plus whatever
gamma = epsilon.groupby(['Date', 'Head', 'User'], as_index = False).agg({'GB Processed':'sum', 'Duration':'sum', 'Count':'sum'})
gamma['Duration'] = gamma['Duration'] / kappal['Count']
gamma['User Info'] = gamma['User'].apply(map_user)
gamma['User Info'] = gamma['User Info'].astype('string')

pts = alt.selection_point(encodings=['x'])

rect = alt.Chart(gamma).mark_rect().encode(
    alt.X('GB Processed:Q').bin(), alt.Y('Duration:Q').bin(),
    alt.Color('count()').scale(scheme='greenblue').title('Graph'))

circ = rect.mark_point().encode(alt.ColorValue('grey'), alt.Size('count()').title('Count')).transform_filter(pts)

bar = alt.Chart(gamma, width=550, height=200).mark_bar().encode(
    x=alt.X('User Info:N', axis = alt.Axis(labelAngle = 0)), y='count()', color=alt.when(pts).then(alt.ColorValue("steelblue")).otherwise(alt.ColorValue("grey"))).add_params(pts)

scatcrle = alt.vconcat(rect + circ, bar).resolve_legend(color="independent",size="independent")
with tabs[i]:
    with tab1:
        st.altair_chart(scatcrle)
################################################################################################################################################################
#TAB 2 START HERE
cols = epsilon.columns.tolist()
asindex = ['Date', 'Day', 'User', 'Reference', 'Time', 'Head', 'Create Time', 'Destination']
numeric_cols = [c for c in cols if c not in asindex]


with tabs[i]:
    with tab2:
        col1, col2 = st.columns(2)
    with col1 :
        control_var = st.selectbox('Control Variable', asindex)
    with col2 :
        selected_bar = st.selectbox('Sum Variable', numeric_cols)
        selected_line = st.selectbox('Average Variable', numeric_cols)

omega = epsilon.groupby(control_var).agg({selected_bar : 'sum' ,
                                           selected_line: 'sum',
                                           'Count' : 'sum'})

omega[f'Average {selected_line}'] = omega[selected_line] / omega['Count']
with tabs[i]:
    with tab2:
        omega
omega = omega.reset_index()
omega[control_var] = omega[control_var].astype(str)

bar = (
    alt.Chart(omega).mark_bar().encode(
        x=alt.X(control_var +':N', sort=['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], title=control_var 
                #,axis=alt.Axis(labelAngle=0)
               ),
        y=alt.Y(selected_bar + ':Q', title=selected_bar),
        tooltip=[control_var, selected_bar, selected_line]))

line = (
    alt.Chart(omega).mark_line(point=True, interpolate = 'monotone').encode(
        x=alt.X(control_var +':N', sort=['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], title=control_var 
                #,axis=alt.Axis(labelAngle=0)
               ),
        y=alt.Y(f'Average {selected_line}' + ':Q', title=selected_line, axis=alt.Axis(titleColor="white")), 
        color=alt.value("red")))

chart = (alt.layer(bar, line).resolve_scale(y="independent").properties(width=700, height=400, title=f"{selected_bar} (Bar) and {selected_line} (Line) per Day"))
with tabs[i]:
    with tab2:
        st.altair_chart(chart)

################################################################################################################################################################
#DATE AND DAY DISTRIBUTION 
################################################################################################################################################################
#TIME DISTRIBUTION
################################################################################################################################################################
#REFERENCE DISTRIBUTION
################################################################################################################################################################
#TDEVELOPER TABS

epsilon = epsilon.drop(columns = 'Time' , axis =1)
cols = epsilon.columns.tolist()
asindex = ['Date', 'Day', 'User', 'Reference', 'Head', 'Create Time', 'Destination'] 
numeric_cols = [c for c in cols if c not in asindex]

with tabs[i]:
    with tab2:
        col1, col2 = st.columns(2)
    with col1 :
        control_var = st.selectbox('Control Variable-1', asindex)
        asindex_false = [a for a in asindex if a != control_var]
        control_var2 = st.selectbox('Control Variable2-1', asindex_false)
    with col2 :
        selected_bar = st.selectbox('Sum Variable-1', numeric_cols)
        selected_line = st.selectbox('Average Variable-1', numeric_cols)

omega = epsilon.groupby([control_var, control_var2]).agg({selected_bar : 'sum', selected_line: 'sum', 'Count' : 'sum'})

omega[f'Average {selected_line}'] = omega[selected_line] / omega['Count']
with tab2:
    omega
omega = omega.reset_index()
omega[control_var] = omega[control_var].astype(str)
omega[control_var2] = omega[control_var2].astype(str)

bar = alt.Chart(omega).mark_bar().encode(
        x=alt.X(control_var +':N', sort=['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], title=control_var, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(selected_bar + ':Q', title=selected_bar),
        color=alt.Color(control_var2, legend=alt.Legend(orient="bottom", title="Category", columns = 4)),
        tooltip=[control_var, control_var2, selected_bar, selected_line])

line = (
    alt.Chart(omega).mark_circle(size=30).encode(
        x=alt.X(control_var +':N', sort=['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], title=control_var, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f'Average {selected_line}' + ':Q', title=selected_line, axis=alt.Axis(titleColor="white")), 
        color=alt.value("red")))

chart = (alt.layer(bar, line).resolve_scale(y="independent").properties(width=700, height=400, title=alt.Title(f"{selected_bar} (Bar) and {selected_line} (Circle) per Day", anchor = 'middle')))
with tabs[i]:
    with tab2:
        st.altair_chart(chart)