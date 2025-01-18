import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Load the dataset (adjust the file path if needed)
df = pd.read_csv('/kaggle/input/unemployment-in-india/Unemployment in India.csv')

df.head()

df.columns
df.info()
df.isnull().sum()
df = df.dropna()  
df = df.drop_duplicates()
# Optionally, rename columns to more concise names
df.rename(columns={
    'Region': 'Region',
    ' Date': 'Date',  # If you want to remove extra spaces from ' Date'
    ' Frequency': 'Frequency',
    ' Estimated Unemployment Rate (%)': 'Estimated Unemployment Rate (%)',
    ' Estimated Employed': 'Estimated Employed',
    ' Estimated Labour Participation Rate (%)': 'Estimated Labour Participation Rate (%)',
    'Area': 'Area'
}, inplace=True)
df.describe()

fig = px.histogram(df, x='Estimated Unemployment Rate (%)', 
                    nbins=20, title='Distribution of Unemployment Rate', 
                    template='plotly_dark')
fig.show()

fig = px.box(df, x='Region', y='Estimated Unemployment Rate (%)', 
             title='Unemployment Rate by Region', 
             template='plotly_dark')
fig.show()

fig = px.box(df, x='Region', y='Estimated Labour Participation Rate (%)', 
             title='Labour Participation Rate by Region', 
             template='plotly_dark')
fig.show()
fig = px.bar(df.groupby('Area')['Estimated Unemployment Rate (%)'].mean().reset_index(), 
             x='Area', y='Estimated Unemployment Rate (%)', 
             title='Average Unemployment Rate by Area', template='plotly_dark')
fig.show()

top_regions = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().reset_index()
top_regions = top_regions.sort_values(by='Estimated Unemployment Rate (%)', ascending=False).head(10)
fig = px.bar(top_regions, x='Region', y='Estimated Unemployment Rate (%)', 
             title='Top 10 Regions with Highest Unemployment Rate', template='plotly_dark')
fig.show()

fig = px.area(df, x='Date', y='Estimated Employed', color='Area', 
              title='Employment Trend in Rural vs Urban Areas', template='plotly_dark')
fig.show()

fig = px.scatter(df, x='Estimated Unemployment Rate (%)', y='Estimated Employed', color='Area', 
                 title='Unemployment vs Employment for Rural and Urban Areas', template='plotly_dark')
fig.show()

top_regions = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().reset_index()
top_regions = top_regions.sort_values(by='Estimated Unemployment Rate (%)', ascending=False).head(5)
top_regions_data = df[df['Region'].isin(top_regions['Region'])]
fig = px.line(top_regions_data, x='Date', y='Estimated Unemployment Rate (%)', color='Region', 
              title='Time-based Trends for Unemployment Rate in Top Regions', template='plotly_dark')
fig.show()

