import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from textblob import TextBlob

data = pd.read_csv("C:/Users/pakhi/Desktop/python/csv file/location_geocode.csv")
#print(data.head())
twi_data = pd.read_csv("C:/Users/pakhi/Desktop/python/csv file/auspol2019.csv")
#print(twi_data.head())
twi_data = twi_data.merge(data,how='inner',left_on = 'user_location' , right_on = 'name')
twi_data = twi_data.drop('name',axis=1)
twi_data.isnull().mean()*100
####
twi_data['created_at'] = pd.to_datetime(twi_data['created_at'])
cnt_srs = twi_data['created_at'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index , cnt_srs.values , alpha = 0.8 , color="green")
plt.xticks(rotation='vertical')
plt.xlabel("date",fontsize=12)
plt.ylabel("number of tweets",fontsize=12)
plt.title("number of tweet accoring to date")
plt.show()
####
twi_data['user_created_at'] = pd.to_datetime(twi_data['user_created_at'])
cont = twi_data['user_created_at'].dt.date.value_counts()
cont = cont[:10,]
plt.figure(figsize=(14,6))
sns.barplot(cont.index , cont.values , alpha = 0.8 )
plt.xticks(rotation='vertical')
plt.xlabel("date",fontsize=12)
plt.ylabel("number of tweets",fontsize=12)
plt.title("number of tweet accoring to date")
plt.show()
####
twi_data['tweeted_day_of_week'] = twi_data['created_at'].dt.weekday_name
count = twi_data['tweeted_day_of_week'].value_counts()
count = count.sort_index()
#print(count)
label = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
sizes = [14651,8600,9139,12891,14622,54879,20598]
plt.pie(sizes,labels = label,shadow='True')
plt.plot()
####
cnt = twi_data['tweeted_day_of_week'].value_counts()
cnt = cnt.sort_index()
fig = {
    "data":[{
        "values":cnt.values,
        "labels":cnt.index,
        "name":"number of tweet per day",
        "hoverinfo":"label+percent+name",
        "hole":.4,
        "type":"pie"
    }],
    "layout":{
        "title":"tweet of the day in week",
        "annotations":[{
            "font":{"size":20},
            "showarrow":False,
            "text":"tweet",
            "x":0.5,
            "y":0.5
        }]
    }
}
py.iplot(fig,filename="basic_pie_plot")
####
twi_data['created_day_of_week'] = twi_data['user_created_at'].dt.weekday_name
cnt_ = twi_data['created_day_of_week'].value_counts()
cnt_ = cnt_.sort_index() 
fig = {
  "data": [
    {
      "values": cnt_.values,
      "labels": cnt_.index,
      "domain": {"x": [0, .5]},
      "name": "Number of tweets per day",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Percentage of created accounts per day",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
             "text": "Percentage of accounts created according to days of the week",
                "x": 0.50,
                "y": 1.1
            },
        ]
    }
}
py.iplot(fig)
####
twi_data['tweeted_of_hour'] = twi_data['created_at'].dt.hour
cnt_ = twi_data['tweeted_of_hour'].value_counts()
cnt_ = cnt_.sort_index() 
trace1 = go.Scatter(
                    x = cnt_.index,
                    y = cnt_.values,
                    mode = "lines"
                    )

data = [trace1]
layout = dict(title = 'Number of tweets per hour')
             
fig = dict(data = data, layout = layout)
py.iplot(fig)
####
twi_data['created_at_hour'] = twi_data['user_created_at'].dt.hour
cnt_ = twi_data['created_at_hour'].value_counts()
cnt_ = cnt_.sort_index() 
trace1 = go.Scatter(
                    x = cnt_.index,
                    y = cnt_.values,
                    mode = "lines"
                    )

data = [trace1]
layout = dict(title = 'Number of tweets per hour')
             
fig = dict(data = data, layout = layout)
py.iplot(fig)
####
twi_data['sentiment'] = twi_data['full_text'].map(lambda text: TextBlob(text).sentiment.polarity)
cut = pd.cut(twi_data["sentiment"],[-np.inf,-.01,.01,np.inf],labels=["negative","netural","positive"]) 
twi_data["polarity"] = cut.values
#print(twi_data[["polarity","sentiment"]].head())
twi_data["polarity"].value_counts()
trace = go.Histogram(
    x = twi_data["sentiment"]
)
data = [trace]
layout = go.Layout(title = "Hispgram of sentiment of tweeted data",
                  xaxis = dict(title = "Sentiment"),
                  yaxis = dict(title = "Count"))
fig = go.Figure(data=data,layout = layout)
py.iplot(fig)
####
