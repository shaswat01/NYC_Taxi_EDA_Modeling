# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython


#%%
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from matplotlib import animation
from matplotlib import cm
from matplotlib.pyplot import *
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from dateutil import parser
from IPython.display import HTML
from subprocess import check_output
import seaborn as sns


#%%
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
plt.rcParams['patch.force_edgecolor'] = 'True'
plt.rcParams['figure.figsize'] = (16,10)
plt.rcParams['axes.unicode_minus'] = False


#%%
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


#%%
df_train.head()


#%%
df_train.describe()


#%%
df_train['log_trip duration'] = np.log(df_train['trip_duration'].values + 1)
plt.hist(df_train['log_trip duration'].values, bins=100)
plt.xlabel("log(trip duration)")
plt.ylabel('number of training records')
plt.show


#%%
#Ignore tHis
N = 20000
city_long_border = (-75, -75)
city_lat_border = (40,40)
fig,ax = plt.subplots(ncols=1)
ax.scatter(df_train['pickup_longitude'].values[:N],
             df_train['pickup_latitude'].values[:N],
             color='blue',s=1,label='train',alpha=0.1)

plt.show()


#%%



#%%
type(df_train['pickup_datetime'])


#%%
df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])


#%%
df_train['dropoff_datetime'] = pd.to_datetime(df_train['dropoff_datetime'])


#%%
df_train['pickup_hr'] = df_train['pickup_datetime'].apply(lambda time:time.hour)
df_train['pickup_min'] = df_train['pickup_datetime'].apply(lambda time:time.minute)
df_train['pickup_sec'] = df_train['pickup_datetime'].apply(lambda time:time.second)


#%%
df_train['dropoff_hr'] = df_train['dropoff_datetime'].apply(lambda time:time.hour)
df_train['dropoff_min'] = df_train['dropoff_datetime'].apply(lambda time:time.minute)
df_train['dropoff_sec'] = df_train['dropoff_datetime'].apply(lambda time:time.second)


#%%
df_train['pickup_day'] = df_train['pickup_datetime'].apply(lambda time:time.dayofweek)
df_train['pickup_month'] = df_train['pickup_datetime'].apply(lambda time:time.month)
df_train['pickup_year'] = df_train['pickup_datetime'].apply(lambda time:time.year)
df_train['dropoff_day'] = df_train['dropoff_datetime'].apply(lambda time:time.dayofweek)
df_train['dropoff_month'] = df_train['dropoff_datetime'].apply(lambda time:time.month)
df_train['dropoff_year'] = df_train['dropoff_datetime'].apply(lambda time:time.year)


#%%
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thur',4:'Fri',5:'Sat',6:'Sun'}
df_train['Pickup Day of Week'] = df_train['pickup_day'].map(dmap)
df_train['Dropoff Day of Week'] = df_train['dropoff_day'].map(dmap)


#%%
df_train.head()


#%%
df_train[df_train['Pickup Day of Week']!= df_train['Dropoff Day of Week']].head()


#%%
df_train[df_train['Pickup Day of Week']!= df_train['Dropoff Day of Week']].describe()


#%%
len(df_train[df_train['Pickup Day of Week']!= df_train['Dropoff Day of Week']])


#%%
sns.countplot('Pickup Day of Week',data=df_train, hue='pickup_month')


#%%
sns.countplot('Pickup Day of Week',data=df_train)


#%%
df_train['Date'] = df_train['pickup_datetime'].apply(lambda t: t.date())


#%%
df_train.groupby('Date').count()['id'].plot()


#%%
sns.countplot('Pickup Day of Week',data=df_train,hue='store_and_fwd_flag',palette='coolwarm')


#%%
sns.heatmap(df_train.corr(),cmap='coolwarm')


#%%
sns.scatterplot(x='pickup_latitude',y='dropoff_latitude',data=df_train)


#%%
sns.scatterplot(x='pickup_longitude',y='dropoff_longitude',data=df_train)


#%%
sns.jointplot(x='pickup_hr',y='dropoff_hr',data=df_train[:10000],kind = "reg")


#%%
# IDK if it is good.....
sns.jointplot(x='pickup_hr',y='dropoff_hr',data=df_train[:10000],kind = "hex")


#%%
xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]

df_train = df_train[(df_train.pickup_longitude> xlim[0]) & (df_train.pickup_longitude < xlim[1])]
df_train = df_train[(df_train.dropoff_longitude> xlim[0]) & (df_train.dropoff_longitude < xlim[1])]
df_train = df_train[(df_train.pickup_latitude> ylim[0]) & (df_train.pickup_latitude < ylim[1])]
df_train = df_train[(df_train.dropoff_latitude> ylim[0]) & (df_train.dropoff_latitude < ylim[1])]

longitude = list(df_train.pickup_longitude) + list(df_train.dropoff_longitude)
latitude = list(df_train.pickup_latitude) + list(df_train.dropoff_latitude)


#%%
plt.figure(figsize = (10,10))
plt.plot(longitude,latitude,'.', alpha = 0.4, markersize = 0.05,color="black")
plt.show()


#%%
km_df = pd.DataFrame()
km_df['longitude'] = longitude
km_df['latitude'] = latitude

#%% [markdown]
# #### Now we will cluster the NYC map based on the cabs pick up and drop off points...

#%%
kmeans = KMeans(n_clusters=15, random_state=2, n_init = 10).fit(km_df)
km_df['label'] = kmeans.labels_

km_df = km_df.sample(200000)
plt.figure(figsize = (10,10))
for label in km_df.label.unique():
    plt.plot(km_df.longitude[km_df.label == label],km_df.latitude[km_df.label == label],'.', alpha = 0.3, markersize = 0.3)

plt.title('Clusters of New York Based on Cab pickup and dropoff points')
plt.show()

#%% [markdown]
# As we can see, the clustering results in a partition which is somewhat similar to the way NY is divided into different neighborhoods. We can see Upper East and West side of Central park in gray and pink respectively. West midtown in blue, Chelsea and West Village in brown, downtown area in blue, East Village and SoHo in purple.
# 
# The airports JFK and La LaGuardia have there own cluster, and so do Queens and Harlem. Brooklyn is divided into 2 clusters, and the Bronx has too few rides to be separated from Harlem.
# 
# Let's plot the cluster centers:

#%%
fig,ax = plt.subplots(figsize = (10,10))
for label in km_df.label.unique():
    ax.plot(km_df.longitude[km_df.label == label],km_df.latitude[km_df.label == label],'.', alpha = 0.4, markersize = 0.1, color = 'gray')
    ax.plot(kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1],'o', color = 'r')
    ax.annotate(label, (kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1]), color = 'b', fontsize = 20)
ax.set_title('Center of Clusters')
plt.show()

#%% [markdown]
# ### Taxi rides from one cluster to another
# 
#  And the following animation, every arrow represents rides from one cluster to another. The width of the arrow is proportional to the relative amount of trips in the relevant hour.

#%%
df_train['pickup_cluster'] = kmeans.predict(df_train[['pickup_longitude','pickup_latitude']])
df_train['dropoff_cluster'] = kmeans.predict(df_train[['dropoff_longitude','dropoff_latitude']])
df_train['pickup_hour'] = df_train.pickup_datetime.apply(lambda x: parser.parse(x).hour )
clusters = pd.DataFrame()
clusters['x'] = kmeans.cluster_centers_[:,0]
clusters['y'] = kmeans.cluster_centers_[:,1]
clusters['label'] = range(len(clusters))
km_df = km_df.sample(5000)


#%%
fig, ax = plt.subplots(1, 1, figsize = (10,10))

def animate(hour):
    ax.clear()
    ax.set_title('Absolute Traffic - Hour ' + str(int(hour)) + ':00')    
    plt.figure(figsize = (10,10));
    for label in km_df.label.unique():
        ax.plot(km_df.longitude[km_df.label == label],km_df.latitude[km_df.label == label],'.', alpha = 1, markersize = 2, color = 'gray');
        ax.plot(kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1],'o', color = 'r');


    for label in clusters.label:
        for dest_label in clusters.label:
            num_of_rides = len(df_train[(df_train.pickup_cluster == label) & (df_train.dropoff_cluster == dest_label) & (df_train.pickup_hour == hour)])
            dist_x = clusters.x[clusters.label == label].values[0] - clusters.x[clusters.label == dest_label].values[0]
            dist_y = clusters.y[clusters.label == label].values[0] - clusters.y[clusters.label == dest_label].values[0]
            pct = np.true_divide(num_of_rides,len(df))
            arr = Arrow(clusters.x[clusters.label == label].values, clusters.y[clusters.label == label].values, -dist_x, -dist_y, edgecolor='white', width = 15*pct)
            ax.add_patch(arr)
            arr.set_facecolor('b')


ani = animation.FuncAnimation(fig,animate,sorted(df_train.pickup_hour.unique()), interval = 1000)
plt.close()
ani.save('animation1.gif', writer='imagemagick', fps=2)
filename = 'animation1.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


#%%



#%%

fig, ax = plt.subplots(1, 1, figsize = (10,10))

def animate(hour):
    ax.clear()
    ax.set_title('Relative Traffic - Hour ' + str(int(hour)) + ':00')    
    plt.figure(figsize = (10,10))
    for label in km_df.label.unique():
        ax.plot(km_df.longitude[km_df.label == label],km_df.latitude[km_df.label == label],'.', alpha = 1, markersize = 2, color = 'gray')
        ax.plot(kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1],'o', color = 'r')


    for label in clusters.label:
        for dest_label in clusters.label:
            num_of_rides = len(df_train[(df_train.pickup_cluster == label) & (df_train.dropoff_cluster == dest_label) & (df_train.pickup_hour == hour)])
            dist_x = clusters.x[clusters.label == label].values[0] - clusters.x[clusters.label == dest_label].values[0]
            dist_y = clusters.y[clusters.label == label].values[0] - clusters.y[clusters.label == dest_label].values[0]
            pct = np.true_divide(num_of_rides,len(df_train[df_train.pickup_hour == hour]))
            arr = Arrow(clusters.x[clusters.label == label].values, clusters.y[clusters.label == label].values, -dist_x, -dist_y, edgecolor='white', width = pct)
            ax.add_patch(arr)
            arr.set_facecolor('b')


ani = animation.FuncAnimation(fig,animate,sorted(df_train.pickup_hour.unique()), interval = 1000)
plt.close()
ani.save('animation.gif', writer='imagemagick', fps=2)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))

#%% [markdown]
# ### We can see that in the morning most of the traffic is in Manhattan island.
# 
# The share of taxis travelling to Brooklyn area becomes much larger in the late evening. Since there's no similar movement in the morning hours (in the opposite direction), this is unlikely to be the result of commuting. Instead, and since the traffic is mostly seen after 22:00, these are probably people going out.
# 
# Since the arrows represent the relative traffic in the relevant hour, it is also possible that the increasing width of the arrows leading to Brooklyn may simply be a result of the reduction in the rides in Manhattan, due to the commercial character of big parts of it. But when looking at the absolute traffic, the arrows from Manhattan to Brooklyn are barely seen for the most part of the day.
# 
# In the very early morning, most of the traffic is to and from the two airports. As we can learn from the absolute graph, this is merely the result of decrease in traffic in the other parts of town.
#%% [markdown]
# ### Analysis of Neighbourhood 

#%%
neighborhood = {-74.0019368351: 'Chelsea',-73.837549761: 'Queens',-73.7854240738: 'JFK',-73.9810421975:'Midtown-North-West',-73.9862336241: 'East Village',
                -73.971273324:'Midtown-North-East',-73.9866739677: 'Brooklyn-parkslope',-73.8690098118: 'LaGuardia',-73.9890572967:'Midtown',-74.0081765545: 'Downtown'
                ,-73.9213024854: 'Queens-Astoria',-73.9470256923: 'Harlem',-73.9555565018: 'Uppe East Side',
               -73.9453487097: 'Brooklyn-Williamsburgt',-73.9745967889:'Upper West Side'}
rides_df = pd.DataFrame(columns = neighborhood.values())
rides_df['name'] = neighborhood.values()

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(np.array(list(neighborhood.keys())).reshape(-1, 1), list(neighborhood.values()))


#%%
df_train['pickup_neighborhood'] = neigh.predict(df_train.pickup_longitude.values.reshape(-1,1))
df_train['dropoff_neighborhood'] = neigh.predict(df_train.dropoff_longitude.values.reshape(-1,1))

for col in rides_df.columns[:-1]:
    rides_df[col] = rides_df.name.apply(lambda x: len(df_train[(df_train.pickup_neighborhood == x) & (df_train.dropoff_neighborhood == col)]))


#%%
rides_df.index = rides_df.name
rides_df = rides_df.drop('name', axis = 1)


#%%
fig,ax = plt.subplots(figsize = (12,12))
for i in range(len(rides_df)):  
    ax.plot(rides_df.sum(axis = 1)[i],rides_df.sum(axis = 0)[i],'o', color = 'b')
    ax.annotate(rides_df.index.tolist()[i], (rides_df.sum(axis = 1)[i],rides_df.sum(axis = 0)[i]), color = 'g', fontsize = 12)

ax.plot([0,300000],[0,300000], color = 'r', linewidth = 1)
ax.grid('off')
ax.set_xlim([0,300000])
ax.set_ylim([0,300000])
ax.set_xlabel('Outbound Taxis')
ax.set_ylabel('Inbound Taxis')
ax.set_title('Inbound and Outbound rides for each cluster')

#%% [markdown]
# We can see that the inbound-outbound ratio for each neighborhood is relatively balanced.
# 
# The two airports have more outbound rides than inbound, which makes sense - drivers would probably go to the airport even without passengers, to have the chance to take people into the city. The residential area - Quuens, Brooklyn and Harlem have more inbound ride, whereas the more commercial and touristic areas have more outbound. with Upper East and West, being both commercial and residential, almost on the curve.
# 
# It seems that people would go into Manhattan by alternative means of transportation, but are more likely to get out of it by a cab.

#%%
df_train['pickup_month'] = df_train.pickup_datetime.apply(lambda x: parser.parse(x).month )


#%%
fig,ax = plt.subplots(2,figsize = (12,12))

rides_df = pd.DataFrame(columns = neighborhood.values())
rides_df['name'] = neighborhood.values()
rides_df.index = rides_df.name


for col in rides_df.columns[:-1]:
    rides_df[col] = rides_df.name.apply(lambda x: len(df_train[(df_train.pickup_neighborhood == x) & (df_train.dropoff_neighborhood == col) & (df_train.pickup_month == 6)]))
for i in range(len(rides_df)):  
    ax[0].plot(rides_df.sum(axis = 1)[i],rides_df.sum(axis = 0)[i],'o', color = 'g')
    ax[0].annotate(rides_df.index.tolist()[i], (rides_df.sum(axis = 1)[i],rides_df.sum(axis = 0)[i]), color = 'g', fontsize = 12)

ax[0].grid('off')
ax[0].set_xlabel('Outbound Taxis')
ax[0].set_ylabel('Inbound Taxis')
ax[0].set_title('Inbound and Outbound rides for each cluster - June')
ax[0].set_xlim([0,40000])
ax[0].set_ylim([0,40000])
ax[0].plot([0,40000],[0,40000])


for col in rides_df.columns[:-1]:
    rides_df[col] = rides_df.name.apply(lambda x: len(df_train[(df_train.pickup_neighborhood == x) & (df_train.dropoff_neighborhood == col) & (df_train.pickup_month == 1)]))
rides_df = rides_df.drop('name', axis = 1)
for i in range(len(rides_df)):  
    ax[1].plot(rides_df.sum(axis = 1)[i],rides_df.sum(axis = 0)[i],'o', color = 'b')
    ax[1].annotate(rides_df.index.tolist()[i], (rides_df.sum(axis = 1)[i],rides_df.sum(axis = 0)[i]), color = 'b', fontsize = 12)

ax[1].grid('off')
ax[1].set_xlabel('Outbound Taxis')
ax[1].set_ylabel('Inbound Taxis')
ax[1].set_title('Inbound and Outbound rides for each cluster - January')
ax[1].set_xlim([0,40000])
ax[1].set_ylim([0,40000])
ax[1].plot([0,40000],[0,40000])


#%%


#%% [markdown]
# ## Random Forest Modelling

#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


#%%
X=df_train[['vendor_id',
       'passenger_count', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'pickup_hr', 'pickup_min',
       'pickup_sec', 'dropoff_hr', 'dropoff_min', 'dropoff_sec', 'pickup_day',
       'pickup_month', 'pickup_year', 'dropoff_day', 'dropoff_month',
       'dropoff_year']]
y=df_train['log_trip duration']


#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)


#%%



#%%
from sklearn.ensemble import RandomForestRegressor


#%%
rf = RandomForestRegressor()


#%%
rf.fit(X_train,y_train)


#%%
pred2 = rf.predict(X_test)


#%%
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred2))
print('MSE:', metrics.mean_squared_error(y_test, pred2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred2)))


#%%
from sklearn.metrics import r2_score
r2_score(y_test, pred2)


#%%
sns.distplot((y_test-pred2),bins=50)


#%%



