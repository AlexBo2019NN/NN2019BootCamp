#!/usr/bin/env python
# coding: utf-8

# # Хакатон Счетной Палаты audithon Трек4.Задача оптимизации участков при работах. 
# 2021 март 27
# Исп.Бочаров А.М. skype bam271074

# In[1]:


#!jupyter labextension install @jupyter-widgets/jupyterlab-manager
#!jupyter labextension install jupyter-leaflet
#!jupyter nbextension install --py --symlink --sys-prefix ipyleaflet
#!jupyter nbextension enable --py --sys-prefix ipyleaflet  # can be skipped for notebook 5.3 and abov
#!jupyter lab --watch


# In[2]:



import pandas as pd
import ipyleaflet
from ipywidgets import HTML
from ipyleaflet import Map, Marker, Popup
#from IPython.display import IFrame
#documentation = IFrame(src='https://ipyleaflet.readthedocs.io/en/latest/', width=1000, height=600)
#display(documentation)


# ## данные для обучения подготовлены в формате excel. Загрузим их.

# In[3]:


df_data=pd.read_excel('c:\\users\\user\\data01.xlsx')
df_data.head(103)


# In[4]:


df_data.shape


# In[5]:


N_list=[];E_list=[]  #lists for coordinates
coord_list=list(df_data['Координаты'])
for i in coord_list:
  buf=str(i).split(';')[0].split(',')
  N_list.append(float(buf[0][1:]));E_list.append(float(buf[1][1:]))
print(N_list[0],E_list[0])


# In[6]:


df_data['lat']=N_list;df_data['lng']=E_list
df_data.head(2)


# In[7]:



object_points = []
i = 0
while i < len(df_data.index):
    object_points.append({'Coordinates': [df_data['lat'][i], df_data['lng'][i]], 
                          'Location': df_data['Наименование'][i], 'PowerOfObject': 1})
    i += 1

marker_coordinates = [obj['Coordinates'] for obj in object_points]
marker_coordinates = [[float(x) for x in y] for y in marker_coordinates] 
# здесь мы проходимся по элементами вложенных списков и меняем их типы со str на float


# In[8]:



m = Map(center=(56.8, 35.8), zoom=11)

markers = [Marker(location=(marker_coordinates[i])) for i in range(len(marker_coordinates))]
info_box_template = """
<dl>
<dt>Адрес:</dt><dd>{Location}</dd>
<dt>Важность объекта:</dt><dd>{PowerOfObject}</dd>
</dl>
"""


# In[9]:


locations_info  = [info_box_template.format(**obj) for obj in object_points]

for i in range(len(markers)):
    markers[i].popup = HTML(locations_info[i])
    m.add_layer(markers[i])

display(m)


# In[10]:


df_data['color']='#FFFF00'


# In[11]:


df_data[['lat','lng','Код объекта','color','Наименование']].to_csv('coord03.csv',index=False)


# In[12]:


print(m)


# In[13]:


from ipyleaflet import AntPath, WidgetControl
from ipywidgets import IntSlider, jslink


# In[14]:


map=Map(center=(56.8, 35.8), zoom=13)


# In[15]:


locations=[]  #list of lists for coordinates [x,y]

for _ in range(len(df_data.index)):
  buf_list=[]
  buf_list.append(float(df_data['lat'][_]))
  buf_list.append(float(df_data['lng'][_]))
  locations.append(buf_list)
  del buf_list


# In[16]:


locations[0]


# In[17]:


my_path=AntPath(
    locations=locations,
    dash_array=[1,10],
    delay=1000,
    color='#9500ff',
    pulse_color='#9500ff'
)


# In[18]:


map.add_layer(my_path)
start_marker=Marker(location=(56.847962835674466, 35.9070110321045))
map.add_layer(start_marker)
finish_marker=Marker(location=(56.881189988867625, 35.927454334450935))
map.add_layer(finish_marker)
start=HTML()
finish=HTML()
start.value='Start'
finish.value='End'
start_marker.popup=start
finish_marker.popup=finish
zoom_slider=IntSlider(
    description='Zoom',
    min=11,
    max=15,
    value=14
)
jslink((zoom_slider,'value'),(m,'zoom'))


# In[19]:


widget_control=WidgetControl(
    widget=zoom_slider,
    position='topright'

)
map.add_control(widget_control)
display(map)


# In[20]:


print(str(map))


# In[21]:


from sklearn.cluster import OPTICS
import numpy as np
X = np.array([[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]])
clustering = OPTICS(min_samples=2).fit(X)
clustering.labels_


# In[24]:


geo_points=np.array(locations)
clusters=OPTICS(min_samples=2).fit(geo_points)
clusters.labels_


# In[25]:


clusters_list=list(clusters.labels_)
clusters_list[0]


# In[26]:


clusters_list[1]


# In[27]:


#let s add clusters to our data
df_data['cluster']=clusters_list
df_data.head()


# In[30]:


#расчитаем сколько участков в каждом кластете
df_data['cluster'].value_counts()
#Кластер  Кол-во участков


# In[ ]:





# In[ ]:




