
#%%
import matplotlib.pyplot as plt
import matplotlib
import json
import pandas as pd
import numpy as np


with open('examples/results/procon_munich.json', 'r', encoding='utf8') as f:
    d = json.load(f)

#%%
d.keys()
#%%
all_links = []
    
page_num = 1
for query in d.keys():
    links = d[query]['1']
    print(links)
    for link in links:
        print(link)
        link['query'] = query
    all_links += links

df = pd.DataFrame(all_links)

df['width'] = df.right - df.left
df['height'] = df.bottom - df.top

#%%
df.head()


#%%
df.head()


#%%
full_width = 10
full_height = 8
#%%
for key in ['width', 'left', 'right']:
    df['norm_{}'.format(key)] = df[key] / df['right'].max()

for key in ['height', 'top', 'bottom']:
    df['norm_{}'.format(key)] = df[key] / df['bottom'].max()

#%%
from urllib.parse import urlparse
def extract(x):
    domain = urlparse(x.href).netloc
    try:
        ret = '.'.join(domain.split('.')[:2])
    except:
        ret = domain
    return ret

df['domain'] = df.apply(extract, axis=1)
df['platform_ugc'] = df['domain'].str.contains('|'.join(
    ['wikipedia', 'twitter', 'facebook']
))
df['wikipedia_appears'] = df['domain'].str.contains('wikipedia')
df.head()

#%%
df[df.domain == 'en.wikipedia']
#%%
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 8}

#matplotlib.rc('font', **font)
#print(d)
fig,ax = plt.subplots(1, 1, figsize=(full_width, full_height))
plt.gca().invert_yaxis()


for i_row, row in df.iterrows():
    if row.width == 0:
        continue

    x = row['norm_left']
    y = row['norm_bottom']
    width = row['norm_width']
    height = row['norm_height']
    domain = row['domain']

    rect = matplotlib.patches.Rectangle((x,y),width,height,linewidth=1,edgecolor='g',facecolor='none')
    if row['platform_ugc']:
        color = 'b'
    elif 'google' in domain:
        color = 'lightgray'
    else:
        color = 'grey'
    plt.annotate(domain, (x, y), color=color)
    # Add the patch to the Axes
    ax.add_patch(rect)

#plt.show()
plt.savefig('overlaid.png')

#%%
roundto = 1
df['grid_left'] = np.round(df['norm_left'], roundto)
df['grid_bottom'] = np.round(df['norm_bottom'], roundto)
df['grid_width'] = np.round(df['norm_width'], roundto)
df['grid_height'] = np.round(df['norm_height'], roundto)
df.head()

# heatmap_points = np.zeros((11, 1))



# for ix, x in enumerate(np.linspace(0, 1, num=11)):
#     for iy, y in enumerate(np.linspace(0, 1, num=11)):
#         print(y, x)
#         try:
#             heatmap_points[iy, ix] = tmp.loc[y, x]
#         except KeyError:
#             heatmap_points[iy, ix] = 0

#%%

#%%
df.groupby(['grid_left', 'grid_bottom']).wikipedia_appears.mean().unstack(level=0).fillna(0)

#%%
heatmap_points = np.zeros((11, 11))
#%%
tmp = df.groupby(['grid_left', 'grid_bottom']).wikipedia_appears.mean().unstack(level=0).fillna(0)
for ix, x in enumerate(np.linspace(0, 1, num=11)):
    for iy, y in enumerate(np.linspace(0, 1, num=11)):
        print(y, x)
        try:
            heatmap_points[iy, ix] = tmp.loc[y, x]
        except KeyError:
            heatmap_points[iy, ix] = 0

#%%
import seaborn as sns
sns.heatmap(heatmap_points)
plt.savefig('heatmap')

#%%
df.groupby('query').wikipedia_appears.agg(any)

#%%
df[df.norm_top < 0.20].groupby('query').wikipedia_appears.agg(any)

#%%
