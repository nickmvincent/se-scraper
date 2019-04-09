#%%
import matplotlib.pyplot as plt
import matplotlib
import json
import pandas as pd

with open('examples/results/coffee.json', 'r') as f:
    d = json.load(f)

#%%
df = pd.DataFrame.from_dict(d['frontpage'])
df['width'] = df.right - df.left
df['height'] = df.bottom - df.top

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
df.head()

#%%
df[df.domain == 'en.wikipedia']
#%%
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 8}

matplotlib.rc('font', **font)
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
    if 'wikipedia' in domain or 'twitter' in domain or 'facebook' in domain:
        color = 'b'
    elif 'google' in domain:
        color = 'lightgray'
    else:
        color = 'grey'
    plt.annotate(domain, (x, y), color=color)
    # Add the patch to the Axes
    ax.add_patch(rect)

plt.show()


#%%
