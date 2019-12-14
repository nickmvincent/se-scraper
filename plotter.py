
#%%
import matplotlib.pyplot as plt
import matplotlib
import json
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import seaborn as sns


#%%
device = 'desktop'
search_engines = [
    'google',
    'bing',
    #'duckduckgo',
]
queries = 'trend_sample'


full_width = 10
full_height = 8

row_dicts = []

def extract(x):
    domain = urlparse(x.href).netloc
    try:
        ret = '.'.join(domain.split('.')[:2])
    except:
        ret = domain
    return ret


#%%
for search_engine in search_engines:
    k = f'{device}_{search_engine}_{queries}'

    with open(f'output/{device}/{search_engine}/{queries}/results.json', 'r', encoding='utf8') as f:
        d = json.load(f)
    all_links = []
        
    for query in d.keys():
        print(query)
        links = d[query]['1_xy']
        #print(links)
        for link in links:
            #print(link)
            link['query'] = query
        all_links += links

    df = pd.DataFrame(all_links)

    df['width'] = df.right - df.left
    df['height'] = df.bottom - df.top
    print(df.head(3))

    for key in ['width', 'left', 'right']:
        df['norm_{}'.format(key)] = df[key] / df['right'].max()

    for key in ['height', 'top', 'bottom']:
        df['norm_{}'.format(key)] = df[key] / df['bottom'].max()

    df['domain'] = df.apply(extract, axis=1)
    df['platform_ugc'] = df['domain'].str.contains('|'.join(
        ['wikipedia', 'twitter', 'facebook', 'instagram', 'reddit', ]
    ))
    df['wikipedia_appears'] = df['domain'].str.contains('wikipedia')

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

    plt.savefig(f'reports/{k}_overlaid.png')

    roundto = 1
    df['grid_left'] = np.round(df['norm_left'], roundto)
    df['grid_bottom'] = np.round(df['norm_bottom'], roundto)
    df['grid_width'] = np.round(df['norm_width'], roundto)
    df['grid_height'] = np.round(df['norm_height'], roundto)
    print(df.head(3))

    gridded = df[df.wikipedia_appears == True].groupby(['grid_left', 'grid_bottom']).wikipedia_appears.sum().unstack(level=0).fillna(0)
    num_queries = len(set(df['query']))
    print('num_queries', num_queries)
    heatmap_points = np.zeros((11, 11))

    for ix in range(0, 11):
        x = np.round(ix * 0.1, 1)
        for iy in range(0, 11):
            y = np.round(iy * 0.1, 1)
            #print(y, x)
            try:
                heatmap_points[iy, ix] = gridded.loc[y, x] / num_queries
            except KeyError:
                heatmap_points[iy, ix] = 0


    hfig, ax = plt.subplots(1, 1)
    hmap = sns.heatmap(heatmap_points, ax=ax)
    hfig.savefig(f'reports/{k}_heatmap.png')
    print(search_engine)
    print(df.groupby('query').wikipedia_appears.agg(any))

    print('The incidence rate is')
    inc_rate = df.groupby('query').wikipedia_appears.agg(any).mean()
    print(inc_rate)

    print('The top-quarter incidence rate is')
    top_quarter_inc_rate = df[df.grid_bottom <= 0.25].groupby('query').wikipedia_appears.agg(any).mean()
    print(top_quarter_inc_rate)

    row_dicts.append({
        'search_engine': search_engine,
        'device': device,
        'inc_rate': inc_rate,
        'top_quarter_inc_rate': top_quarter_inc_rate,
    })


# %%
results_df = pd.DataFrame(row_dicts)
results_df