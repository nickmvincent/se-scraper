
#%%
import matplotlib.pyplot as plt
import matplotlib
import json
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import seaborn as sns
import glob

#%%
# Helpers
def extract(x):
    domain = urlparse(x.href).netloc
    try:
        ret = '.'.join(domain.split('.')[:2])
    except:
        ret = domain
    return ret

#%%
# Display parameters
full_width = 8
full_height = 8


#%%
# Experiment parameters (which experiments to load)
device = 'desktop'
search_engines = [
    'google',
    'bing',
    'duckduckgo',
]
queries = 'top'
# toss results in here for easy dataframe creation
row_dicts = []

#%%
dfs = {}

err_queries = {}
query_dfs = {}
for search_engine in search_engines:
    print('Search engine:', search_engine)
    k = f'{device}_{search_engine}_{queries}'

    folder = f'scraper_output/{device}/{search_engine}/{queries}'

    with open(f'{folder}/results.json', 'r', encoding='utf8') as f:
        d = json.load(f)

    images = glob.glob(f'{folder}/*.png')
    print('# images', len(images))
    all_links = []
    err_queries[search_engine] = {}
    dfs[search_engine] = {}
        
    n_queries = len(d.keys())
    print('# queries collected:', n_queries)
    for query in d.keys():
        #print(query)
        try:
            links = d[query]['1_xy']
            #print(links)
            for link in links:
                #print(link)
                link['query'] = query
            all_links += links
            query_dfs[search_engines][query] = pd.DataFrame(links)
        except:
            err_queries[search_engine][query] = d[query]
    dfs[search_engine] = pd.DataFrame(all_links)
    print('# errs', len(err_queries[search_engine]))

#%%
dfs['google']

#%%
def norm_df(df):
    df['width'] = df.right - df.left
    df['height'] = df.bottom - df.top

    # normalize all x-axis values relative to rightmost point
    for key in ['width', 'left', 'right']:
        df['norm_{}'.format(key)] = df[key] / df['right'].max()

    # normalize all y-axis values relative to bottommost point
    for key in ['height', 'top', 'bottom']:
        df['norm_{}'.format(key)] = df[key] / df['bottom'].max()

    df['domain'] = df.apply(extract, axis=1)

    df['platform_ugc'] = df['domain'].str.contains('|'.join(
        ['wikipedia', 'twitter', 'facebook', 'instagram', 'reddit', ]
    ))
    df['wikipedia_appears'] = df['domain'].str.contains('wikipedia')
    return df


#%%
for search_engine in search_engines:
    df = dfs[search_engine]
    df = norm_df(df)
    ratio = df['bottom'].max() / df['right'].max()

    for query in list(query_dfs.keys()) + [None]:
        if query:
            subdf = df[df['query'] == query]
        else:
            subdf = df
        fig, ax = plt.subplots(1, 1, figsize=(full_width, full_width * height))
        plt.gca().invert_yaxis()
        for i_row, row in subdf.iterrows():
            if row.width == 0:
                continue
            x = row['norm_left']
            y = row['norm_bottom']
            width = row['norm_width']
            height = row['norm_height']
            domain = row['domain']

            if row['platform_ugc']:
                color = 'b'
            elif 'google' in domain:
                color = 'lightgray'
            else:
                color = 'grey'
            plt.annotate(domain, (x, y), color=color)
            # Add the patch to the Axes
            rect = matplotlib.patches.Rectangle((x,y),width,height,linewidth=1,edgecolor=color,facecolor='none')
            ax.add_patch(rect)

            plt.savefig(f'reports/{k}_{query}_overlaid.png')

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

    print('The upper left incidence rate is')
    upper_left_inc_rate = df[(df.grid_bottom <= 0.5) & (df.grid_left <= 0.5)].groupby('query').wikipedia_appears.agg(any).mean()
    print(upper_left_inc_rate)

    row_dicts.append({
        'search_engine': search_engine,
        'device': device,
        'inc_rate': inc_rate,
        'top_quarter_inc_rate': top_quarter_inc_rate,
        'upper_left_inc_rate': upper_left_inc_rate,
    })


# %%
results_df = pd.DataFrame(row_dicts)
results_df

# %%
