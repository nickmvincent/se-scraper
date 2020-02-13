
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
def norm_df(df):
    df['width'] = df.right - df.left
    df['height'] = df.bottom - df.top
    right_max = df['right'].max()
    bot_max = df['bottom'].max()

    # normalize all x-axis values relative to rightmost point
    for key in ['width', 'left', 'right']:
        df['norm_{}'.format(key)] = df[key] / right_max

    # normalize all y-axis values relative to bottommost point
    for key in ['height', 'top', 'bottom']:
        df['norm_{}'.format(key)] = df[key] / bot_max

    df['domain'] = df.apply(extract, axis=1)

    df['platform_ugc'] = df['domain'].str.contains('|'.join(
        ['wikipedia', 'twitter', 'facebook', 'instagram', 'reddit', ]
    ))
    df['wikipedia_in'] = df['domain'].str.contains('wikipedia')
    df['wikipedia_appears'] = (
        df['domain'].str.contains('wikipedia') &
        (df.width != 0) & (df.height != 0)
    )
    kp_line = 780 / right_max
    # source
    noscroll_line = 789 / bot_max

    df['wikipedia_appears_kp'] = (
        (df['wikipedia_appears']) &
        (df.norm_left > kp_line)
    )

    df['wikipedia_appears_noscroll'] = (
        (df['wikipedia_appears']) &
        (df.norm_top > noscroll_line)
    )
    
    return df

#%%
# Display parameters
full_width = 8

#%%
# Experiment parameters (which experiments to load)
devices = [
    'desktop',
    'mobile'
]
search_engines = [
    'google',
    'bing',
    'duckduckgo',
]
query_sets = [
    'top',
    'med',
]
configs = []
for device in devices:
    for search_engine in search_engines:
        for queries in query_sets:
            configs.append({
                'device': device,
                'search_engine': search_engine,
                'queries': queries,
            })


#%%
from collections import defaultdict

infinite_defaultdict = lambda: defaultdict(infinite_defaultdict)
dfs = infinite_defaultdict()
# device, search_engine, queries

err_queries = infinite_defaultdict()
query_dfs = infinite_defaultdict()
for config in configs:
    device = config['device']
    search_engine = config['search_engine']
    queries = config['queries']

    print(device, search_engine, queries)
    k = f'{device}_{search_engine}_{queries}'

    folder = f'scraper_output/{device}/{search_engine}/{queries}'

    try:
        with open(f'{folder}/results.json', 'r', encoding='utf8') as f:
            d = json.load(f)
    except FileNotFoundError:
        print('  ...Skipping')
        continue

    images = glob.glob(f'{folder}/*.png')
    print('  # images', len(images))
    all_links = []
        
    n_queries = len(d.keys())
    print('  # queries collected:', n_queries)
    if len(images) != n_queries:
        print('  Mismatch')
        # see if we already collected the erroneous queries
    
    num_errs = 0
    for query in d.keys():
        try:
            links = d[query]['1_xy']
            #print(links)
            for link in links:
                #print(link)
                link['query'] = query
            all_links += links
            query_dfs[device][search_engine][queries][query] = pd.DataFrame(links)
        except KeyError:
            err_queries[device][search_engine][queries][search_engine][query] = d[query]
            num_errs += 1
    if num_errs > 0:
        print('# errs,', num_errs)
        try:
            err_folder = f'scraper_output/{device}/{search_engine}/err_{device}_{search_engine}_{queries}'
        with open(f'{err_folder}/results.json', 'r', encoding='utf8') as f:
            err_d = json.load(f)
            try:
                links = err_d[query]['1_xy']
                for link in links:
                    link['query'] = query
                all_links += links
                query_dfs[device][search_engine][queries][query] = pd.DataFrame(links)
            except KeyError:
                print('Error in the "err" file. Manually check!')
    dfs[device][search_engine][queries] = pd.DataFrame(all_links)
    print('  # errs', len(err_queries[search_engine]))

#%%
err_queries

#%%
# let's see which queries we're missing and write a new file to scrape them
cmds = []
for config in configs:
    device = config['device']
    search_engine = config['search_engine']
    queries = config['queries']

    cur_queries = list(query_dfs[device][search_engine][queries].keys())

    with open(f'search_queries/prepped/{queries}.txt', 'r') as f:
        lines = f.read().splitlines()
    print(device, search_engine, queries)
    #print(set(lines))
    #print(set(cur_queries))
    missing = set(lines) - set(cur_queries)
    print(  'Missing')
    print(  missing)
    if missing:
        with open(f'search_queries/prepped/errs_{device}_{search_engine}_{queries}.txt', 'w') as f:
            f.write('\n'.join(list(missing)))
        cmds.append(
            f'/usr/bin/time -v node driver.js {device} {search_engine} errs_{device}_{search_engine}_{queries} &> logs/errs_{device}_{search_engine}_{queries}.txt'
        )
with open(f'errs.sh', 'w') as f:
    f.write('\n'.join(cmds))




#%%
for config in configs:
    device = config['device']
    search_engine = config['search_engine']
    queries = config['queries']

    print(device, search_engine, queries)
    df = dfs[device][search_engine][queries]
    df = norm_df(df)
    right_max = df['right'].max()
    bot_max = df['bottom'].max()
    ratio = bot_max / right_max
    k = f'{device}_{search_engine}_{queries}'

    cur_queries = list(query_dfs[device][search_engine][queries].keys())
    for query in cur_queries + [None]:
        
        if query:
            subdf = df[df['query'] == query]
        else:
            subdf = df
        fig, ax = plt.subplots(1, 1, figsize=(full_width, full_width * ratio))
        plt.gca().invert_yaxis()
        #print('Query:', query, '# links', len(subdf))
        for i_row, row in subdf.iterrows():
            if row.width == 0 or row.height == 0:
                continue
            x = row['norm_left']
            y = row['norm_bottom']
            width = row['norm_width']
            height = row['norm_height']
            # x = row['left']
            # y = row['bottom']
            # width = row['width']
            # height = row['height']
            domain = row['domain']

            if row['wikipedia_appears']:
                color = 'g'
            elif row['platform_ugc']:
                color = 'b'
            elif 'google' in domain:
                color = 'lightgray'
            else:
                color = 'grey'
            plt.annotate(domain, (x, y), color=color)
            # Add the patch to the Axes
            rect = matplotlib.patches.Rectangle((x,y),width,height,linewidth=1,edgecolor=color,facecolor='none')
            ax.add_patch(rect)

        kp_line = 820 / right_max
        scroll_line = 670 / bot_max
        border_line = 900 / bot_max
        plt.axvline(kp_line)
        plt.axvline(border_line)
        plt.axhline(scroll_line)

        #print(full_width, full_width * ratio)
        plt.savefig(f'reports/overlays/{k}_{query}.png')
        plt.close()

#%%
# toss results in here for easy dataframe creation
row_dicts = []
for config in configs:
    device = config['device']
    search_engine = config['search_engine']
    queries = config['queries']

    print(device, search_engine, queries)
    df = dfs[device][search_engine][queries]
    if type(df) == defaultdict:
        continue
    df = norm_df(df)
    roundto = 1
    df['grid_left'] = np.round(df['norm_left'], roundto)
    df['grid_bottom'] = np.round(df['norm_bottom'], roundto)
    df['grid_width'] = np.round(df['norm_width'], roundto)
    df['grid_height'] = np.round(df['norm_height'], roundto)

    gridded = df[(df.wikipedia_appears == True) & (df.width !=0)].groupby(['grid_left', 'grid_bottom']).wikipedia_appears.sum().unstack(level=0).fillna(0)
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
    #print(df.groupby('query').wikipedia_appears.agg(any))

    print('The incidence rate is')
    inc_rate = df.groupby('query').wikipedia_appears.agg(any).mean()
    print(inc_rate)

    matches = set(df[df.wikipedia_appears == True]['query'])

    print('The top-quarter incidence rate is')
    top_quarter_inc_rate = df[df.grid_bottom <= 0.25].groupby('query').wikipedia_appears.agg(any).mean()
    print(top_quarter_inc_rate)

    # print('The upper left incidence rate is')
    # upper_left_inc_rate = df[(df.grid_bottom <= 0.5) & (df.grid_left <= 0.5)].groupby('query').wikipedia_appears.agg(any).mean()
    # print(upper_left_inc_rate)

    print('The kp incidence rate is')
    kp_inc_rate = df.groupby('query').wikipedia_appears_kp.agg(any).mean()
    print(kp_inc_rate)

    print('The no scroll incidence rate is')
    noscroll_inc_rate = df.groupby('query').wikipedia_appears_noscroll.agg(any).mean()
    print(noscroll_inc_rate)

    row_dicts.append({
        'queries': queries,
        'search_engine': search_engine,
        'device': device,
        'inc_rate': inc_rate,
        #'top_quarter_inc_rate': top_quarter_inc_rate,
        # 'upper_left_inc_rate': upper_left_inc_rate,
        'kp_inc_rate': kp_inc_rate,
        'noscroll_inc_rate': noscroll_inc_rate,
        'matches': matches
    })


# %%
results_df = pd.DataFrame(row_dicts)
results_df[['device', 'queries', 'search_engine', 'inc_rate', 'kp_inc_rate', 'noscroll_inc_rate']]


# %%
for _, row in results_df.iterrows():
    print(row[['device', 'queries', 'search_engine', 'inc_rate']])
    print(row['matches'])
#results_df[['device', 'queries', 'search_engine', 'inc_rate', 'matches']]


# %%
results_df[['device', 'queries', 'search_engine', 'inc_rate', 'kp_inc_rate', 'noscroll_inc_rate']][
    (results_df.search_engine == 'google') & (results_df.queries == 'med')
]


# %%
