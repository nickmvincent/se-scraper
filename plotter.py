
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

    domains = [
        'wikipedia',
        'twitter','youtube', 'facebook',
        'instagram', 'linkedin', 'yelp',
        # 'cnn', 'foxnews', 'pinterest',

        # 'webmd', 'medicalnewstoday', 'mayoclinic',
        # 'imdb', 'spotify', 
        # 'yelp', 
    ]
    for domain in domains:
        df[f'{domain}_in'] = df['domain'].str.contains(domain)
        df[f'{domain}_appears'] = (
            df['domain'].str.contains('wikipedia') &
            (df.width != 0) & (df.height != 0)
        )
        kp_line = 780 / right_max
        # source: 
        noscroll_line = 789 / bot_max

        df[f'{domain}_appears_kp'] = (
            (df[f'{domain}_appears']) &
            (df.norm_left > kp_line)
        )

        df[f'{domain}_appears_noscroll'] = (
            (df[f'{domain}_appears']) &
            (df.norm_top > noscroll_line)
        )

        df[f'{domain}_appears_kpnoscroll'] = (
            (df[f'{domain}_appears_kp']) &
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
    'trend',
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
        for itera in [1,2]:
            try:
                err_folder = f'scraper_output/{device}/{search_engine}/errs{itera}_{device}_{search_engine}_{queries}'
                with open(f'{err_folder}/results.json', 'r', encoding='utf8') as f:
                    err_d = json.load(f)
                    print('Loaded errfile')
                    for query in err_d.keys():
                        links = err_d[query]['1_xy']
                        for link in links:
                            link['query'] = query
                        all_links += links
                        query_dfs[device][search_engine][queries][query] = pd.DataFrame(links)

                print('success!')
            except Exception as e:
                print(e)

    dfs[device][search_engine][queries] = norm_df(pd.DataFrame(all_links))
    print('  # errs', len(err_queries[search_engine]))

#%%
err_queries

#%%
# let's see which queries we're missing and write a new file to scrape them
cmds = []
# manual increment
itera = 3
for config in configs:
    device = config['device']
    search_engine = config['search_engine']
    queries = config['queries']

    cur_queries = list(query_dfs[device][search_engine][queries].keys())

    with open(f'search_queries/prepped/{queries}.txt', 'r', encoding='utf8') as f:
        lines = f.read().splitlines()
    print(device, search_engine, queries)
    #print(set(lines))
    #print(set(cur_queries))
    missing = set(lines) - set(cur_queries)
    print(  'Missing')
    print(  missing)
    if missing:
        with open(
            f'search_queries/prepped/errs{itera}_{device}_{search_engine}_{queries}.txt',
            'w', encoding='utf8') as f:
            f.write('\n'.join(list(missing)))
        cmds.append(
            f'/usr/bin/time -v node driver.js {device} {search_engine} errs{itera}_{device}_{search_engine}_{queries} &> logs/errs{itera}_{device}_{search_engine}_{queries}.txt'
        )
with open(f'errs.sh', 'w') as f:
    f.write('\n'.join(cmds))


#%%
# Let's see which links are most common
for config in configs:
    device = config['device']
    if device == 'mobile':
        continue
    search_engine = config['search_engine']
    queries = config['queries']
    print(device, search_engine, queries)
    df = dfs[device][search_engine][queries]
    print(df['domain'].value_counts()[:20])



#%%
# create the coordinate visualization
DO_COORDS = False
if DO_COORDS:
    for config in configs:
        device = config['device']
        search_engine = config['search_engine']
        queries = config['queries']

        print(device, search_engine, queries)
        df = dfs[device][search_engine][queries]
        if type(df) == defaultdict:
            continue
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
                # elif row['platform_ugc']:
                #     color = 'b'
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
            plt.axvline(kp_line, color='r')
            plt.axvline(border_line, color='k')
            plt.axhline(scroll_line, color='k')

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
    k = f'{device}_{search_engine}_{queries}'
    df = dfs[device][search_engine][queries]
    if type(df) == defaultdict:
        continue
    roundto = -1
    df['grid_right'] = np.round(df['right'], roundto)
    df['grid_bottom'] = np.round(df['bottom'], roundto)
    df['grid_width'] = np.round(df['width'], roundto)
    df['grid_height'] = np.round(df['height'], roundto)

    gridded = df[(df.wikipedia_appears == True) & (df.width!=0)].groupby(['grid_right', 'grid_bottom']).wikipedia_appears.sum().unstack(level=0).fillna(0)
    # num_queries = len(set(df['query']))
    # print('num_queries', num_queries)
    heatmap_points = np.zeros((101, 101))

    right_max = df['right'].max()
    bot_max = df['bottom'].max()

    for ix in range(0, 11):
        x = np.round(ix / 10 * right_max, roundto)
        for iy in range(0, 11):
            y = np.round(iy / 100 * bot_max, roundto)
            try:
                heatmap_points[iy, ix] = gridded.loc[y, x]
            except KeyError:
                heatmap_points[iy, ix] = 0

    print(heatmap_points)
    hfig, ax = plt.subplots(1, 1)
    hmap = sns.heatmap(heatmap_points, ax=ax)
    hfig.savefig(f'reports/{k}_heatmap.png')
    #print(df.groupby('query').wikipedia_appears.agg(any))

    inc_rate = df.groupby('query').wikipedia_appears.agg(any).mean()

    matches = set(df[df.wikipedia_appears == True]['query'])

    top_quarter_inc_rate = df[df.grid_bottom <= 0.25].groupby('query').wikipedia_appears.agg(any).mean()


    kp_inc_rate = df.groupby('query').wikipedia_appears_kp.agg(any).mean()

    noscroll_inc_rate = df.groupby('query').wikipedia_appears_noscroll.agg(any).mean()

    kpnoscroll_inc_rate = df.groupby('query').wikipedia_appears_kpnoscroll.agg(any).mean()

    row_dicts.append({
        'queries': queries,
        'search_engine': search_engine,
        'device': device,
        'inc_rate': inc_rate,
        #'top_quarter_inc_rate': top_quarter_inc_rate,
        # 'upper_left_inc_rate': upper_left_inc_rate,
        'kp_inc_rate': kp_inc_rate,
        'noscroll_inc_rate': noscroll_inc_rate,
        'kpnoscroll_inc_rate': kpnoscroll_inc_rate,
        'matches': matches
    })
#%%
df[(df.wikipedia_appears == True) & (df.width!=0)].groupby(['grid_right', 'grid_bottom']).wikipedia_appears.sum().unstack(level=0).fillna(0)

# %%
FP = 'Full-page incidence rate'
RH = 'Right-hand incidence rate'
AF = 'Above-the-fold incidence rate'
results_df = pd.DataFrame(row_dicts)
tmp = results_df[['device', 'search_engine', 'queries', 'inc_rate', 'kp_inc_rate', 'noscroll_inc_rate']]
tmp.rename(columns={
    'device': 'Device', 'search_engine': 'Search Engine',
    'queries': 'Queries', 'inc_rate': FP,
    'kp_inc_rate': RH,
    'noscroll_inc_rate': AF,
}, inplace=True)
tmp.to_csv('reports/main.csv', float_format="%.2f", index=False)
tmp


#%%
import seaborn as sns
tmp2 = tmp.melt(id_vars=['Device', 'Search Engine', 'Queries'])
g = sns.catplot(
    x="Device", y="value",
    hue="Search Engine", col="Queries", row='variable',
    row_order=[FP, AF, RH],
    data=tmp2, kind="bar",
    height=4, ci=None)

    

#%%
g = sns.catplot(
    x="Search Engine", y="Right-hand incidence rate",
    col="Queries",
    data=tmp[tmp.Device == 'desktop'], kind="bar",
    height=4, ci=None)

#%%
import seaborn as sns
g = sns.catplot(
    x="Search Engine", y="Above-the-fold incidence rate",
    hue="Device", col="Queries",
    data=tmp, kind="bar",
    height=4, ci=None)


# %%
for _, row in results_df.iterrows():
    print(row[['device', 'queries', 'search_engine', 'inc_rate']])
    print(row['matches'])
#results_df[['device', 'queries', 'search_engine', 'inc_rate', 'matches']]


# %%
results_df[['device', 'queries', 'search_engine', 'inc_rate', 'kp_inc_rate', 'noscroll_inc_rate']][
    (results_df.search_engine == 'duckduckgo') & (results_df.queries == 'med')
]


# %%

# which DDG cases have Right-hand but no AF?
# mobile RH needs to be set to zero...
# actually sample 10???