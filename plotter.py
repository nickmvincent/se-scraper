
#%%
import matplotlib.pyplot as plt
import matplotlib
import json
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import seaborn as sns
import glob
import sys
from PIL import Image
import seaborn as sns


#%%
# Helpers
def extract(x):
    domain = urlparse(x.href).netloc
    return domain
    # try:
    #     ret = '.'.join(domain.split('.')[:2])
    # except:
    #     ret = domain
    # return ret

def norm_df(df, mobile=False):
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

    # treat links to DDG twitter & reddit as internal
    df.loc[df.href == 'https://twitter.com/duckduckgo', 'href'] = 'www.duckduckgo.com'
    df.loc[df.href == 'https://reddit.com/r/duckduckgo', 'href'] = 'www.duckduckgo.com'

    df['domain'] = df.apply(extract, axis=1)

    domains = [
        'wikipedia',
        'twitter', 'youtube',
        'facebook',
    ]

    df['platform_ugc'] = df['domain'].str.contains('|'.join(
        domains
    ))
    
    for domain in domains:
        df[f'{domain}_in'] = df['domain'].str.contains(domain)
        df[f'{domain}_appears'] = (
            df['domain'].str.contains(domain) &
            (df.width != 0) & (df.height != 0)
        )
        kp_line = 780 / right_max
        # source: 

        if mobile:
            df[f'{domain}_appears_right'] = 0
            df[f'{domain}_appears_noscrollleft'] = 0
            # from what's my viewport, iphone X
            mobile_noscroll_line = IPHONE_SCROLL_PIX / bot_max

            df[f'{domain}_appears_noscroll'] = (
                (df[f'{domain}_appears']) &
                (df.norm_top < mobile_noscroll_line)
            )

        else:
            noscroll_line = MACBOOK_SCROLL_PIX / bot_max

            df[f'{domain}_appears_right'] = (
                (df[f'{domain}_appears']) &
                (df.norm_left > kp_line)
            )

            df[f'{domain}_appears_left'] = (
                (df[f'{domain}_appears']) &
                (df.norm_left <= kp_line)
            )

            df[f'{domain}_appears_noscroll'] = (
                (df[f'{domain}_appears']) &
                (df.norm_top < noscroll_line)
            )

            df[f'{domain}_appears_noscrollleft'] = (
                (df[f'{domain}_appears_left']) &
                (df.norm_top < noscroll_line) &
            )

        # df[f'{domain}_appears_rightnoscroll'] = (
        #     (df[f'{domain}_appears_right']) &
        #     (df.norm_top > noscroll_line)
        # )
    
    return df

#%%
# Display parameters
full_width = 8
KP_PIX = 780
IPHONE_SCROLL_PIX = 635
MACBOOK_SCROLL_PIX = 789
BORDER_PIX = 1440


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
        print(f'  {folder}/results.json')
        continue

    # images = glob.glob(f'{folder}/*.png')
    # print('  # images', len(images))
    all_links = []
        
    n_queries = len(d.keys())
    print('  # queries collected:', n_queries)
    
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
    print('# errs,', num_errs)
    for itera in [1,2,3,4,5]:
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

            print(f'success for itera {itera}!')
            print(len(err_d.keys()))
        except Exception as e:
            pass

    dfs[device][search_engine][queries] = norm_df(pd.DataFrame(all_links), device == 'mobile')
    print('  # errs', len(err_queries[search_engine]))

#%%
err_queries

#%%
# let's see which queries we're missing and write a new file to scrape them
cmds = []
# manual increment
itera = 5
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
    print(missing)
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
for_concat_list = []
for config in configs:
    device = config['device']
    if device == 'mobile':
        continue
    search_engine = config['search_engine']
    queries = config['queries']
    print(device, search_engine, queries)
    for_concat_df = dfs[device][search_engine][queries][['domain']]
    for_concat_list.append(for_concat_df)
    #print(for_concat_df['domain'].value_counts()[:20])
pd.concat(for_concat_list)['domain'].value_counts()[:15]


#%%
# source: https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python


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
        np.random.seed(0)
        chosen_ones = np.random.choice(cur_queries, 10, replace=False)
        with open(f'reports/samples/{k}.txt', 'w', encoding='utf8') as f:
            f.write('\n'.join(chosen_ones))
        for query in cur_queries + [None]:
            
            if query:
                subdf = df[df['query'] == query]
            else:
                subdf = df
            fig, ax = plt.subplots(1, 1, figsize=(full_width, full_width * ratio))
            plt.gca().invert_yaxis()
            #print('Query:', query, '# links', len(subdf))
            add_last = []
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
                    add_last.append([domain, (x,y,), width, height])
                else:
                    if row['platform_ugc']:
                        color = 'b'
                    elif 'google' in domain or 'bing' in domain or 'duckduckgo' in domain:
                        color = 'lightgray'
                    else:
                        color = 'grey'
                    plt.annotate(domain, (x, y), color=color)
                    # Add the patch to the Axes
                    rect = matplotlib.patches.Rectangle((x,y),width,height,linewidth=1,edgecolor=color,facecolor='none')
                    ax.add_patch(rect)
            for domain, coords, width, height in add_last:
                plt.annotate(domain, coords, color='g')
                rect = matplotlib.patches.Rectangle(coords,width,height,linewidth=2,edgecolor=color,facecolor='none')
                ax.add_patch(rect)

            kp_line = KP_PIX / right_max
            if device == 'mobile':
                scroll_line = IPHONE_SCROLL_PIX / bot_max
            else:
                scroll_line = MACBOOK_SCROLL_PIX / bot_max
            border_line = BORDER_PIX / right_max
            plt.axvline(kp_line, color='r', linestyle='-')
            plt.axvline(border_line, color='k', linestyle='-')
            plt.axhline(scroll_line, color='k', linestyle='-')

            #print(full_width, full_width * ratio)
            plt.savefig(f'reports/overlays/{k}_{query}.png')
            if query == 'nba':
                plt.savefig(f'reports/{k}_{query}.png')
            plt.close()
            if query in chosen_ones:
                screenshot_path = f'scraper_output/{device}/{search_engine}/{queries}/results.json_{query}.png'
                # the overlay will be smaller
                try:
                    screenshot_img = Image.open(screenshot_path)
                    big_w, big_h = screenshot_img.size
                    overlay_img = Image.open(f'reports/overlays/{k}_{query}.png')
                    small_w, small_h = overlay_img.size
                except FileNotFoundError: 
                    # can happen b/c 
                    continue

                h_percent = (big_h/float(small_h))
                new_w = int((float(small_w) * float(h_percent)))
                resized_overlay = overlay_img.resize((new_w,big_h), Image.ANTIALIAS)

                total_width = new_w + big_w

                new_im = Image.new('RGB', (total_width, big_h))

                x_offset = 0
                for im in (screenshot_img, resized_overlay):
                    new_im.paste(im, (x_offset,0))
                    x_offset += im.size[0]

                new_im.save(f'reports/samples/concat_{k}_{query}.png')


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
    # roundto = -1
    # df['grid_right'] = np.round(df['right'], roundto)
    # df['grid_bottom'] = np.round(df['bottom'], roundto)
    # df['grid_width'] = np.round(df['width'], roundto)
    # df['grid_height'] = np.round(df['height'], roundto)

    # gridded = df[(df.wikipedia_appears == True) & (df.width!=0)].groupby(['grid_right', 'grid_bottom']).wikipedia_appears.sum().unstack(level=0).fillna(0)

    # heatmap_points = np.zeros((101, 101))

    # right_max = df['right'].max()
    # bot_max = df['bottom'].max()

    # for ix in range(0, 11):
    #     x = np.round(ix / 10 * right_max, roundto)
    #     for iy in range(0, 11):
    #         y = np.round(iy / 100 * bot_max, roundto)
    #         try:
    #             heatmap_points[iy, ix] = gridded.loc[y, x]
    #         except KeyError:
    #             heatmap_points[iy, ix] = 0

    # print(heatmap_points)
    # hfig, ax = plt.subplots(1, 1)
    # hmap = sns.heatmap(heatmap_points, ax=ax)
    # hfig.savefig(f'reports/{k}_heatmap.png')
    #print(df.groupby('query').wikipedia_appears.agg(any))

    inc_rate = df.groupby('query').wikipedia_appears.agg(any).mean()    
    matches = set(df[df.wikipedia_appears == True]['query'])
    kp_inc_rate = df.groupby('query').wikipedia_appears_right.agg(any).mean()
    noscroll_inc_rate = df.groupby('query').wikipedia_appears_noscroll.agg(any).mean()

    row_dict = {
        'queries': queries,
        'search_engine': search_engine,
        'device': device,
        'inc_rate': inc_rate,
        'kp_inc_rate': kp_inc_rate,
        'noscroll_inc_rate': noscroll_inc_rate,
        'matches': matches
    }
    for domain in [
        'twitter', 'youtube',
        'facebook',
    ]:
        row_dict[f'{domain}_inc_rate'] = df.groupby('query')[f'{domain}_appears'].agg(any).mean() 


    row_dicts.append(row_dict)
#%%
results_df = pd.DataFrame(row_dicts)
results_df

# %%
FP = 'Full-page incidence rate'
RH = 'Right-hand incidence rate'
AF = 'Above-the-fold incidence rate'
renamed = results_df[['device', 'search_engine', 'queries', 'inc_rate', 'kp_inc_rate', 'noscroll_inc_rate', 'youtube_inc_rate', 'twitter_inc_rate',]]
renamed.rename(columns={
    'device': 'Device', 'search_engine': 'Search Engine',
    'queries': 'Queries', 'inc_rate': FP,
    'kp_inc_rate': RH,
    'noscroll_inc_rate': AF,
    'youtube_inc_rate': 'Youtube incidence rate',
    'twitter_inc_rate': 'Twitter incidence rate',
}, inplace=True)
renamed.replace(to_replace={
    'top': 'common',
    'med': 'medical',
    'trend': 'trending',
}, inplace=True)
renamed.to_csv('reports/wikipedia.csv', float_format="%.2f", index=False)
renamed

# #%%
# baseline_df = results_df[['device', 'search_engine', 'queries', 'twitter_inc_rate', 'youtube_inc_rate', 'facebook_inc_rate']]
# baseline_df.rename(columns={
#     'device': 'Device', 'search_engine': 'Search Engine',
#     'queries': 'Queries'
# }, inplace=True)
# baseline_df.to_csv('reports/other_domains.csv', float_format="%.2f", index=False)



#%%
melted = renamed.melt(id_vars=['Device', 'Search Engine', 'Queries'])
melted.rename(columns={
    'variable': 'y-axis',
    'value': 'Incidence rate',
}, inplace=True)
sns.set()
g = sns.catplot(
    x="Queries", y='Incidence rate',
    hue="Search Engine", col="Device", row='y-axis',
    palette=['g', 'b', 'y'],
    order=['common', 'trending', 'medical'],
    #row_order=[FP, AF, RH],
    data=melted[melted['y-axis'] == FP], kind="bar",
    height=4, aspect=1.5, ci=None,
    sharex=False,
)
plt.savefig('reports/FP_catplot.png', dpi=300)
    
#%%
g = sns.catplot(
    x="Queries", y='Incidence rate',
    hue="Search Engine", col="Device", row='y-axis',
    palette=['g', 'b', 'y'],
    order=['common', 'trending', 'medical'],
    #row_order=[FP, AF, RH],
    data=melted[melted['y-axis'] == AF], kind="bar",
    height=4, aspect=1.5, ci=None,
    sharex=False,
)
plt.savefig('reports/FP_catplot.png', dpi=300)

#%%
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
results_df.groupby(['device', 'queries']).agg(lambda x: max(x) - min(x))['inc_rate']

#%%
# diff between FP and AF
melted[
    (melted['y-axis'] == FP) | (melted['y-axis'] == AF)
].groupby(['Device', 'Queries', 'Search Engine']).agg(lambda x: max(x) - min(x))

# which DDG cases have Right-hand but no AF?
# Pairwise differences
# rename top to common
# rename trend to trending
# rename med to medical

# %%
se_minus_se = {}
se_to_matches = {}
sub = results_df[(results_df.device == 'mobile') & (results_df.queries == 'top')]
for i, row in sub.iterrows():
    se_to_matches[row.search_engine] = set(row.matches)
se_to_matches
for k1, v1 in se_to_matches.items():
    for k2, v2 in se_to_matches.items():
        if k1 == k2:
            continue
        se_minus_se[f'{k1}_{k2}'] = v1 - v2
# what's in the first but not in the second

#%%
from pprint import pprint
pprint(se_minus_se)


# %%
