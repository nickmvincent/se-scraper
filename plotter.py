
#%%
# defaults
import json
import glob
import sys
from collections import defaultdict
from urllib.parse import urlparse
from pprint import pprint


# plotting
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# scipy
import pandas as pd
import numpy as np

from PIL import Image
infinite_defaultdict = lambda: defaultdict(infinite_defaultdict)



#%%
# Display parameters
full_width = 8
LH_W = 780

IPHONE_SE_H = 568
IPHONE_6_H = 667
IPHONE_X_H = 812

#110% zoom
MACBOOK13_11_H = 717
MACBOOK13_FULL_H = 789
# 90% zoom
MACBOOK13_9_H = 877

mobile_lines = {
    'noscroll_lb': IPHONE_SE_H,
    'noscroll_mg': IPHONE_6_H,
    'noscroll_ub': IPHONE_X_H
}

desktop_lines = {
    'noscroll_lb': MACBOOK13_11_H,
    'noscroll_mg': MACBOOK13_FULL_H,
    'noscroll_ub': MACBOOK13_9_H,
}

BORDER_PIX = 1440

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
        kp_line = LH_W / right_max
        # source: 

        if mobile:
            # no right-hand incidence
            df[f'{domain}_appears_rh'] = 0
            # no lefthand above-the-fold incidence
            df[f'{domain}_appears_lh'] = 0
            for name, line in mobile_lines.items():
                mobile_noscroll_line = line / bot_max

                df[f'{domain}_appears_{name}'] = (
                    (df[f'{domain}_appears']) &
                    (df.norm_top < mobile_noscroll_line)
                )

                df[f'{domain}_appears_lh_{name}'] = 0

        else:
            df[f'{domain}_appears_rh'] = (
                (df[f'{domain}_appears']) &
                (df.norm_left > kp_line)
            )

            df[f'{domain}_appears_lh'] = (
                (df[f'{domain}_appears']) &
                (df.norm_left <= kp_line)
            )

            for name, line in desktop_lines.items():
                noscroll_line = line / bot_max

                df[f'{domain}_appears_{name}'] = (
                    (df[f'{domain}_appears']) &
                    (df.norm_top < noscroll_line)
                )

                df[f'{domain}_appears_lh_{name}'] = (
                    (df[f'{domain}_appears_lh']) &
                    (df.norm_top < noscroll_line)
                )
    return df


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
itera = 6
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
DO_COORDS = True
if DO_COORDS:
    for config in configs:
        device = config['device']
        search_engine = config['search_engine']
        queries = config['queries']
        if search_engine != 'google':
            continue
        if queries != 'top':
            continue

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
                # x = row['norm_left']
                # y = row['norm_bottom']
                # width = row['norm_width']
                # height = row['norm_height']
                x = row['left']
                y = row['bottom']
                width = row['width']
                height = row['height']
                domain = row['domain']

                if row['wikipedia_appears']:
                    add_last.append([domain, (x,y,), width, height])
                else:
                    # if row['platform_ugc']:
                    #     color = 'b'
                    if 'google' in domain or 'bing' in domain or 'duckduckgo' in domain:
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

            # kp line = lefthand width border.
            kp_line = LH_W
            if device == 'mobile':
                scroll_line = mobile_lines['noscroll_mg']
            else:
                scroll_line = desktop_lines['noscroll_mg']
            #scroll_line /= bot_max
            plt.axvline(kp_line, color='r', linestyle='-')

            #border_line = BORDER_PIX / right_max
            #plt.axvline(border_line, color='k', linestyle='-')
            plt.axhline(scroll_line, color='k', linestyle='-')

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

    inc_rate = df.groupby('query').wikipedia_appears.agg(any).mean()
    rh_inc_rate = df.groupby('query').wikipedia_appears_rh.agg(any).mean()
    lh_inc_rate = df.groupby('query').wikipedia_appears_lh.agg(any).mean()


    if device == 'mobile':
        d = mobile_lines
    else:
        d = desktop_lines
    matches = set(df[df.wikipedia_appears == True]['query'])

    row_dict = {
        'queries': queries,
        'search_engine': search_engine,
        'device': device,
        'inc_rate': inc_rate,
        'rh_inc_rate': rh_inc_rate,
        'lh_inc_rate': lh_inc_rate,
        'matches': matches
    }
    for name in d.keys():
        row_dict[f'{name}_inc_rate'] = df.groupby('query')[f'wikipedia_appears_{name}'].agg(any).mean()
        row_dict[f'lh_{name}_inc_rate'] = df.groupby('query')[f'wikipedia_appears_lh_{name}'].agg(any).mean()
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
FP = 'Full-page incidence'
RH = 'Right-hand incidence'
LH = 'Left-hand incidence'
AF_MG = 'Above-the-fold incidence'
AF_pretty = 'Above-the-fold incidence (lower bound - upper bound)'

LH_AF_pretty = 'Left-hand above-the-fold incidence (lower bound - upper bound)'
LH_AF_LB = 'Left-hand above-the-fold incidence (lower bound)' 
LH_AF_MG = 'Left-hand above-the-fold incidence'
LH_AF_UB = 'Left-hand above-the-fold incidence (upper bound)' 

AF_LB = 'Above-the-fold incidence (lower bound)'
AF_UB = 'Above-the-fold incidence (upper bound)'


cols = [
    'device', 'search_engine', 'queries', 'inc_rate', 'rh_inc_rate',
    'lh_inc_rate',
]
for name in mobile_lines.keys():
    cols += [f'{name}_inc_rate', f'lh_{name}_inc_rate']
print(cols)

renamed = results_df[cols]
renamed.rename(columns={
    'device': 'Device', 'search_engine': 'Search Engine',
    'queries': 'Query Category', 'inc_rate': FP,
    'rh_inc_rate': RH,
    'lh_inc_rate': LH,
    'lh_noscroll_lb_inc_rate': LH_AF_LB,
    'lh_noscroll_mg_inc_rate': LH_AF_MG,
    'lh_noscroll_ub_inc_rate': LH_AF_UB,
    'noscroll_lb_inc_rate': AF_LB,
    'noscroll_mg_inc_rate': AF_MG,
    'noscroll_ub_inc_rate': AF_UB,
    'youtube_inc_rate': 'Youtube incidence rate',
    'twitter_inc_rate': 'Twitter incidence rate',
}, inplace=True)

def pretty_bounds(row):
    mg = row[AF_MG]
    lb = row[AF_LB]
    ub = row[AF_UB]
    return f'{mg:.2f} ({lb:.2f} - {ub:.2f})'

def pretty_bounds_lh(row):
    mg = row[LH_AF_MG]
    lb = row[LH_AF_LB]
    ub = row[LH_AF_UB]
    return f'{mg:.2f} ({lb:.2f} - {ub:.2f})'

renamed[AF_pretty] = renamed.apply(pretty_bounds, axis=1)
renamed[LH_AF_pretty] = renamed.apply(pretty_bounds_lh, axis=1)

renamed.replace(to_replace={
    'top': 'common',
    'med': 'medical',
    'trend': 'trending',
}, inplace=True)
renamed

renamed[[
    'Device', 'Search Engine', 'Query Category',
    FP, RH, LH, AF_pretty, LH_AF_pretty
]].to_csv('reports/main.csv', float_format="%.2f", index=False)

#%%
renamed

#%%
baseline_df = results_df[['device', 'search_engine', 'queries', 'twitter_inc_rate', 'youtube_inc_rate', 'facebook_inc_rate']]
baseline_df.rename(columns={
    'device': 'Device', 'search_engine': 'Search Engine',
    'queries': 'Queries'
}, inplace=True)
baseline_df.to_csv('reports/other_domains.csv', float_format="%.2f", index=False)



#%%
melted = renamed.melt(id_vars=['Device', 'Search Engine', 'Query Category'])
melted.rename(columns={
    'variable': 'y-axis',
    'value': 'Incidence rate',
}, inplace=True)
sns.set()
g = sns.catplot(
    x="Query Category", y='Incidence rate',
    hue="Search Engine", col="Device", row='y-axis',
    palette=['g', 'b', 'y'],
    order=['common', 'trending', 'medical'],
    #row_order=[FP, AF, RH],
    data=melted[melted['y-axis'] == FP], kind="bar",
    height=3, aspect=1.5, ci=None,
    sharex=False,
)
plt.savefig('reports/FP_catplot.png', dpi=300)

#%%
# lh vs rh
g = sns.catplot(
    x="Query Category", y='Incidence rate',
    hue="Search Engine", col='y-axis',
    col_order=[LH, RH],
    palette=['g', 'b', 'y'],
    order=['common', 'trending', 'medical'],
    data=melted[
        ((melted['y-axis'] == LH) | (melted['y-axis'] == RH))
        & (melted['Device'] == 'desktop')],
    kind="bar",
    height=3, aspect=1.5, ci=None,
    sharex=False,
)
plt.savefig('reports/LHRH_catplot.png', dpi=300)
#%%
g = sns.catplot(
    x="Query Category", y='Incidence rate',
    hue="Search Engine", col="Device", row='y-axis',
    palette=['g', 'b', 'y'],
    order=['common', 'trending', 'medical'],
    #row_order=[FP, AF, RH],
    data=melted[melted['y-axis'] == AF_MG], kind="bar",
    height=3, aspect=1.5, ci=None,
    sharex=False,
)
plt.savefig('reports/AF_catplot.png', dpi=300)

#%%
g = sns.catplot(
    x="Queries", y='Incidence rate',
    hue="Search Engine", col="Device", row='y-axis',
    palette=['g', 'b', 'y'],
    order=['common', 'trending', 'medical'],
    data=melted[melted['y-axis'] == LH_AF_MG], kind="bar",
    height=2, aspect=1, ci=None,
    sharex=False,
)
plt.savefig('reports/LH_AF_catplot.png', dpi=300)


#results_df[['device', 'queries', 'search_engine', 'inc_rate', 'matches']]


# %%
# differences between search engines
results_df.groupby(['device', 'queries']).agg(lambda x: max(x) - min(x))['inc_rate']


#%%
# differences between devices
results_df.groupby(['search_engine', 'queries']).agg(lambda x: max(x) - min(x))['inc_rate']

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
pprint(se_minus_se)


# %%
