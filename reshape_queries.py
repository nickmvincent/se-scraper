#%%
import pandas as pd
load_folder = 'search_queries'
save_folder = 'search_queries/prepped'
n = 10

#%%
from data_prep.georgetown_medical_bing import MED_QUERIES
med_df = pd.DataFrame(MED_QUERIES)
med_df

#%%
med_df[0].to_csv(f'{save_folder}/med.txt', index=False, header=False)
med_df[0].sample(n).to_csv(f'{save_folder}/med_sample.txt', index=False, header=False)


#%%
top_df = pd.read_csv(f'{load_folder}/ahrefs_top2019_google.txt', delimiter='\t', header=None)
top_df

#%%
top_df[1].to_csv(f'{save_folder}/top.txt', index=False, header=False)
top_df[1].sample(n).to_csv(f'{save_folder}/top_sample.txt', index=False, header=False)


#%%
trends_df = pd.read_csv(f'{load_folder}/google_2018_queries.csv')
trends_df


# %%
melted = trends_df.melt(var_name='g', value_name='q')
melted.head()

# %%
melted['q'] = melted['q'].apply(lambda x: x[3:])
melted.head()


# %%
melted.to_csv('google_2018_queries_melted.csv', index=False)


# %%
melted.sample(n)[['q']].to_csv(f'{save_folder}/trend_sample.txt', index=False, header=False)
melted[['q']].to_csv(f'{save_folder}/trend.txt', index=False, header=False)


# %%
