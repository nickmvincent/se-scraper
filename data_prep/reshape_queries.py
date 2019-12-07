#%%
import pandas as pd

#%%
df = pd.read_csv('google_2018_queries.csv')
df


# %%
df = df.melt(var_name='g', value_name='q')
df

# %%
df['q'] = df['q'].apply(lambda x: x[3:])


# %%
df.to_csv('google_2018_queries_melted.csv', index=False)


# %%
df.sample(10)[['q']].to_csv('google_2018_queries_melted_sample.txt', index=False, header=False)

# %%
