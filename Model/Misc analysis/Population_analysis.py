# Script for generating population summary (see Supplementary Material)

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

agents = pd.read_csv('../Data/Synthetic population/urbanpop_network_nm_001_processed.csv')

median_national_income = 70784
mean_hh_size = 2.51

med_us_adj_hh_income = median_national_income / mean_hh_size**0.5
ses_quantile_low = (2/3) * med_us_adj_hh_income 
ses_quantile_high = 2 * med_us_adj_hh_income

agents_hh = agents.groupby(by=['household_id']).count()
hh_size_lookup = dict(zip(agents_hh.index,agents_hh['household_income']))

agents['hh_size_scaler'] = [hh_size_lookup[h]**0.5 for h in agents['household_id']]

agents['household_income_adjusted'] = agents['household_income'] / agents['hh_size_scaler']
agents['household_income_adjusted'] = np.clip(a=agents['household_income_adjusted'], a_min=0, a_max=155000)

fig, ax = plt.subplots(nrows=2, ncols=2)

# Age: overall
sb.histplot(agents, x='age', stat='percent',binwidth=5, binrange=(0,95), ax=ax[0,0], color='#1f77b4')
ax[0,0].set(title='Age', xlabel="Age", ylabel="Percentage of total \npopulation (%)")

# Race: overall
race_dist_labels = ['White', 'Black', 'Asian', 'AIAN',  'NHPI','Other','Multi']
sb.countplot(agents, x='race', stat='percent', ax=ax[1,0], color='#1f77b4')
ax[1,0].set(title='Race', xlabel="Race", ylabel="Percentage of total \npopulation (%)")
ax[1,0].xaxis.set_tick_params(rotation=45,pad=13)
ax[1,0].set_xticklabels(labels=race_dist_labels,va='center')
for patch in ax[1,0].patches:
    patch.set_linewidth(1)
    patch.set_edgecolor('black')

# Ethnicity: overall
eth_dist_labels = ['Non-Hispanic', 'Hispanic']
sb.countplot(agents, x='ethnicity', stat='percent', ax=ax[0,1], color='#1f77b4')
ax[0,1].set(title='Ethnicity', xlabel="Ethnicity", ylabel="Percentage of total \npopulation (%)")
ax[0,1].xaxis.set_tick_params(pad=10)
ax[0,1].set_xticklabels(labels=eth_dist_labels,va='center')
for patch in ax[0,1].patches:
    patch.set_linewidth(1)
    patch.set_edgecolor('black')


# Income: overall
sb.histplot(agents, x='household_income_adjusted', stat='percent',binwidth=10000, binrange=(0,160000), ax=ax[1,1], color='#1f77b4')
ax[1,1].set(title='Income', xlabel="Income", ylabel="Percentage of total \npopulation (%)")
ax[1,1].axvline(x=ses_quantile_low, color='red', linestyle='dashed')
ax[1,1].axvline(x=ses_quantile_high, color='red', linestyle='dashed')
ax[1,1].set_ylim(top=25)
ax[1,1].text(x=-6000,y=22,s='Lower',color='red')
ax[1,1].text(x=40000,y=22,s='Middle',color='red')
ax[1,1].text(x=110000,y=22,s='Upper',color='red')

for spine in ('top', 'right'):
    ax[1,1].spines[spine].set_visible(False)
    ax[0,1].spines[spine].set_visible(False)
    ax[0,0].spines[spine].set_visible(False)
    ax[1,0].spines[spine].set_visible(False)



fig.tight_layout()
fig.savefig('../Figures/Supplementary Material/Supp_population_distribution.pdf')

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,3))

# agents['race_subset'] = [h if h not in  [0,1,2,4,6] else None for h in agents['race']]
agents['Race'] = ['Non-White' if h!= 0 else 'White' for h in agents['race']]

# Race: age
sb.histplot(agents, x='age', stat='percent', hue='Race', binwidth=5, binrange=(0,95), ax=ax[0])
ax[0].set(title='Race-Age', xlabel="Age", ylabel="Percentage of total \npopulation (%)")
ax[0].legend(title='', loc='upper center', bbox_to_anchor=(0.5, -0.22),
          fancybox=False, labels=['Non-White', 'White'], ncol=2,frameon=False)


# Ethnicity: age
sb.histplot(agents, x='age', stat='percent', hue='ethnicity', binwidth=5, binrange=(0,95), ax=ax[1])
ax[1].set(title='Ethnicity-Age', xlabel="Age", ylabel="Percentage of total \npopulation (%)")
ax[1].legend(title='', loc='upper center', bbox_to_anchor=(0.5, -0.22),
          fancybox=False, labels=['Hispanic', 'Non-Hispanic'], ncol=2,frameon=False)

# Income: age
sb.histplot(agents, x='age', stat='percent', hue='income', binwidth=5, binrange=(0,95), ax=ax[2])
ax[2].set(title='Income-Age', xlabel="Age", ylabel="Percentage of total \npopulation (%)")
ax[2].legend(title='', loc='upper center', bbox_to_anchor=(0.5, -0.22),
          fancybox=False, labels=['Upper', 'Middle', 'Lower'], ncol=2, frameon=False)


fig.tight_layout()
fig.savefig('../Figures/Supplementary Material/Supp_population_age_distribution.pdf')