# Script for running bias contrbution by transmission setting (see Supplementary Material)

from matplotlib.ticker import FixedFormatter, FuncFormatter, MultipleLocator
from matplotlib.transforms import ScaledTranslation
import networkx as nx
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd

# AGE ANALYSIS

experiment = 'exp1'
pathogen = 'C_2_9'
metric =  'ar'

input_network = 'NM_network'
attr = 'a'
N_pop = 2089388

fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(8.27,11.69))

input_params = ['Overall','Household','Community','School','Workplace',]

input_bias = '4'


for i in range(len(input_params)):
    cm_age_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_bias + '_*' + '__biased__raw__' + attr + '__' + input_params[i] + '.npy')
    cm_age_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_bias + '_*' + '__gt__raw__' + attr + '__' + input_params[i] + '.npy')

    # sb.heatmap(np.transpose(cm_age_sampled), ax=ax[i][0], vmin = 0, cbar_kws={'label': 'Mean # unique contacts per day\n'})
    sb.heatmap(np.transpose(cm_age_biased), ax=ax[i][0], vmin = 0, cbar_kws={'label': 'Mean # unique contacts \n per day\n'})

        
    cm_age = cm_age_biased - cm_age_sampled

    diff_max = round(max([h for j in range(len(cm_age)) for h in cm_age[j]]),1) + 0.1 
    sb.heatmap(np.transpose(cm_age), ax=ax[i][1], cmap="RdBu", center = 0, vmin = -1 * diff_max, vmax = diff_max, cbar_kws={'label': '$\Delta$Mean # unique contacts \n per day\n'})

    ax[i][0].invert_yaxis()

    ax[i][0].set(title='Biased', xlabel="Participant Age", ylabel=input_params[i].upper() + "\n\n Contact Age")
    ax[i][0].xaxis.set_major_locator(MultipleLocator(8))
    ax[i][0].xaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,40))))
    ax[i][0].yaxis.set_major_locator(MultipleLocator(8))
    ax[i][0].yaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,40))))
    ax[i][0].tick_params(which='major', pad=2, labelsize=7)

    # ax[i][1].invert_yaxis()

    # ax[i][1].set(title='Biased', xlabel="Participant Age", ylabel="Contact Age")
    # ax[i][1].xaxis.set_major_locator(MultipleLocator(8))
    # ax[i][1].xaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,40))))
    # ax[i][1].yaxis.set_major_locator(MultipleLocator(8))
    # ax[i][1].yaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,40))))
    # ax[i][1].tick_params(which='major', pad=2, labelsize=7)

    # Get the colorbar object
    cbar = ax[i][1].collections[0].colorbar

    # Define a custom formatter function
    def custom_formatter(x, pos):
        if x > 0:
            return f'+{x:.1f}' # Add '+' for positive values, format as integer
        else:
            return f'{x:.1f}' # Format negative values as integer

    # Apply the custom formatter to the colorbar ticks
    formatter = FuncFormatter(custom_formatter)
    cbar.formatter = formatter
    cbar.update_ticks()

    ax[i][1].invert_yaxis()

    ax[i][1].set(title='$\Delta$[Biased - True]', xlabel="Participant Age", ylabel="Contact Age")
    ax[i][1].xaxis.set_major_locator(MultipleLocator(8))
    ax[i][1].xaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,40))))
    ax[i][1].yaxis.set_major_locator(MultipleLocator(8))
    ax[i][1].yaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,40))))
    ax[i][1].tick_params(which='major', pad=2, labelsize=7)

    # ax[0][2].text(
    #         0, 1.0, 'A)', transform=(
    #             ax[0][0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
    #         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)

plt.tight_layout()
plt.savefig('../Figures/Supplementary Material/bias_impact_per_setting_age.pdf')
# plt.show()
# exit()


# RACE ANALYSIS

experiment = 'exp2'
pathogen = 'C_2_9'
metric =  'ar'

input_network = 'NM_network'
attr = 'r'
N_pop = 2089388

fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(8.27,11.69))

input_params = ['Overall','Household','Community','School','Workplace',]

input_bias = 'tract_7'


for i in range(len(input_params)):
    cm_age_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_bias + '_*' + '__biased__raw__' + attr + '__' + input_params[i] + '.npy')
    cm_age_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_bias + '_*' + '__gt__raw__' + attr + '__' + input_params[i] + '.npy')

    # sb.heatmap(np.transpose(cm_age_sampled), ax=ax[i][0], vmin = 0, cbar_kws={'label': 'Mean # unique contacts per day\n'})
    sb.heatmap(np.transpose(cm_age_biased), ax=ax[i][0], vmin = 0, cbar_kws={'label': 'Mean # unique contacts \n per day\n'})

        
    cm_age = cm_age_biased - cm_age_sampled

    diff_max = round(max([h for j in range(len(cm_age)) for h in cm_age[j]]),1) + 0.1 
    sb.heatmap(np.transpose(cm_age), ax=ax[i][1], cmap="RdBu", center = 0, vmin = -1 * diff_max,  vmax = 1 * diff_max, cbar_kws={'label': '$\Delta$Mean # unique contacts \n per day\n'})

    ax[i][0].invert_yaxis()

    ax[i][0].set(title='Biased', xlabel="Participant Race", ylabel=input_params[i].upper() + "\n\n Contact Race")
    race_dist_labels = ['.', 'White', 'Black', 'Asian', 'AIAN',  'NHPI','Other','Multi']
    ax[i][0].xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
    ax[i][0].xaxis.set_major_formatter(FixedFormatter(race_dist_labels))
    ax[i][0].yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
    ax[i][0].yaxis.set_major_formatter(FixedFormatter(race_dist_labels))
    ax[i][0].tick_params(which='major', pad=2, labelsize=7,labelrotation=45)
    ax[i][0].set_yticklabels(labels=race_dist_labels,va='center')

    # Get the colorbar object
    cbar = ax[i][1].collections[0].colorbar

    # Define a custom formatter function
    def custom_formatter(x, pos):
        if x > 0:
            return f'+{x:.1f}' # Add '+' for positive values, format as integer
        else:
            return f'{x:.1f}' # Format negative values as integer

    # Apply the custom formatter to the colorbar ticks
    formatter = FuncFormatter(custom_formatter)
    cbar.formatter = formatter
    cbar.update_ticks()

    ax[i][1].invert_yaxis()

    ax[i][1].set(title='$\Delta$[Biased - True]', xlabel="Participant Race", ylabel="Contact Race")
    ax[i][1].xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
    ax[i][1].xaxis.set_major_formatter(FixedFormatter(race_dist_labels))
    ax[i][1].yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
    ax[i][1].yaxis.set_major_formatter(FixedFormatter(race_dist_labels))
    ax[i][1].tick_params(which='major', pad=2, labelsize=7,labelrotation=45)
    ax[i][1].set_yticklabels(labels=race_dist_labels,va='center')

    # ax[0][2].text(
    #         0, 1.0, 'A)', transform=(
    #             ax[0][0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
    #         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)


plt.tight_layout()
plt.savefig('../Figures/Supplementary Material/bias_impact_per_setting_race.pdf')

# AGE GROUP ANALYSIS


experiment = 'exp1'
pathogen = 'C_2_9'
metric =  'ar'

input_network = 'NM_network'
attr = 'a'
age_pop_dist = [120467, 133396, 141965, 136600, 130735, 134971, 138814, 128595,
       118190, 122808, 133518, 143473, 139797, 124096,  94146,  64665,
        43637,  39515]

age_pop_dist_reduced = [sum(age_pop_dist[0:4]), sum(age_pop_dist[4:13]), sum(age_pop_dist[13:])]

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8.27/1.5,11.69/1.5))

input_bias = '4'

cm_age_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_bias + '_*' + '__biased__raw__' + attr + '__Overall.npy')
cm_age_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_bias + '_*' + '__gt__raw__' + attr + '__Overall.npy')

for i in range(len(cm_age_biased)):
    for j in range(len(cm_age_biased)):
        cm_age_biased[i][j] = cm_age_biased[i][j] * age_pop_dist[i] * (10000 / N_pop)
        cm_age_sampled[i][j] = cm_age_sampled[i][j] * age_pop_dist[i] * (10000 / N_pop)

cm_age_biased = [ sum(sum(cm_age_biased[:,0:4])), sum(sum(cm_age_biased[:,4:13])), sum(sum(cm_age_biased[:,13:]))]
cm_age_sampled = [ sum(sum(cm_age_sampled[:,0:4])), sum(sum(cm_age_sampled[:,4:13])), sum(sum(cm_age_sampled[:,13:]))]


# cm_age = [ (cm_age_biased[h] - cm_age_sampled[h]) for h in range(len(cm_age_biased))]

# data = pd.DataFrame()

# data['Age group'] = ['Children', 'Adults', 'Older Adults']
# data['Change in reported contacts'] = cm_age

# sb.barplot(data, x='Age group', y='Change in reported contacts',ax=ax[0])

# ax[0].set(title='Change in average number of reported contacts \n due to age perception bias')

cm_age = [ (cm_age_biased[h] - cm_age_sampled[h]) / age_pop_dist_reduced[h] for h in range(len(cm_age_biased))]

data = pd.DataFrame()

data['Age group'] = ['Children', 'Adults', 'Older Adults']
data['Normalised change in reported contacts'] = cm_age

sb.barplot(data, x='Age group', y='Normalised change in reported contacts',ax=ax[0])

ax[0].set(title='Normalised change in average number of reported contacts \n due to age perception bias')

# plt.savefig('../Figures/Supplementary Material/perc_change_reported_contacts_age.pdf')


# RACE GROUP ANALYSIS

experiment = 'exp2'
pathogen = 'C_2_9'
metric =  'ar'

input_network = 'NM_network'
attr = 'r'
N_pop = 2089388
race_pop_dist = [1568835,   42843,   32668,  199247,    1454,  178453,   65888]

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11,4))

input_bias = 'tract_7'

cm_race_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_bias + '_*' + '__biased__raw__' + attr + '__Overall.npy')
cm_race_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_bias + '_*' + '__gt__raw__' + attr + '__Overall.npy')

for i in range(len(cm_race_biased)):
    for j in range(len(cm_race_biased)):
        cm_race_biased[i][j] = cm_race_biased[i][j] * race_pop_dist[i] * (10000 / N_pop)
        cm_race_sampled[i][j] = cm_race_sampled[i][j] * race_pop_dist[i] * (10000 / N_pop)

cm_race_biased = [ sum(cm_race_biased[:,h]) for h in range(len(cm_race_biased))]
cm_race_sampled = [ sum(cm_race_sampled[:,h]) for h in range(len(cm_race_sampled))]

# cm_race = [ (cm_race_biased[h] - cm_race_sampled[h])   for h in range(len(cm_race_biased))] #/ cm_race_sampled[h]

# data = pd.DataFrame()

# data['Racial group'] = ['White', 'Black', 'Asian', 'AIAN', 'NHPI', 'Other', 'Multi']
# data['Change in reported contacts'] = cm_race

# sb.barplot(data, x='Racial group', y='Change in reported contacts',ax=ax[1])

# ax[1].set(title='Change in average number of reported contacts \n due to racial perception bias')

cm_race = [ (cm_race_biased[h] - cm_race_sampled[h]) / race_pop_dist[h]  for h in range(len(cm_race_biased))]

data = pd.DataFrame()

data['Racial group'] = ['White', 'Black', 'Asian', 'AIAN', 'NHPI', 'Other', 'Multi']
data['Normalised change in reported contacts'] = cm_race

sb.barplot(data, x='Racial group', y='Normalised change in reported contacts',ax=ax[1])

ax[1].set(title='Normalised change in average number of reported contacts \n due to racial perception bias')

plt.tight_layout()
plt.savefig('../Figures/Supplementary Material/perc_change_in_reported_contacts.pdf')

