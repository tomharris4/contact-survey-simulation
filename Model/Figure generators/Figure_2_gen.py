# Script for generating Figure 2 plot

from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedFormatter, MultipleLocator
from matplotlib.transforms import ScaledTranslation
import networkx as nx
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 7

input_network = 'NM_network'
context = 'Overall' # Transmission setting
processed = False

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6,5.2))

#Panel A - age
attr = 'a'
if not processed:
    cm_age = np.load('../Data/Contact matrices/' + input_network + '__full_pop__'  + attr + '__' + context + '.npy')
    sb.heatmap(np.transpose(cm_age), ax=ax[0,0], vmin = 0, cbar_kws={'label': 'Mean # unique contacts per day'})
else:
    cm_age = np.load('../Data/Contact matrices/' + input_network + '__full_pop__processed__'  + attr + '__' + context + '.npy')
    sb.heatmap(np.transpose(cm_age), ax=ax[0,0], vmin = 0, cbar_kws={'label': 'Per-capita contact rate'})


ax[0,0].invert_yaxis()

ax[0,0].set(title='Age', xlabel="Participant Age", ylabel="Contact Age")
ax[0,0].xaxis.set_major_locator(MultipleLocator(2))
ax[0,0].xaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,10))))
ax[0,0].yaxis.set_major_locator(MultipleLocator(2))
ax[0,0].yaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,10))))
ax[0,0].tick_params(which='major', pad=2, labelsize=7)

ax[0,0].text(
        0.0, 1.0, 'A)', transform=(
            ax[0,0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)

#Panel C - ethnicity
attr = 'e'
if not processed:
    cm_eth = np.load('../Data/Contact matrices/' + input_network + '__full_pop__'  + attr + '__' + context + '.npy')
    sb.heatmap(np.transpose(cm_eth), ax=ax[1,0], vmin = 0, cbar_kws={'label': 'Mean # unique contacts per day'})
else:
    cm_eth = np.load('../Data/Contact matrices/' + input_network + '__full_pop__processed__'  + attr + '__' + context + '.npy')
    sb.heatmap(np.transpose(cm_eth), ax=ax[1,0], vmin = 0, cbar_kws={'label': 'Per-capita contact rate'})

ax[1,0].invert_yaxis()

ax[1,0].set(title='Ethnicity', xlabel="Participant Ethnicity", ylabel="Contact Ethnicity")
eth_dist_labels = ['.', 'Non-Hispanic','Hispanic']
ax[1,0].xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[1,0].xaxis.set_major_formatter(FixedFormatter(eth_dist_labels))
ax[1,0].yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[1,0].yaxis.set_major_formatter(FixedFormatter(eth_dist_labels))
ax[1,0].tick_params(which='major', pad=2, labelsize=7,labelrotation=45)
ax[1,0].set_yticklabels(labels=eth_dist_labels,va='center')

ax[1,0].text(
        0.0, 1.0, 'C)', transform=(
            ax[1,0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)

#Panel B - race
attr = 'r'
if not processed:
    cm_race = np.load('../Data/Contact matrices/' + input_network + '__full_pop__'  + attr + '__' + context + '.npy')
    sb.heatmap(np.transpose(cm_race), ax=ax[0,1], vmin = 0, cbar_kws={'label': 'Mean # unique contacts per day'})
else:
    cm_race = np.load('../Data/Contact matrices/' + input_network + '__full_pop__processed__'  + attr + '__' + context + '.npy')
    sb.heatmap(np.transpose(cm_race), norm=LogNorm(), ax=ax[0,1],  cbar_kws={'label': 'Per-capita contact rate'})

ax[0,1].invert_yaxis()

race_dist_labels = ['.', 'White', 'Black', 'Asian', 'AIAN',  'NHPI','Other','Multi']
ax[0,1].set(title='Race', xlabel="Participant Race", ylabel="Contact Race")
ax[0,1].xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[0,1].xaxis.set_major_formatter(FixedFormatter(race_dist_labels))
ax[0,1].yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[0,1].yaxis.set_major_formatter(FixedFormatter(race_dist_labels))
ax[0,1].tick_params(which='major', pad=2, labelsize=7,labelrotation=45)
ax[0,1].set_yticklabels(labels=race_dist_labels,va='center')

ax[0,1].text(
        0.0, 1.0, 'B)', transform=(
            ax[0,1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)

#Panel D - income
attr = 's'
if not processed:
    cm_income = np.load('../Data/Contact matrices/' + input_network + '__full_pop__'  + attr + '__' + context + '.npy')
    sb.heatmap(np.transpose(cm_income), ax=ax[1,1], vmin = 0, cbar_kws={'label': 'Mean # unique contacts per day'})
else:
    cm_income = np.load('../Data/Contact matrices/' + input_network + '__full_pop__processed__'  + attr + '__' + context + '.npy')
    sb.heatmap(np.transpose(cm_income), ax=ax[1,1], vmin = 0, cbar_kws={'label': 'Per-capita contact rate'})

ax[1,1].invert_yaxis()

ax[1,1].set(title='Income', xlabel="Participant Income", ylabel="Contact Income")
income_dist_labels = ['.', 'Low', 'Medium', 'High']
ax[1,1].xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[1,1].xaxis.set_major_formatter(FixedFormatter(income_dist_labels))
ax[1,1].yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[1,1].yaxis.set_major_formatter(FixedFormatter(income_dist_labels))
ax[1,1].tick_params(which='major', pad=2, labelsize=7,labelrotation=45)
ax[1,1].set_yticklabels(labels=income_dist_labels,va='center')

ax[1,1].text(
        0.0, 1.0, 'D)', transform=(
            ax[1,1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)

plt.tight_layout()
plt.savefig('../Figures/figure2_' + context + '_final.pdf')
