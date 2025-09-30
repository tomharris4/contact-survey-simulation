# Script for running bias contrbution by transmission setting (see Supplementary Material)

from matplotlib.ticker import FixedFormatter, FuncFormatter, MultipleLocator
from matplotlib.transforms import ScaledTranslation
import networkx as nx
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import copy

experiment = 'supp_exp_context'
pathogen = 'C_2_9'
metric =  'ar'

days = 100

input_network = 'NM_network_v3'
date = '2025-09-18'
attr = 'a'
N_pop = 2089388

# AGE ANALYSIS

fig = plt.figure()
ax1 = plt.subplot2grid((3, 3), (0, 0))
ax2 = plt.subplot2grid((3, 3), (0, 1))
ax3 = plt.subplot2grid((3, 3), (0, 2))
ax4 = plt.subplot2grid((3, 3), (1, 0), colspan=3, rowspan=2)


#Panel A - age contact matrices
input_params = '2'

cm_age_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__raw__' + attr + '__Overall.npy')
cm_age_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__raw__' + attr + '__Overall.npy')

cm_age = cm_age_biased - cm_age_sampled

diff_max = max([h for j in range(len(cm_age)) for h in cm_age[j]])
sb.heatmap(np.transpose(cm_age), ax=ax1, cmap="RdBu", center = 0, vmin = -1 * diff_max,  vmax = diff_max)#, cbar_kws={'label': '$\Delta$Mean # unique contacts\nper day'})

# Get the colorbar object
cbar = ax1.collections[0].colorbar

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

ax1.invert_yaxis()

ax1.set(title='School (S)', ylabel="Contact Age")
ax1.xaxis.set_major_locator(MultipleLocator(8))
ax1.xaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,40))))
ax1.yaxis.set_major_locator(MultipleLocator(8))
ax1.yaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,40))))
ax1.tick_params(which='major', pad=2, labelsize=7)

ax1.text(
        0, 1.0, 'A)', transform=(
            ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=8)

#Panel A - age contact matrices 
input_params = '3'

cm_age_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__raw__' + attr + '__Overall.npy')
cm_age_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__raw__' + attr + '__Overall.npy')

cm_age = cm_age_biased - cm_age_sampled

diff_max = max([h for j in range(len(cm_age)) for h in cm_age[j]])
sb.heatmap(np.transpose(cm_age), ax=ax2, cmap="RdBu", center = 0, vmin = -1 * diff_max,  vmax = diff_max)#, cbar_kws={'label': '$\Delta$Mean # unique contacts\nper day'})

# Get the colorbar object
cbar = ax2.collections[0].colorbar

# Apply the custom formatter to the colorbar ticks
formatter = FuncFormatter(custom_formatter)
cbar.formatter = formatter
cbar.update_ticks()

ax2.invert_yaxis()

ax2.set(title='Work (W)', xlabel="Participant Age")
ax2.xaxis.set_major_locator(MultipleLocator(8))
ax2.xaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,40))))
ax2.yaxis.set_major_locator(MultipleLocator(8))
ax2.yaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,40))))
ax2.tick_params(which='major', pad=2, labelsize=7)

ax2.text(
        0, 1.0, 'B)', transform=(
            ax2.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=8)


#Panel A - age contact matrices
input_params = '1'

cm_age_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__raw__' + attr + '__Overall.npy')
cm_age_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__raw__' + attr + '__Overall.npy')

cm_age = cm_age_biased - cm_age_sampled

diff_max = max([h for j in range(len(cm_age)) for h in cm_age[j]])
sb.heatmap(np.transpose(cm_age), ax=ax3, cmap="RdBu", center = 0, vmin = -1 * diff_max,  vmax = diff_max, cbar_kws={'label': '$\Delta$Mean # unique \ncontacts per day'})

# Get the colorbar object
cbar = ax3.collections[0].colorbar

# Apply the custom formatter to the colorbar ticks
formatter = FuncFormatter(custom_formatter)
cbar.formatter = formatter
cbar.update_ticks()

ax3.invert_yaxis()

ax3.set(title='Community (C)')
ax3.xaxis.set_major_locator(MultipleLocator(8))
ax3.xaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,40))))
ax3.yaxis.set_major_locator(MultipleLocator(8))
ax3.yaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,40))))
ax3.tick_params(which='major', pad=2, labelsize=7)

ax3.text(
        0, 1.0, 'C)', transform=(
            ax3.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=8)

attr = 'a'
input_params_all = ['0','2','3','1','6','4','5','7']#[str(h) for h in range(8)]
target_group_all = ['elderly','children','adults']

r_1 = ['None','S','W','C','S & W','S & C','W & C','All']

# #Panel C - SIR dynamics
ar_diff_overall = []
for input_params in input_params_all:
    ar_diff_overall.append(0)

ar_subpop = []

for target_group in target_group_all:
    ar_diff = []
    for input_params in input_params_all:
        if metric == 'ar':
            I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
            I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
        elif metric == 'prev':
            I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
            I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
        ar_diff.append(max(I_biased) - max(I_groundtruth))
        ar_diff_overall[len(ar_diff)-1] = ar_diff_overall[len(ar_diff)-1] + ar_diff[-1]
    if target_group == 'elderly':
        ar_subpop = copy.deepcopy(ar_diff)


ar_subpop = [h / ar_subpop[-1] for h in ar_subpop]
ar_diff_overall = [h / ar_diff_overall[-1] for h in ar_diff_overall]

ax4.plot(r_1, ar_subpop, alpha=0.5, label='Older', marker='x', linestyle='None')
ax4.plot(r_1, ar_diff_overall, alpha=0.5, lw=1.5, label='Overall', marker='o', linestyle='None')
ax4.set_xlabel('Active bias by transmission setting')

ax4.set_ylabel('Proportion of $\Delta$[Final Size]')
ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
ax4.yaxis.set_tick_params(length=0)
ax4.xaxis.set_tick_params(length=0)
legend = ax4.legend(fontsize=8, loc='upper left')
for spine in ('top', 'right', 'bottom', 'left'):
    ax4.spines[spine].set_visible(False)
ax4.set(title='Transmission Setting Contribution to $\Delta$[Final Size]')
ax4.grid()

ax4.text(
        0.0, 1.0, 'D)', transform=(
            ax4.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=8)

plt.tight_layout(w_pad=0.01)
plt.savefig('../Figures/Supplementary Material/Context_analysis_v3_' + pathogen + '_' + metric + '.pdf')

# RACE ANALYSIS

input_network = 'NM_network_v3'
date = '2025-09-18'
attr = 'r'
N_pop = 2089388

fig = plt.figure()
ax1 = plt.subplot2grid((3, 3), (0, 0))
ax2 = plt.subplot2grid((3, 3), (0, 1))
ax3 = plt.subplot2grid((3, 3), (0, 2))
ax4 = plt.subplot2grid((3, 3), (1, 0), colspan=3, rowspan=2)

race_dist_labels = ['.', 'White', 'Black', 'Asian', 'AIAN',  'NHPI','Other','Multi']


#Panel A - age contact matrices
input_params = '2'

cm_age_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__raw__' + attr + '__Overall.npy')
cm_age_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__raw__' + attr + '__Overall.npy')

cm_age = cm_age_biased - cm_age_sampled

diff_max = 0.65#max([h for j in range(len(cm_age)) for h in cm_age[j]])
sb.heatmap(np.transpose(cm_age), ax=ax1, cmap="RdBu", center = 0, vmin = -1 * diff_max,  vmax = diff_max)#, cbar_kws={'label': '$\Delta$Mean # unique contacts\nper day'})

# Get the colorbar object
cbar = ax1.collections[0].colorbar

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

ax1.invert_yaxis()

ax1.set(title='School (S)', ylabel="Contact Race")
ax1.xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax1.xaxis.set_major_formatter(FixedFormatter(race_dist_labels))
ax1.yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax1.yaxis.set_major_formatter(FixedFormatter(race_dist_labels))
ax1.tick_params(which='major', pad=2, labelsize=5,labelrotation=45)
ax1.set_yticklabels(labels=race_dist_labels,va='center')

ax1.text(
        0.0, 1.0, 'A)', transform=(
            ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=8)

#Panel A - age contact matrices 
input_params = '3'

cm_age_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__raw__' + attr + '__Overall.npy')
cm_age_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__raw__' + attr + '__Overall.npy')

cm_age = cm_age_biased - cm_age_sampled

# diff_max = max([h for j in range(len(cm_age)) for h in cm_age[j]])
sb.heatmap(np.transpose(cm_age), ax=ax2, cmap="RdBu", center = 0, vmin = -1 * diff_max,  vmax = diff_max)#, cbar_kws={'label': '$\Delta$Mean # unique contacts\nper day'})

# Get the colorbar object
cbar = ax2.collections[0].colorbar

# Apply the custom formatter to the colorbar ticks
formatter = FuncFormatter(custom_formatter)
cbar.formatter = formatter
cbar.update_ticks()

ax2.invert_yaxis()

ax2.set(title='Work (W)', xlabel="Participant Race")
ax2.xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax2.xaxis.set_major_formatter(FixedFormatter(race_dist_labels))
ax2.yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax2.yaxis.set_major_formatter(FixedFormatter(race_dist_labels))
ax2.tick_params(which='major', pad=2, labelsize=5,labelrotation=45)
ax2.set_yticklabels(labels=race_dist_labels,va='center')

ax2.text(
        0.0, 1.0, 'B)', transform=(
            ax2.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=8)

#Panel A - age contact matrices
input_params = '1'

cm_age_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__raw__' + attr + '__Overall.npy')
cm_age_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__raw__' + attr + '__Overall.npy')

cm_age = cm_age_biased - cm_age_sampled

# diff_max = max([h for j in range(len(cm_age)) for h in cm_age[j]])
sb.heatmap(np.transpose(cm_age), ax=ax3, cmap="RdBu", center = 0, vmin = -1 * diff_max,  vmax = diff_max, cbar_kws={'label': '$\Delta$Mean # unique \ncontacts per day'})

# Get the colorbar object
cbar = ax3.collections[0].colorbar

# Apply the custom formatter to the colorbar ticks
formatter = FuncFormatter(custom_formatter)
cbar.formatter = formatter
cbar.update_ticks()

ax3.invert_yaxis()

ax3.set(title='Community (C)')
ax3.xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax3.xaxis.set_major_formatter(FixedFormatter(race_dist_labels))
ax3.yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax3.yaxis.set_major_formatter(FixedFormatter(race_dist_labels))
ax3.tick_params(which='major', pad=2, labelsize=5,labelrotation=45)
ax3.set_yticklabels(labels=race_dist_labels,va='center')

ax3.text(
        0.0, 1.0, 'C)', transform=(
            ax3.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=8)

attr = 'ar'
input_params_all = ['0','2','3','1','6','4','5','7']#[str(h) for h in range(8)]
target_group_all = ['White','Non-White']

r_1 = ['None','S','W','C','S & W','S & C','W & C','All']

# #Panel C - SIR dynamics
ar_diff_overall = []
ar_diff_white = []
ar_diff_non_white = []

for input_params in input_params_all:
    ar_diff_overall.append([])
    ar_diff_white.append([])
    ar_diff_non_white.append([])
ar_diff_overall = []
for input_params in input_params_all:
    ar_diff_overall.append(0)

ar_subpop = []

for target_group in target_group_all:
    ar_diff = []
    for input_params in input_params_all:
        if metric == 'ar':
            I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_' + '*' +'_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
            I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_' + '*' +'_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
        elif metric == 'prev':
            I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
            I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
        ar_diff.append(max(I_biased) - max(I_groundtruth))
        ar_diff_overall[len(ar_diff)-1] = ar_diff_overall[len(ar_diff)-1] + ar_diff[-1]
    if target_group == 'Non-White':
        ar_subpop_nw = copy.deepcopy(ar_diff)
    else:
        ar_subpop_w = copy.deepcopy(ar_diff)

ar_diff_overall = [h / ar_diff_overall[-1] for h in ar_diff_overall]

ar_subpop_nw = [h / ar_subpop_nw[-1] for h in ar_subpop_nw]
ar_subpop_w = [h / ar_subpop_w[-1] for h in ar_subpop_w]

ax4.plot(r_1, ar_subpop_nw, alpha=0.5, label='non-White', marker='x', linestyle='None')
ax4.plot(r_1, ar_diff_overall, alpha=0.5, lw=1.5, label='Overall', marker='o', linestyle='None')

ax4.set_xlabel('Active bias by transmission setting')

ax4.set_ylabel('Proportion of $\Delta$[Final Size]')
ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
ax4.yaxis.set_tick_params(length=0)
ax4.xaxis.set_tick_params(length=0)
legend = ax4.legend(fontsize=8, loc='upper left')
for spine in ('top', 'right', 'bottom', 'left'):
    ax4.spines[spine].set_visible(False)
ax4.set(title='Transmission Setting Contribution to $\Delta$[Final Size]')
ax4.grid()

ax4.text(
        0.0, 1.0, 'D)', transform=(
            ax4.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=8)


plt.tight_layout(w_pad=0.01)
plt.savefig('../Figures/Supplementary Material/Context_analysis_ar_v3_' + pathogen + '_' + metric + '.pdf')