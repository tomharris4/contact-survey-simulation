# Script for generating ethnicity bias analysis (see Supplementary Material)

from matplotlib.ticker import FixedFormatter, MultipleLocator
from matplotlib.transforms import ScaledTranslation
import networkx as nx
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt

experiment = 'supp_eth'
pathogen = 'C_2_9'
metric =  'ar'

days = 100

input_network = 'NM_network'
attr = 'e'

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,2.5))

#Panel A - ethnicity contact matrices
input_params = 'tract_7'
eth_dist_labels = ['.', 'Non-Hispanic','Hispanic']

cm_eth_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*' + '__biased__raw__' + attr + '__Overall.npy')
cm_eth_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*' + '__gt__raw__' + attr + '__Overall.npy')
cm_eth = cm_eth_biased - cm_eth_sampled
diff_max = max([h for j in range(len(cm_eth)) for h in cm_eth[j]])
sb.heatmap(np.transpose(cm_eth), ax=ax[0], cmap="RdBu", center = 0, vmin = -1 * diff_max,  vmax = diff_max, cbar_kws={'label': 'Mean # unique contacts \n per day'})

ax[0].invert_yaxis()

ax[0].set(title='$\Delta$[Biased - True]', xlabel="Participant Ethnicity", ylabel="Contact Ethnicity")
ax[0].xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[0].xaxis.set_major_formatter(FixedFormatter(eth_dist_labels))
ax[0].yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[0].yaxis.set_major_formatter(FixedFormatter(eth_dist_labels))
ax[0].tick_params(which='major', pad=2, labelsize=7,labelrotation=45)
ax[0].set_yticklabels(labels=eth_dist_labels,va='center')

ax[0].text(
        -0.15, 1.0, 'A)', transform=(
            ax[0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)

#Panel B - Hispanic SIR
attr = 'ae'
input_params = 'tract_7'
target_groups = 'Hispanic'

if metric == 'ar':
    I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups + '_*' + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
    I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups + '_*' + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
elif metric == 'prev':
    I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups + '_*' + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
    I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups + '_*' + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
t = np.linspace(0, days, days)


ax[1].plot(t, I_biased[0:days], 'b', alpha=0.5, lw=1.5, label='Biased')
ax[1].plot(t, I_groundtruth[0:days], 'r', alpha=0.5, lw=1.5, label='True')
for i in range(10):
    if metric == 'ar':
        I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups + '_' + str(i) + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
        I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups + '_' + str(i) + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
    elif metric == 'prev':
        I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups + '_' + str(i) + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
        I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups + '_' + str(i) + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
    ax[1].plot(t, I_biased[0:days], 'b', alpha=0.2, lw=0.5)
    ax[1].plot(t, I_groundtruth[0:days], 'r', alpha=0.2, lw=0.5)
ax[1].set_xlabel('Time (days)')
if metric == 'ar':
    ax[1].set_ylabel('Cumulative \nIncidence')
elif metric == 'prev':
    ax[1].set_ylabel('Prevalence')
ax[1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
ax[1].yaxis.set_tick_params(length=0)
ax[1].xaxis.set_tick_params(length=0)
legend = ax[1].legend(fontsize=8)
for spine in ('top', 'right', 'bottom', 'left'):
    ax[1].spines[spine].set_visible(False)
ax[1].set(title='Hispanic pop.')
ax[1].grid()
ax[1].ticklabel_format(style='sci', axis='y')

ax[1].text(
        0.0, 1.0, 'B)', transform=(
            ax[1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)


# #Panel C - SIR hispanic difference
attr = 'ae'
input_params_all = ['tract_' + str(h) for h in range(9)]

r_1 = np.linspace(0, 2.05, 9)
ar_diff_hispanic = []
target_groups_input = 'Hispanic'
for input_params in input_params_all:
    if metric == 'ar':
        I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
        I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
    elif metric == 'prev':
        I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
        I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
    ar_diff_hispanic.append( max(I_biased) - max(I_groundtruth))


ar_diff_non_hispanic = []

# #Panel C - SIR non-hispanic difference
target_groups_input = 'non-Hispanic'
for input_params in input_params_all:
    if metric == 'ar':
        I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
        I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
    elif metric == 'prev':
        I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
        I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
    ar_diff_non_hispanic.append(max(I_biased) - max(I_groundtruth))


ax[2].plot(r_1, ar_diff_non_hispanic, alpha=0.5, lw=1.5, label='non-Hispanic')
ax[2].plot(r_1, ar_diff_hispanic, alpha=0.5, lw=1.5, label='Hispanic')
ax[2].set_xlabel('Bias magnitude ($e_{H}$)')
if metric == 'ar':
    ax[2].set_ylabel('$\Delta$[Attack Rate]')
elif metric == 'prev':
    ax[2].set_ylabel('$\Delta$[Peak Prevalence]')
ax[2].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
ax[2].yaxis.set_tick_params(length=0)
ax[2].xaxis.set_tick_params(length=0)
legend = ax[2].legend(fontsize=8)
for spine in ('top', 'right', 'bottom', 'left'):
    ax[2].spines[spine].set_visible(False)
ax[2].set(title='$\Delta$[Biased - True]')
ax[2].grid()

ax[2].text(
        0.0, 1.0, 'C)', transform=(
            ax[2].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)


plt.tight_layout()
plt.savefig('../Figures/Supplementary Material/ethnicity_bias_' + pathogen + '_' + metric + '.pdf')
