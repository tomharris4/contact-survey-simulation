from matplotlib.ticker import FixedFormatter, MultipleLocator
from matplotlib.transforms import ScaledTranslation
import networkx as nx
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt

experiment = 'exp2a'
pathogen = 'X'
if pathogen == 'I' or pathogen == 'X':
    days = 200
else:
    days = 100

input_network = 'NM_network_v3'
date = '2025-06-24'
attr = 'r'

fig, ax = plt.subplots(nrows=2, ncols=2)#, figsize=(8.3,9))

#Panel A - age
input_params = '0'
race_dist_labels = ['.', 'White', 'Black', 'Asian', 'AIAN',  'NHPI','Other','Multi']

cm_race_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__raw__' + attr + '__Overall.npy')
cm_race_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__raw__' + attr + '__Overall.npy')
cm_race = cm_race_biased - cm_race_sampled
diff_max = max([h for j in range(len(cm_race)) for h in cm_race[j]])
sb.heatmap(cm_race, ax=ax[0,0], cmap="RdBu", center = 0, vmin = -1 * diff_max,  vmax = diff_max, cbar_kws={'label': 'Mean # unique contacts \n per day'})

ax[0,0].invert_yaxis()

ax[0,0].set(title='$\Delta$[Biased - True]', xlabel="Participant Race", ylabel="Contact Race")
ax[0,0].xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[0,0].xaxis.set_major_formatter(FixedFormatter(race_dist_labels))
ax[0,0].yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[0,0].yaxis.set_major_formatter(FixedFormatter(race_dist_labels))
ax[0,0].tick_params(which='major', pad=2, labelsize=7,labelrotation=45)
ax[0,0].set_yticklabels(labels=race_dist_labels,va='center')

ax[0,0].text(
        -0.15, 1.0, 'A)', transform=(
            ax[0,0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)

#Panel C - ethnicity
# attr = 'e'
# input_params = ''

# cm_eth_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__raw__' + attr + '__Overall.npy')
# cm_eth_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__raw__' + attr + '__Overall.npy')
# cm_eth = cm_eth_biased - cm_eth_sampled
# diff_max = max([h for j in range(len(cm_eth)) for h in cm_eth[j]])
# sb.heatmap(cm_eth, ax=ax[1,0], cmap="RdBu", center = 0, vmin = -1 * diff_max,  vmax = diff_max, cbar_kws={'label': 'Mean # unique contacts \n per day'})

# ax[1,0].invert_yaxis()

# ax[1,0].set(title='$\Delta$[Biased - True]', xlabel="Participant Ethnicity", ylabel="Contact Ethnicity")
# eth_dist_labels = ['.', 'Non-Hispanic','Hispanic']
# ax[1,0].xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
# ax[1,0].xaxis.set_major_formatter(FixedFormatter(eth_dist_labels))
# ax[1,0].yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
# ax[1,0].yaxis.set_major_formatter(FixedFormatter(eth_dist_labels))
# ax[1,0].tick_params(which='major', pad=2, labelsize=7,labelrotation=45)
# ax[1,0].set_yticklabels(labels=eth_dist_labels,va='center')

# ax[1,0].text(
#         -0.15, 1.0, 'C)', transform=(
#             ax[1,0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
#          va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)

attr = 'ar'
# #Panel B - l=0 - SIR Elderly
I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
t = np.linspace(0, days, days)



ax[0,1].plot(t, I_biased[0:days], 'b', alpha=0.5, lw=1.5, label='Biased')
ax[0,1].plot(t, I_groundtruth[0:days], 'r', alpha=0.5, lw=1.5, label='True')
for i in range(10):
    I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
    I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
    ax[0,1].plot(t, I_biased[0:days], 'b', alpha=0.2, lw=0.5)
    ax[0,1].plot(t, I_groundtruth[0:days], 'r', alpha=0.2, lw=0.5)
ax[0,1].set_xlabel('Time (days)')
# ax[0,1].set_ylabel('Prevalence')
ax[0,1].set_ylabel('Cumulative Incidence')
# ax.set_ylim(0,1.2)
ax[0,1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
ax[0,1].yaxis.set_tick_params(length=0)
ax[0,1].xaxis.set_tick_params(length=0)
legend = ax[0,1].legend(fontsize=8)
# legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax[0,1].spines[spine].set_visible(False)
ax[0,1].set(title='Non-White')
ax[0,1].grid()
# ax[0,1].ticklabel_format(style='sci', axis='y')

ax[0,1].text(
        0.0, 1.0, 'B)', transform=(
            ax[0,1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)

# attr = 'ae'
# # #Panel B - l=0 - SIR Hispanic
# I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall.npy')
# I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall.npy')
# t = np.linspace(0, days, days)

# ax[1,1].plot(t, I_biased[0:days], 'b', alpha=0.5, lw=1.5, label='Biased')
# ax[1,1].plot(t, I_groundtruth[0:days], 'r', alpha=0.5, lw=1.5, label='True')
# for i in range(10):
#     I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall.npy')
#     I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall.npy')
#     ax[1,1].plot(t, I_biased[0:days], 'b', alpha=0.2, lw=0.5)
#     ax[1,1].plot(t, I_groundtruth[0:days], 'r', alpha=0.2, lw=0.5)
# ax[1,1].set_xlabel('Time (days)')
# ax[1,1].set_ylabel('Prevalence')
# # ax.set_ylim(0,1.2)
# ax[1,1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
# ax[1,1].yaxis.set_tick_params(length=0)
# ax[1,1].xaxis.set_tick_params(length=0)
# # legend = ax[0,1].legend()
# # legend.get_frame().set_alpha(0.5)
# for spine in ('top', 'right', 'bottom', 'left'):
#     ax[1,1].spines[spine].set_visible(False)
# ax[1,1].set(title='Hispanic')
# ax[1,1].grid()


# ax[1,1].text(
#         0.0, 1.0, 'D)', transform=(
#             ax[1,1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
#          va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)

plt.tight_layout()
plt.savefig('../Figures/Supplementary Material/supp_dominant_' + pathogen + '.pdf')
