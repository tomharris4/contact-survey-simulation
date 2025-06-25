from matplotlib.ticker import FixedFormatter, MultipleLocator
from matplotlib.transforms import ScaledTranslation
import networkx as nx
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt

experiment = 'exp3'
pathogen = 'X'
metric =  'prev' # 'ar' #

if pathogen == 'I':
    days = 200
elif pathogen == 'X':
    days = 250
else:
    days = 100

input_network = 'NM_network_v3'
date = '2025-06-25'
attr = 'r'

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,2.5))

#Panel A - age
input_params = 'random'
race_dist_labels = ['.', 'White', 'Black', 'Asian', 'AIAN',  'NHPI','Other','Multi']

cm_race_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__raw__' + attr + '__Overall.npy')
cm_race_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__raw__' + attr + '__Overall.npy')
cm_race = cm_race_sampled - cm_race_biased
diff_max = max([h for j in range(len(cm_race)) for h in cm_race[j]])
sb.heatmap(cm_race, ax=ax[0], cmap="RdBu", center = 0, vmin = -1 * diff_max,  vmax = diff_max, cbar_kws={'label': 'Mean # unique contacts \n per day'})

ax[0].invert_yaxis()

ax[0].set(title='$\Delta$[True - Biased]', xlabel="Participant Race", ylabel="Contact Race")
ax[0].xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[0].xaxis.set_major_formatter(FixedFormatter(race_dist_labels))
ax[0].yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[0].yaxis.set_major_formatter(FixedFormatter(race_dist_labels))
ax[0].tick_params(which='major', pad=2, labelsize=7,labelrotation=45)
ax[0].set_yticklabels(labels=race_dist_labels,va='center')

ax[0].text(
        -0.15, 1.0, 'A)', transform=(
            ax[0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)

#Panel C - ethnicity
# attr = 'e'
# input_params = '8'

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
input_params = 'random'
# #Panel B - l=0 - SIR Elderly
if metric == 'ar':
    I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
    I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
elif metric == 'prev':
    I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
    I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
t = np.linspace(0, days, days)


ax[1].plot(t, I_biased[0:days], 'r', alpha=0.5, lw=1.5, label='Biased')
ax[1].plot(t, I_groundtruth[0:days], 'b', alpha=0.5, lw=1.5, label='True')
for i in range(10):
    if metric == 'ar':
        I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
        I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
    elif metric == 'prev':
        I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
        I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
    ax[1].plot(t, I_biased[0:days], 'r', alpha=0.2, lw=0.5)
    ax[1].plot(t, I_groundtruth[0:days], 'b', alpha=0.2, lw=0.5)
ax[1].set_xlabel('Time (days)')
if metric == 'ar':
    ax[1].set_ylabel('Cumulative Incidence')
elif metric == 'prev':
    ax[1].set_ylabel('Prevalence')
# ax.set_ylim(0,1.2)
ax[1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
ax[1].yaxis.set_tick_params(length=0)
ax[1].xaxis.set_tick_params(length=0)
legend = ax[1].legend(fontsize=8)
# legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax[1].spines[spine].set_visible(False)
ax[1].set(title='Non-White')
ax[1].grid()
ax[1].ticklabel_format(style='sci', axis='y')

ax[1].text(
        0.0, 1.0, 'B)', transform=(
            ax[1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)


attr = 'ar'
input_params_all = ['tract','state','dominant','random']

r_1 = ['Tract','State*','Majority','Uniform*']
ar_diff = []

# #Panel C - SIR Elderly
for input_params in input_params_all:
    if metric == 'ar':
        I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
        I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
    elif metric == 'prev':
        I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
        I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
    ar_diff.append(max(I_groundtruth) - max(I_biased))


sb.barplot(x=r_1,y=ar_diff,ax=ax[2], label = 'True - Biased')
# for i in range(10):
#     I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall.npy')
#     I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall.npy')
#     ax[0,1].plot(t, I_biased[0:days], 'b', alpha=0.2, lw=0.5)
#     ax[0,1].plot(t, I_groundtruth[0:days], 'r', alpha=0.2, lw=0.5)
ax[2].set_xlabel('Biased selection method')
if metric == 'ar':
    ax[2].set_ylabel('$\Delta$[Attack Rate]')
elif metric == 'prev':
    ax[2].set_ylabel('$\Delta$[Peak Prevalence]')
# ax.set_ylim(0,1.2)
ax[2].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
ax[2].yaxis.set_tick_params(length=0)
ax[2].xaxis.set_tick_params(length=0)
legend = ax[2].legend(fontsize=8)
# legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax[2].spines[spine].set_visible(False)
ax[2].set(title='Non-White')
ax[2].grid()
# ax[0,1].ticklabel_format(style='sci', axis='y')

ax[2].text(
        0.0, 1.0, 'C)', transform=(
            ax[2].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)

# attr = 'ae'
# # #Panel B - l=0 - SIR Hispanic
# input_params_all = [str(h) for h in range(9)]

# r_3 = np.linspace(0,0.8,9)
# ar_diff = []

# #Panel B - l=0 - SIR Elderly
# for input_params in input_params_all:
#     if metric == 'ar':
#         I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
#         I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
#     elif metric == 'prev':
#         I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
#         I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
#     ar_diff.append(max(I_groundtruth) - max(I_biased))



# ax[1,1].plot(r_3, ar_diff, 'b', alpha=0.5, lw=1.5, label='True - Biased')
# # for i in range(10):
# #     I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall.npy')
# #     I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall.npy')
# #     ax[0,1].plot(t, I_biased[0:days], 'b', alpha=0.2, lw=0.5)
# #     ax[0,1].plot(t, I_groundtruth[0:days], 'r', alpha=0.2, lw=0.5)
# ax[1,1].set_xlabel('$r_{1}$')
# if metric == 'ar':
#     ax[1,1].set_ylabel('$\Delta$[Attack rate]')
# elif metric == 'prev':
#     ax[1,1].set_ylabel('$\Delta$[Peak prevalence]')
# # ax.set_ylim(0,1.2)
# # ax[1,1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
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
plt.savefig('../Figures/supp_geo_selection_race_' + pathogen + '_' + metric + '.pdf')
