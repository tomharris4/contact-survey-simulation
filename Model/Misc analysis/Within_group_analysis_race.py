from matplotlib.ticker import FixedFormatter, MultipleLocator
from matplotlib.transforms import ScaledTranslation
import networkx as nx
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt

experiment = 'exp2b'
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

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,2.5))

#Panel A - age
input_params = '0'
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


# Panel B - l=0 - SIR Non-White
attr = 'ar'
input_params = '0'
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
input_params_all = [str(h) for h in range(3)]

r_1 = np.linspace(0, 1, 3)
ar_diff = []

# #Panel C - SIR Non-White difference
for input_params in input_params_all:
    if metric == 'ar':
        I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
        I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
    elif metric == 'prev':
        I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
        I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
    ar_diff.append(max(I_groundtruth) - max(I_biased))


ax[2].plot(r_1, ar_diff, 'g', alpha=0.5, lw=1.5, label='True - Biased')
ax[2].scatter([r_1[0]],[ar_diff[0]],color='g',label='$l=0$')
# for i in range(10):
#     I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall.npy')
#     I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall.npy')
#     ax[0,1].plot(t, I_biased[0:days], 'b', alpha=0.2, lw=0.5)
#     ax[0,1].plot(t, I_groundtruth[0:days], 'r', alpha=0.2, lw=0.5)
ax[2].set_xlabel('$l$')
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


plt.tight_layout()
plt.savefig('../Figures/supp_within_group_race_' + pathogen + '_' + metric + '.pdf')
