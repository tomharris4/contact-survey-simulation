from matplotlib.ticker import FixedFormatter, MultipleLocator
from matplotlib.transforms import ScaledTranslation
import networkx as nx
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt

pathogen = 'C'
if pathogen == 'I':
    days = 200
else:
    days = 100

input_network = 'NM_network_v3'
date = '2025-06-06'
attr = 's'
experiment = 'exp3'


fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8.3,9))

#Panel A - tract
input_params = 'tract'
ses_dist_labels = ['.', 'Lower', 'Middle', 'Upper']

cm_ses_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__raw__' + attr + '__Overall.npy')
cm_ses_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__raw__' + attr + '__Overall.npy')
cm_ses = cm_ses_biased - cm_ses_sampled
diff_max = max([h for j in range(len(cm_ses)) for h in cm_ses[j]])
sb.heatmap(cm_ses, ax=ax[0,0], cmap="RdBu", center = 0, vmin = -1 * diff_max,  vmax = diff_max, cbar_kws={'label': 'Mean # unique contacts \n per day'})

ax[0,0].invert_yaxis()

ax[0,0].set(title='$\Delta$[Biased - True], (Tract)', xlabel="Participant Social Class", ylabel="Contact Social Class")
ax[0,0].xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[0,0].xaxis.set_major_formatter(FixedFormatter(ses_dist_labels))
ax[0,0].yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[0,0].yaxis.set_major_formatter(FixedFormatter(ses_dist_labels))
ax[0,0].tick_params(which='major', pad=2, labelsize=7,labelrotation=45)
ax[0,0].set_yticklabels(labels=ses_dist_labels,va='center')

#Panel C - state
input_params = 'state'
ses_dist_labels = ['.', 'Lower', 'Middle', 'Upper']

cm_ses_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__raw__' + attr + '__Overall.npy')
cm_ses_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__raw__' + attr + '__Overall.npy')
cm_ses = cm_ses_biased - cm_ses_sampled
diff_max = max([h for j in range(len(cm_ses)) for h in cm_ses[j]])
sb.heatmap(cm_ses, ax=ax[1,0], cmap="RdBu", center = 0, vmin = -1 * diff_max,  vmax = diff_max, cbar_kws={'label': 'Mean # unique contacts \n per day'})

ax[1,0].invert_yaxis()

ax[1,0].set(title='$\Delta$[Biased - True], (State)', xlabel="Participant Social Class", ylabel="Contact Social Class")
ax[1,0].xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[1,0].xaxis.set_major_formatter(FixedFormatter(ses_dist_labels))
ax[1,0].yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[1,0].yaxis.set_major_formatter(FixedFormatter(ses_dist_labels))
ax[1,0].tick_params(which='major', pad=2, labelsize=7,labelrotation=45)
ax[1,0].set_yticklabels(labels=ses_dist_labels,va='center')

#Panel E - random
input_params = 'random'
ses_dist_labels = ['.', 'Lower', 'Middle', 'Upper']

cm_ses_biased = np.load('../Data/Contact matrices/' + input_network +  '__' + experiment + '__' + input_params + '_*_' + date + '__biased__raw__' + attr + '__Overall.npy')
cm_ses_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__raw__' + attr + '__Overall.npy')
cm_ses = cm_ses_biased - cm_ses_sampled
diff_max = max([h for j in range(len(cm_ses)) for h in cm_ses[j]])
sb.heatmap(cm_ses, ax=ax[2,0], cmap="RdBu", center = 0, vmin = -1 * diff_max,  vmax = diff_max, cbar_kws={'label': 'Mean # unique contacts \n per day'})

ax[2,0].invert_yaxis()

ax[2,0].set(title='$\Delta$[Biased - True], (Random)', xlabel="Participant Social Class", ylabel="Contact Social Class")
ax[2,0].xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[2,0].xaxis.set_major_formatter(FixedFormatter(ses_dist_labels))
ax[2,0].yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[2,0].yaxis.set_major_formatter(FixedFormatter(ses_dist_labels))
ax[2,0].tick_params(which='major', pad=2, labelsize=7,labelrotation=45)
ax[2,0].set_yticklabels(labels=ses_dist_labels,va='center')

attr = 'as'

#Panel B - tract - SIR lower
input_params = 'tract'
I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall.npy')
I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall.npy')
t = np.linspace(0, days, days)

ax[0,1].plot(t, I_biased[0:days], 'b', alpha=0.5, lw=1.5, label='Biased')
ax[0,1].plot(t, I_groundtruth[0:days], 'r', alpha=0.5, lw=1.5, label='True')
for i in range(10):
    I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall.npy')
    I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall.npy')
    ax[0,1].plot(t, I_biased[0:days], 'b', alpha=0.2, lw=0.5)
    ax[0,1].plot(t, I_groundtruth[0:days], 'r', alpha=0.2, lw=0.5)
ax[0,1].set_xlabel('Time (days)')
ax[0,1].set_ylabel('Prevalence')
ax[0,1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
# ax.set_ylim(0,1.2)
ax[0,1].yaxis.set_tick_params(length=0)
ax[0,1].xaxis.set_tick_params(length=0)
legend = ax[0,1].legend()
# legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax[0,1].spines[spine].set_visible(False)
ax[0,1].set(title='Lower class')
ax[0,1].grid()


#Panel D - state - SIR lower
input_params = 'state'
I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall.npy')
I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall.npy')
t = np.linspace(0, days, days)

ax[1,1].plot(t, I_biased[0:days], 'b', alpha=0.5, lw=1.5, label='Biased')
ax[1,1].plot(t, I_groundtruth[0:days], 'r', alpha=0.5, lw=1.5, label='True')
for i in range(10):
    I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall.npy')
    I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall.npy')
    ax[1,1].plot(t, I_biased[0:days], 'b', alpha=0.2, lw=0.5)
    ax[1,1].plot(t, I_groundtruth[0:days], 'r', alpha=0.2, lw=0.5)
ax[1,1].set_xlabel('Time (days)')
ax[1,1].set_ylabel('Prevalence')
ax[1,1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
# ax.set_ylim(0,1.2)
ax[1,1].yaxis.set_tick_params(length=0)
ax[1,1].xaxis.set_tick_params(length=0)
# legend = ax[0,1].legend()
# legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax[1,1].spines[spine].set_visible(False)
ax[1,1].set(title='Lower class')
ax[1,1].grid()


#Panel F - random - SIR lower
input_params = 'random'
I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall.npy')
I_groundtruth = np.load('../Data/SIR trajectories/' + input_network +  '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall.npy')
t = np.linspace(0, days, days)

ax[2,1].plot(t, I_biased[0:days], 'b', alpha=0.5, lw=1.5, label='Biased')
ax[2,1].plot(t, I_groundtruth[0:days], 'r', alpha=0.5, lw=1.5, label='True')
for i in range(10):
    I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall.npy')
    I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall.npy')
    ax[2,1].plot(t, I_biased[0:days], 'b', alpha=0.2, lw=0.5)
    ax[2,1].plot(t, I_groundtruth[0:days], 'r', alpha=0.2, lw=0.5)
ax[2,1].set_xlabel('Time (days)')
ax[2,1].set_ylabel('Prevalence')
ax[2,1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
# ax.set_ylim(0,1.2)
ax[2,1].yaxis.set_tick_params(length=0)
ax[2,1].xaxis.set_tick_params(length=0)
# legend = ax[2,1].legend()
# legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax[2,1].spines[spine].set_visible(False)
ax[2,1].set(title='Lower class')
ax[2,1].grid()

plt.tight_layout()
plt.savefig('../Figures/figure5_' + pathogen + '.pdf')
