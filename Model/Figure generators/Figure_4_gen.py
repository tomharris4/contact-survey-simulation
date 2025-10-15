# Script for generating Figure 4 plot

from matplotlib.ticker import FixedFormatter, FuncFormatter, MultipleLocator
from matplotlib.transforms import ScaledTranslation
import networkx as nx
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt

experiment = 'exp2'
pathogen = 'C_2_9'
metric =  'ar'

if pathogen in ['C_2_9','X_2_9']:  
    days = 100
elif pathogen in ['C_1_4','X_1_4']:
    days = 250

input_network = 'NM_network'
attr = 'r'
N_pop = 2089388

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9,7))

#Panel A - race contact matrices
input_params = 'tract_7'
race_dist_labels = ['.', 'White', 'Black', 'Asian', 'AIAN',  'NHPI','Other','Multi']
pop_dist = {0: 1568835,   1: 42843,   2: 32668,  3: 199247,    4: 1454,  5: 178453,   6: 65888}

cm_race_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*' + '__biased__raw__' + attr + '__Overall.npy')
cm_race_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*' + '__gt__raw__' + attr + '__Overall.npy')

cm_race = cm_race_biased - cm_race_sampled

diff_max = max([abs(h) for j in range(len(cm_race)) for h in cm_race[j]])
sb.heatmap(np.transpose(cm_race), ax=ax[0][0], cmap="RdBu", center = 0, vmin = -1 * diff_max,  vmax = diff_max, cbar_kws={'label': '$\Delta$Mean # unique contacts per day\n'})

# Get the colorbar object
cbar = ax[0][0].collections[0].colorbar

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

ax[0][0].invert_yaxis()

ax[0][0].set(title='$\Delta$[Biased - True]', xlabel="Participant Race", ylabel="Contact Race")
ax[0][0].xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[0][0].xaxis.set_major_formatter(FixedFormatter(race_dist_labels))
ax[0][0].yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
ax[0][0].yaxis.set_major_formatter(FixedFormatter(race_dist_labels))
ax[0][0].tick_params(which='major', pad=2, labelsize=7,labelrotation=45)
ax[0][0].set_yticklabels(labels=race_dist_labels,va='center')

ax[0][0].text(
        0.0, 1.0, 'A)', transform=(
            ax[0][0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)

#Panel B - non-white SIR
attr = 'ar'
input_params = 'tract_7'
target_groups = 'Non-White'

if metric == 'ar':
    I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups + '_*' + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
    I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups + '_*' + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
elif metric == 'prev':
    I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups + '_*' + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
    I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups + '_*' + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
t = np.linspace(0, days, days)


ax[0][1].plot(t, I_biased[0:days], 'b', alpha=0.5, lw=1.5, label='Biased')
ax[0][1].plot(t, I_groundtruth[0:days], 'r', alpha=0.5, lw=1.5, label='True')
for i in range(10):
    if metric == 'ar':
        I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups + '_' + str(i) + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
        I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups + '_' + str(i) + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
    elif metric == 'prev':
        I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups + '_' + str(i) + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
        I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups + '_' + str(i) + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
    ax[0][1].plot(t, I_biased[0:days], 'b', alpha=0.2, lw=0.5)
    ax[0][1].plot(t, I_groundtruth[0:days], 'r', alpha=0.2, lw=0.5)
ax[0][1].set_xlabel('Time (days)')
if metric == 'ar':
    ax[0][1].set_ylabel('Cumulative Incidence')
    ax[0][1].set(title='Disease incidence among \nnon-White people')
elif metric == 'prev':
    ax[0][1].set_ylabel('Prevalence')
    ax[0][1].set(title='Disease prevalence among \nnon-White people')
ax[0][1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
ax[0][1].yaxis.set_tick_params(length=0)
ax[0][1].xaxis.set_tick_params(length=0)
legend = ax[0][1].legend(fontsize=8)
for spine in ('top', 'right', 'bottom', 'left'):
    ax[0][1].spines[spine].set_visible(False)
ax[0][1].grid()
ax[0][1].ticklabel_format(style='sci', axis='y')

ax[0][1].text(
        0.0, 1.0, 'B)', transform=(
            ax[0][1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)


# #Panel C - SIR dynamics
attr = 'ar'
input_params_all = ['tract_' + str(h) for h in range(9)]
target_group_all = ['White','Non-White']
r_1 = np.linspace(0, 2.05, 9)

average = True

if average:
    
    ar_diff_white = []
    target_groups_input = 'White'
    for input_params in input_params_all:
        if metric == 'ar':
            I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
            I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
        elif metric == 'prev':
            I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
            I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
        ar_diff_white.append( max(I_biased) - max(I_groundtruth))


    ar_diff_non_white = []

    target_groups_input = 'non-White'
    for input_params in input_params_all:
        if metric == 'ar':
            I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
            I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
        elif metric == 'prev':
            I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
            I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
        ar_diff_non_white.append(max(I_biased) - max(I_groundtruth))

    ar_diff_overall = [ar_diff_non_white[i] + ar_diff_white[i] for i in range(len(ar_diff_white))]

    ax[1][0].plot(r_1, ar_diff_non_white, alpha=0.5, lw=1.5, label='non-White')
    ax[1][0].plot(r_1, ar_diff_white, alpha=0.5, lw=1.5, label='White')
    ax[1][0].plot(r_1, ar_diff_overall, alpha=0.5, lw=1.5, label='Overall')
else:
    ar_diff_overall = []
    ar_diff_white = []
    ar_diff_non_white = []

    for input_params in input_params_all:
        ar_diff_overall.append([])
        ar_diff_white.append([])
        ar_diff_non_white.append([])

    for k in range(10):
        for input_params in range(len(input_params_all)):
            for target_group in target_group_all:
                if metric == 'ar':
                    I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params_all[input_params] + '__' + target_group + '_' + str(k) + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
                    I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params_all[input_params] + '__' + target_group + '_' + str(k) + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
                elif metric == 'prev':
                    I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params_all[input_params] + '__' + target_group + '_' + str(k) + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
                    I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params_all[input_params] + '__' + target_group + '_' + str(k) + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
                ar_diff = (max(I_biased) - max(I_groundtruth))
                if target_group == 'White':
                    ar_diff_white[input_params].append((max(I_biased) - max(I_groundtruth)))
                elif target_group == 'Non-White':
                    ar_diff_non_white[input_params].append(ar_diff)

            ar_diff_overall[input_params].append(ar_diff_white[input_params][-1] + ar_diff_non_white[input_params][-1])
            

    ar_diff_white_err = [np.std(h) for h in ar_diff_white]
    ar_diff_non_white_err = [np.std(h) for h in ar_diff_non_white]
    ar_diff_overall_err = [np.std(h) for h in ar_diff_overall]
    ax[1][0].errorbar(r_1, [np.mean(h) for h in ar_diff_white], yerr= ar_diff_white_err, alpha=0.5, lw=1.5, label='White')
    ax[1][0].errorbar(r_1, [np.mean(h) for h in ar_diff_non_white], yerr= ar_diff_non_white_err, alpha=0.5, lw=1.5, label='non-White')
    ax[1][0].errorbar(r_1, [np.mean(h) for h in ar_diff_overall], yerr= ar_diff_overall_err, alpha=0.5, lw=1.5, label='Overall')


ax[1][0].set_xlabel('Bias magnitude ($r_{NW}$)')
if metric == 'ar':
    ax[1][0].set_ylabel('$\Delta$[Final size]')
elif metric == 'prev':
    ax[1][0].set_ylabel('$\Delta$[Peak Prevalence]')
ax[1][0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
ax[1][0].yaxis.set_tick_params(length=0)
ax[1][0].xaxis.set_tick_params(length=0)
legend = ax[1][0].legend(fontsize=8)
for spine in ('top', 'right', 'bottom', 'left'):
    ax[1][0].spines[spine].set_visible(False)
ax[1][0].set(title='$\Delta$[Biased - True]')
ax[1][0].grid()

ax[1][0].text(
        0.0, 1.0, 'C)', transform=(
            ax[1][0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)


# Panel D
pathogen = pathogen[0]
target_group_all = ['Non-White']

r = [round(h,2) for h in np.linspace(0, 2.05, 9)]
r0 = [1.2,1.6,2,2.4,2.8,3.2,3.6,4,4.4,4.8,5.2,5.6,6]
i = -1
ar_diff = []

for r_temp in r0:
    i += 1
    ar_diff.append([])
    for target_group in target_group_all:
        for input_params in input_params_all:
            if metric == 'ar':
                I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*' + '__biased__processed__' + attr + '__' + pathogen + '_' + str(r_temp).replace('.','_') + '__Overall_Recovered.npy')
                I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*' + '__gt__processed__' + attr + '__' + pathogen + '_' + str(r_temp).replace('.','_') + '__Overall_Recovered.npy')
            elif metric == 'prev':
                I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*' + '__biased__processed__' + attr + '__' + pathogen + '_' + str(r_temp).replace('.','_') + '__Overall_Infectious.npy')
                I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*' + '__gt__processed__' + attr + '__' + pathogen + '_' + str(r_temp).replace('.','_') + '__Overall_Infectious.npy')
            ar_diff[i].append(max(I_biased)/max(I_groundtruth))


if metric == 'ar':
    sb.heatmap(ar_diff, ax=ax[1][1], cbar_kws={'label': 'Biased AR / True AR'})
elif metric == 'prev':
    sb.heatmap(ar_diff, ax=ax[1][1], cbar_kws={'label': 'Biased PP / True PP'})
ax[1][1].invert_yaxis()

ax[1][1].set_xticklabels(r)
ax[1][1].set_yticklabels(r0)

ax[1][1].set_xlabel('Bias magnitude ($r_{NW}$)')
ax[1][1].set_ylabel('$R_0$')
ax[1][1].set(title='Attack Rate (AR) among \nnon-White people')

ax[1][1].tick_params(which='major', pad=2, labelsize=8,labelrotation=45)

ax[1][1].text(
        0.0, 1.0, 'D)', transform=(
            ax[1][1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)

plt.tight_layout()
plt.savefig('../Figures/figure_4_' + pathogen + '_' + metric + '_final.pdf')
