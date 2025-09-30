# Script for generating Figure 3 plot

from matplotlib.ticker import FixedFormatter, FuncFormatter, MultipleLocator
from matplotlib.transforms import ScaledTranslation
import networkx as nx
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

experiment = 'exp1'
pathogen = 'C_2_9'
metric =  'ar'

days = 100

input_network = 'NM_network_v3'
date = '2025-06-30'
attr = 'a'
N_pop = 2089388

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9,7))


#Panel A - age contact matrices
input_params = '4'

cm_age_biased = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__raw__' + attr + '__Overall.npy')
cm_age_sampled = np.load('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__raw__' + attr + '__Overall.npy')

cm_age = cm_age_biased - cm_age_sampled

diff_max = max([h for j in range(len(cm_age)) for h in cm_age[j]])
sb.heatmap(np.transpose(cm_age), ax=ax[0][0], cmap="RdBu", center = 0, vmin = -1 * diff_max,  vmax = diff_max, cbar_kws={'label': '$\Delta$Mean # unique contacts\nper day'})

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

ax[0][0].set(title='$\Delta$[Biased - True]', xlabel="Participant Age", ylabel="Contact Age")
ax[0][0].xaxis.set_major_locator(MultipleLocator(2))
ax[0][0].xaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,10))))
ax[0][0].yaxis.set_major_locator(MultipleLocator(2))
ax[0][0].yaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,10))))
ax[0][0].tick_params(which='major', pad=2, labelsize=7)

ax[0][0].text(
        0, 1.0, 'A)', transform=(
            ax[0][0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)


#Panel B - SIR Elderly
attr = 'a'
input_params = '4'
target_group = 'elderly'
if metric == 'ar':
    I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
    I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
elif metric == 'prev':
    I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
    I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
t = np.linspace(0, days, days)


ax[0][1].plot(t, I_biased[0:days], 'b', alpha=0.5, lw=1.5, label='Biased')
ax[0][1].plot(t, I_groundtruth[0:days], 'r', alpha=0.5, lw=1.5, label='True')
for i in range(10):
    if metric == 'ar':
        I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_' + str(i) + '_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
        I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_' + str(i) + '_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
    elif metric == 'prev':
        I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_' + str(i) + '_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
        I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_' + str(i) + '_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
    ax[0][1].plot(t, I_biased[0:days], 'b', alpha=0.2, lw=0.5)
    ax[0][1].plot(t, I_groundtruth[0:days], 'r', alpha=0.2, lw=0.5)
ax[0][1].set_xlabel('Time (days)')
if metric == 'ar':
    ax[0][1].set_ylabel('Cumulative Incidence')
elif metric == 'prev':
    ax[0][1].set_ylabel('Prevalence')
ax[0][1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
ax[0][1].yaxis.set_tick_params(length=0)
ax[0][1].xaxis.set_tick_params(length=0)
legend = ax[0][1].legend(fontsize=8)
for spine in ('top', 'right', 'bottom', 'left'):
    ax[0][1].spines[spine].set_visible(False)
ax[0][1].set(title='Older people (65+ years)')
ax[0][1].grid()
ax[0][1].ticklabel_format(style='sci', axis='y')

ax[0][1].text(
        0.0, 1.0, 'B)', transform=(
            ax[0][1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)


# Panel C - SIR dynamics
attr = 'a'
input_params_all = [str(h) for h in range(7)]
target_group_all = ['elderly','children','adults']

r_1 = np.linspace(0, 3.84, 7)

ar_diff_overall = []

# Boolean indicating whether to use average matrices or process individual trajectories
average = True

if average:
    for input_params in input_params_all:
        ar_diff_overall.append(0)
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
            ax[1][0].plot(r_1, ar_diff, alpha=0.5, lw=1.5, label='Older')
        else:
            ax[1][0].plot(r_1, ar_diff, alpha=0.5, lw=1.5, label=target_group.title())

    ax[1][0].plot(r_1, ar_diff_overall, alpha=0.5, lw=1.5, label='Overall')
    ax[1][0].set_xlabel('Bias magnitude ($b_{age}$)')
else:
    ar_diff_children = []
    ar_diff_adults = []
    ar_diff_older = []

    for input_params in input_params_all:
        ar_diff_overall.append([])
        ar_diff_children.append([])
        ar_diff_adults.append([])
        ar_diff_older.append([])
    for k in range(10):
        for input_params in range(len(input_params_all)):
            for target_group in target_group_all:
                if metric == 'ar':
                    I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params_all[input_params] + '__' + target_group + '_' + str(k) + '_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
                    I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params_all[input_params] + '__' + target_group + '_' + str(k) + '_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy')
                elif metric == 'prev':
                    I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params_all[input_params] + '__' + target_group + '_' + str(k) + '_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
                    I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params_all[input_params] + '__' + target_group + '_' + str(k) + '_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy')
                ar_diff = (max(I_biased) - max(I_groundtruth))
                if target_group == 'elderly':
                    ar_diff_older[input_params].append((max(I_biased) - max(I_groundtruth)))
                elif target_group == 'children':
                    ar_diff_children[input_params].append(ar_diff)
                else:
                    ar_diff_adults[input_params].append(ar_diff)
            ar_diff_overall[input_params].append(ar_diff_adults[input_params][-1] + ar_diff_children[input_params][-1] + ar_diff_older[input_params][-1])

    # 95% CIs
    # ar_diff_older_err = [stats.t.interval(0.95, df=len(d)-1, loc=np.mean(d), scale=np.std(d, ddof=1) / np.sqrt(len(d)))[1]
    #                      - stats.t.interval(0.95, df=len(d)-1, loc=np.mean(d), scale=np.std(d, ddof=1) / np.sqrt(len(d)))[0] for d in ar_diff_older]

    # Std Dev
    ar_diff_older_err = [np.std(h) for h in ar_diff_older]
    ar_diff_adults_err = [np.std(h) for h in ar_diff_adults]
    ar_diff_children_err = [np.std(h) for h in ar_diff_children]
    ar_diff_overall_err = [np.std(h) for h in ar_diff_overall]

    ax[1][0].errorbar(r_1, [np.mean(h) for h in ar_diff_children], yerr= ar_diff_children_err, alpha=0.5, lw=1.5, label='Children')
    ax[1][0].errorbar(r_1, [np.mean(h) for h in ar_diff_adults], yerr= ar_diff_adults_err, alpha=0.5, lw=1.5, label='Adults')
    ax[1][0].errorbar(r_1, [np.mean(h) for h in ar_diff_older], yerr= ar_diff_older_err, alpha=0.5, lw=1.5, label='Older')
    ax[1][0].errorbar(r_1, [np.mean(h) for h in ar_diff_overall], yerr= ar_diff_overall_err, alpha=0.5, lw=1.5, label='Overall')
    ax[1][0].set_xlabel('Bias magnitude ($b_{age}$)')

if metric == 'ar':
    ax[1][0].set_ylabel('$\Delta$[Final Size]')
elif metric == 'prev': 
    ax[1][0].set_ylabel('$\Delta$[Peak Prevalence]')
ax[1][0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
ax[1][0].yaxis.set_tick_params(length=0)
ax[1][0].xaxis.set_tick_params(length=0)
legend = ax[1][0].legend(fontsize=8, loc='lower left')
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
r_r0 = []
target_group_all = ['elderly']

r = np.linspace(0, 3.84, 7)
r0 = [1.2,1.6,2,2.4,2.8,3.2,3.6,4,4.4,4.8,5.2,5.6,6]
i = -1

ar_diff = []

for r_temp in r0:
    i += 1
    ar_diff.append([])
    for target_group in target_group_all:
        for input_params in input_params_all:
            if metric == 'ar':
                I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '_' + str(r_temp).replace('.','_') + '__Overall_Recovered.npy')
                I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '_' + str(r_temp).replace('.','_') + '__Overall_Recovered.npy')
            elif metric == 'prev':
                I_biased = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '_' + str(r_temp).replace('.','_') + '__Overall_Infectious.npy')
                I_groundtruth = np.load('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_group + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '_' + str(r_temp).replace('.','_') + '__Overall_Infectious.npy')
            ar_diff[i].append(max(I_biased)/max(I_groundtruth))


if metric == 'ar':
    sb.heatmap(ar_diff, ax=ax[1][1], cbar_kws={'label': 'Biased AR / True AR'})
elif metric == 'prev':
    sb.heatmap(ar_diff, ax=ax[1][1], cbar_kws={'label': 'Biased PP / True PP'})
ax[1][1].invert_yaxis()

ax[1][1].set_xticklabels(r)
ax[1][1].set_yticklabels(r0)

ax[1][1].set_xlabel('Bias magnitude ($b_{age}$)')
ax[1][1].set_ylabel('$R_0$')
ax[1][1].set(title='Older people (65+ years)')

ax[1][1].tick_params(which='major', pad=2, labelsize=8,labelrotation=45)

ax[1][1].text(
        0, 1.0, 'D)', transform=(
            ax[1][1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
         va='bottom', fontfamily='sans-serif', fontweight='bold', size=12)

plt.tight_layout(w_pad=0.01)
plt.savefig('../Figures/figure_3_' + pathogen + '_' + metric + '_final.pdf')
