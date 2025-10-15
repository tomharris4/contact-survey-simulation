# Script for comparing racial bias results to experimental estimates

import numpy as np
import pickle
import seaborn as sb
import matplotlib.pyplot as plt

experiment = 'exp2__survey_tract_'
input_network = 'NM_network'
attr = 'r'

white_bias = []
n_white_bias = []
error_rate = []
target_groups = [1,2,3,4,5,6] # non-White

input_params = range(9)
survey_reps = range(10)
race_dist_labels = ['.', 'White', 'Black', 'Asian', 'AIAN',  'NHPI','Other','Multi']

G = pickle.load(open('../Data/Contact network/' + input_network + '.pickle', 'rb'))


for i in input_params:
    correct = 0
    total = 0
    white_white_prediction = 0
    white_contacts = 0
    n_white_white_prediction = 0
    n_white_contacts = 0

    for j in survey_reps:
        contact_survey = pickle.load(open('../Data/Contact survey data/' + input_network + '__' + experiment + str(i) + '_' + str(j) + '.pickle', 'rb'))
        for n in contact_survey:
            for u,v,data in G.edges(n, data=True):
                if G.nodes[v]['race'] in target_groups:
                    if data['context'] == 'C_N' or data['context'] == 'C_D':
                        total = total + 1
                        if G.nodes[v]['race'] == contact_survey[u][v]['recall_race_estimate']:
                            correct = correct + 1

    error_rate.append(correct/total)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

aian_study_names = ['Campbell et al.','Kressin et al.','Kelly et al.','Gomez et al.']
aian_innacuracy_magnitude = [0.5761,0.7718,0.571,0.5]

hispanic_study_names = ['Kressin et al.','Kelly et al.','Feliciano','Gomez & Glasner','Moscou et al.']
hispanic_innacuracy_magnitude = [0.1661,0.237,0.2027,0.3153,0.26]

# error_rate = [1.0, 0.9175623585078828, 0.8357142142092039, 0.7564444175296231, 0.6737449768476022, 0.5935073600776448, 0.5097031151571803, 0.42721653662985104, 0.34823715190962556]

r1 = np.linspace(0, 2.05, 9)
rate_of_misid = [1 - h for h in error_rate]
sb.lineplot(x=r1,y=rate_of_misid,ax=ax,label='non-White rate (model)')
plt.axvline(x=1.79, linestyle='dotted',label='$r_{NW}=1.79$ (model)')
for i in range(len(aian_study_names)):
    if i == 0:
        plt.axhline(y=aian_innacuracy_magnitude[i], linestyle='dashed',color='red', alpha=0.5, label='AIAN estimate')
    else:
        plt.axhline(y=aian_innacuracy_magnitude[i], linestyle='dashed',color='red', alpha=0.5)
    if i == 0:
        plt.text(x=2.2,y=aian_innacuracy_magnitude[i]-0.005+0.01,s=aian_study_names[i],size=7)
    elif i == 2:
        plt.text(x=2.2,y=aian_innacuracy_magnitude[i]-0.005-0.01,s=aian_study_names[i],size=7)
    else:
        plt.text(x=2.2,y=aian_innacuracy_magnitude[i]-0.005,s=aian_study_names[i],size=7)


for i in range(len(hispanic_study_names)):
    if i == 0:
        plt.axhline(y=hispanic_innacuracy_magnitude[i], linestyle='dashed',color='purple', alpha=0.5,label='Hispanic estimate')
    else:
        plt.axhline(y=hispanic_innacuracy_magnitude[i], linestyle='dashed',color='purple', alpha=0.5)
    plt.text(x=2.2,y=hispanic_innacuracy_magnitude[i]-0.005,s=hispanic_study_names[i],size=7)

ax.set(title='Bias magnitude ($r_{NW}$) vs. \nrate of non-White misidentification')
ax.set_xlabel('Bias magnitude ($r_{NW}$)')
ax.set_ylabel('Rate of racial misidentification')
for spine in ('top', 'right'):
    ax.spines[spine].set_visible(False)
plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig('../Figures/Supplementary Material/fig4__empirical comparison.pdf')