# Script for running SIR epidemic model given a contact matrix

import glob
import numpy as np
from scipy.integrate import odeint

# Compute dominant eigenvalue of next generation matrix (NGM)
def get_R0(cm, p_ratio, base_beta, gamma):
    T = []

    for i in range(len(cm)):
        T.append([])
        for j in range(len(cm)):
            T[i].append(0)


    for i in range(len(cm)):
        for j in range(len(cm)):
            T[i][j] = base_beta[i] * p_ratio[i][j] * cm[i][j]

    sigma = -1 * np.diag([gamma]*len(cm))
    
    E_ = np.diag([1]*len(cm))

    K = -1 * np.linalg.multi_dot([np.transpose(E_),T,np.linalg.inv(sigma),E_])

    y = np.linalg.eigvals(K)

    return float(max(abs(y)))

# Update function of SIR model
def deriv(y, t, N, beta, gamma, c, n_g):
    
    S_ = y[0:n_g]
    I_ = y[n_g:2*n_g]
    R_ = y[2*n_g:3*n_g]

    lambda_ = []
    dSdt = []
    dIdt = []
    dRdt = []

    for i in range(0,n_g):
        lambda_.append([])
        dSdt.append(0)
        dIdt.append(0)
        dRdt.append(0)
        for j in range(0,n_g):
            lambda_[i].append(0)

    for i in range(0,n_g):
        for j in range(0,n_g):
            lambda_[i][j] = beta[i] * I_[j] * c[i][j]

    for i in range(0,n_g):
        dSdt[i] = -sum(lambda_[i]) * S_[i]
        dIdt[i] = sum(lambda_[i]) * S_[i] - gamma * I_[i]
        dRdt[i] = gamma * I_[i]
    return dSdt + dIdt + dRdt

# Execute SIR model 
def run_SIR(c,N,n_g,pop_split,pathogen,target_groups,target_R):

    I0 = []
    R0 = []
    S0 = []

    for i in range(n_g):
        if i in pop_split:
            I0.append(0.0001*pop_split[i])
            R0.append(0)
            S0.append(pop_split[i] - (0.0001*pop_split[i]))
        else:
            I0.append(0)
            R0.append(0)
            S0.append(0)

    # Age-dependent susceptibility - COVID-19 (C), uniform (X)
    if pathogen[0] == 'C':
        rel_susc = [0.4, 0.4, 0.38, 0.38, 0.79, 0.79, 0.86, 0.86, 0.8, 0.8, 0.82, 0.82, 0.88, 0.88, 0.74, 0.74, 0.74, 0.74]
        gamma = 0.2
    elif pathogen[0] == 'X':
        rel_susc = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        gamma = 0.2

    beta_scaler = 1

    beta = [beta_scaler * h for h in rel_susc for _ in range(0,int(n_g / 18))]

    p_split = []

    for i in range(len(c)):
        p_split.append([])
        for j in range(len(c)):
            p_split[i].append(0)

    for i in range(len(c)):
        for j in range(len(c)):
            if i in pop_split:
                p_split[i][j] = pop_split[i] 
            else:
                p_split[i][j] = 0

    y = get_R0(cm=c,p_ratio=p_split,base_beta=beta,gamma=gamma)
    beta_scaler = y/target_R

    beta = [h/beta_scaler for h in beta]

    t = np.linspace(0, 600, 600)

    y0 = S0 + I0 + R0

    ret = odeint(deriv, y0, t, args=(N, beta, gamma, c, n_g))
    out = ret.T

    S_ = []
    I_ = []
    R_ = []

    for t_ in range(len(t)):
        temp_S = 0
        temp_I = 0
        temp_R = 0
        for i in target_groups:
            temp_S += out[i][t_]
            temp_I += out[n_g + i][t_]
            temp_R += out[(2*n_g) + i][t_] - out[(2*n_g) + i][0]
        S_.append(temp_S)
        I_.append(temp_I)
        R_.append(temp_R)

    return S_, I_, R_

if __name__ == '__main__':

    # Define input contact network
    input_network = 'NM_network'

    # Define experimental conditions for contact survey simulation
    # Main analysis: Experiment 1 ('exp1'), Experiment 2 - part A ('exp2_A'), Experiment 2 - part B ('exp2_B')
    # Supplemental analysis: ethnicity bias ('supp_eth'), income bias ('supp_income'), within-group bias ('supp_wg'), transmissing setting SA ('supp_exp_context')
    experiments = ['exp1','exp2_A','exp2_B']

    # Define total size of synthetic population
    N_pop = 2089388

    # Define total number of groups under different stratification settings
    group_lens = {'a':18, 'r':7, 'e':2, 's':3, 'ae':36, 'ar':126, 'as':54}

    # Define target pathogen - COVID-19 (C), uniform age-specific susceptibility (X)
    pathogen = 'C'

    # Define whether to analyse processed and/or raw contact matrices
    raw_all = [False]

    # Define whether to analyse biased and/or true contact matrices
    groundtruth_all = [False,True]

    for experiment in experiments:

        # Define whether to use average matrices or individual matrices from single survey simulations (use both by default)
        average_all = [True,False]

        # Define default target R0 values 
        R_0 = [1.2,1.4,1.5,1.6,2,2.4,2.8,2.9,3.2,3.6,4,4.4,4.8,5.2,5.6,6]

        # Define parameter range specific to experiment
        if experiment == 'exp1':       
            input_params_all = [str(h) for h in range(7)]
            target_groups_all = {'a':['elderly','children','adults']}
            attr_all = ['a']
        elif experiment == 'exp2_A':
            input_params_all = ['tract_7']
            target_groups_all = {'ar':['Non-White']}
            attr_all = ['ar']
            R_0 = [1.4,2.9]
            experiment = 'exp2'
        elif experiment == 'exp2_B':
            input_params_all = [p + '_' + str(h) for h in range(9) for p in ['tract']]
            target_groups_all = {'ar':['Non-White','White']}
            attr_all = ['ar']
            average_all = [True]
            experiment = 'exp2'
        elif experiment == 'supp_wg':
            input_params_all = [str(h) for h in range(5)]
            target_groups_all = {'ar':['Non-White', 'White']}
            attr_all = ['ar']
            R_0 = [2.9]
        elif experiment == 'supp_exp_context':
            input_params_all = [str(h) for h in range(8)]
            target_groups_all = {'ar':['Non-White', 'White'],'a':['elderly','children','adults']}
            attr_all = ['a','ar']
            average_all = [True]
            R_0 = [2.9]
        elif experiment == 'supp_eth':
            input_params_all = [p + '_' + str(h) for h in range(9) for p in ['tract']]
            target_groups_all = {'ae':['Non-Hispanic','Hispanic']}
            attr_all = ['ae']
            R_0 = [1.4,2.9]
        elif experiment == 'supp_income':
            input_params_all = [p + '_' + str(h) for h in range(9) for p in ['tract']]
            target_groups_all = {'as':['Low','Medium','High']}
            attr_all = ['as']
            R_0 = [1.4,2.9]
        

        # Iterate through SIR model parameter combinations; run SIR model for each setting and output infectious+recovered curves to file
        for input_params in input_params_all:
            for attr in attr_all:
                for r in R_0:
                    for target_groups_input in target_groups_all[attr]:
                        for average in average_all:
                            for raw in raw_all:
                                for groundtruth in groundtruth_all:
                                    if groundtruth:
                                        search_string = '../Data/Contact matrices/' + input_network +  '__' + experiment + '__' + input_params + '_[0-9]' + '__gt__processed__' + attr + '__Overall.npy'
                                    else:
                                        search_string = '../Data/Contact matrices/' + input_network +  '__' + experiment + '__' + input_params + '_[0-9]' + '__biased__processed__' + attr + '__Overall.npy'

                                    if average:
                                        if groundtruth:
                                            list_of_files = ['../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*' + '__gt__processed__' + attr + '__Overall.npy']
                                        else:
                                            list_of_files = ['../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*' + '__biased__processed__' + attr + '__Overall.npy']
                                    else:
                                        list_of_files = glob.glob(search_string)

                                    
                                    if attr == 'a':
                                        if target_groups_input == 'elderly':
                                            target_groups = [x for x in list(range(group_lens[attr])) if x > 12] # elderly
                                        elif target_groups_input == 'children':
                                            target_groups = [x for x in list(range(group_lens[attr])) if x < 4] # children
                                        else:
                                            target_groups = [x for x in list(range(group_lens[attr])) if x >= 4 and x <= 12] # adults
                                        pop_dist = {0: 120467, 1: 133396, 2: 141965, 3: 136600, 4: 130735, 5: 134971, 6: 138814, 7: 128595, 8: 118190, 9: 122808, 10: 133518, 11: 143473, 12: 139797, 13: 124096, 14: 94146, 15: 64665, 16: 43637, 17: 39515}
                                    
                                    if attr == 'ae':
                                        if target_groups_input == 'Hispanic':
                                            target_groups = [x for x in list(range(group_lens[attr])) if x % 2 != 0] # hispanic
                                        else:
                                            target_groups = [x for x in list(range(group_lens[attr])) if x % 2 == 0] # non-hispanic
                                        pop_dist = {0: 48810, 1: 71657, 2: 54460, 3: 78936, 4: 56836, 5: 85129, 6: 53764, 7: 82836, 8: 58819, 9: 71916, 10: 64166, 11: 70805, 12: 68870, 13: 69944, 14: 62894, 15: 65701, 16: 57878, 17: 60312, 18: 62080, 19: 60728, 20: 71078, 21: 62440, 22: 84997, 23: 58476, 24: 87833, 25: 51964, 26: 82300, 27: 41796, 28: 63948, 29: 30198, 30: 43126, 31: 21539, 32: 28585, 33: 15052, 34: 26199, 35: 13316}

                                    if attr == 'ar':
                                        if target_groups_input == 'White':
                                            target_groups = [x for x in list(range(group_lens[attr])) if x % 7 == 0] # white
                                        elif target_groups_input == 'Non-White':
                                            target_groups = [x for x in list(range(group_lens[attr])) if x % 7 != 0] # non-white
                                        elif target_groups_input == 'AIAN':
                                            target_groups = [x for x in list(range(group_lens[attr])) if x % 7 == 3] # AIAN
                                        pop_dist = {0: 85398, 1: 2367, 2: 1842, 3: 13452, 4: 141, 5: 9503, 6: 7764, 7: 92728, 8: 2700, 9: 1380, 10: 16448, 11: 107, 12: 11203, 13: 8830, 14: 100459, 15: 2249, 16: 1704, 17: 16850, 18: 103, 19: 13359, 20: 7241, 21: 97282, 22: 3197, 23: 1845, 24: 15171, 25: 57, 26: 13428, 27: 5620, 28: 91478, 29: 4336, 30: 2031, 31: 14757, 32: 290, 33: 12565, 34: 5278, 35: 95819, 36: 3371, 37: 3031, 38: 14397, 39: 139, 40: 13394, 41: 4820, 42: 102244, 43: 3667, 44: 3065, 45: 14671, 46: 121, 47: 11164, 48: 3882, 49: 94305, 50: 2785, 51: 2704, 52: 12424, 53: 75, 54: 12900, 55: 3402, 56: 85656, 57: 2478, 58: 2195, 59: 12740, 60: 40, 61: 12420, 62: 2661, 63: 91060, 64: 2916, 65: 2296, 66: 11293, 67: 189, 68: 12480, 69: 2574, 70: 98412, 71: 2504, 72: 2514, 73: 13349, 74: 22, 75: 13972, 76: 2745, 77: 113279, 78: 2012, 79: 2195, 80: 11183, 81: 44, 82: 11961, 83: 2799, 84: 112633, 85: 2722, 86: 2129, 87: 10752, 88: 50, 89: 8811, 90: 2700, 91: 103649, 92: 2559, 93: 1745, 94: 7415, 95: 13, 96: 6950, 97: 1765, 98: 78994, 99: 1225, 100: 1142, 101: 5656, 102: 32, 103: 5903, 104: 1194, 105: 55788, 106: 568, 107: 340, 108: 3739, 110: 3224, 111: 1006, 112: 36144, 113: 452, 114: 304, 115: 2773, 116: 31, 117: 2966, 118: 967, 119: 33507, 120: 735, 121: 206, 122: 2177, 124: 2250, 125: 640}

                                    if attr == 'as':
                                        if target_groups_input == 'Low':
                                            target_groups = [x for x in list(range(group_lens[attr])) if x % 3 == 0] # low income
                                        elif target_groups_input == 'Medium':
                                            target_groups = [x for x in list(range(group_lens[attr])) if x % 3 == 1] # medium income
                                        else:
                                            target_groups = [x for x in list(range(group_lens[attr])) if x % 3 == 2] # high income
                                        pop_dist = {0: 70917, 1: 44331, 2: 5219, 3: 76846, 4: 50304, 5: 6246, 6: 75339, 7: 58841, 8: 7785, 9: 74227, 10: 55413, 11: 6960, 12: 72959, 13: 52356, 14: 5420, 15: 64633, 16: 63976, 17: 6362, 18: 61985, 19: 67422, 20: 9407, 21: 57531, 22: 60122, 23: 10942, 24: 46923, 25: 59701, 26: 11566, 27: 45636, 28: 61132, 29: 16040, 30: 49150, 31: 63878, 32: 20490, 33: 49946, 34: 68360, 35: 25167, 36: 51769, 37: 65372, 38: 22656, 39: 45096, 40: 59699, 41: 19301, 42: 37941, 43: 43820, 44: 12385, 45: 28675, 46: 28969, 47: 7021, 48: 21250, 49: 18242, 50: 4145, 51: 20637, 52: 15597, 53: 3281}

                                    if average:
                                        cm = np.load(list_of_files[0])
                                        S_out, I_out, R_out = run_SIR(c=cm, N=N_pop, n_g=group_lens[attr], pop_split=pop_dist, pathogen=pathogen, target_groups=target_groups, target_R=r)
                                        if groundtruth:
                                            np.save('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__gt__processed__' + attr + '__' + pathogen + '_' + str(r).replace('.','_') + '__Overall_Infectious.npy',arr=I_out)
                                            np.save('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__gt__processed__' + attr + '__' + pathogen + '_' + str(r).replace('.','_') + '__Overall_Recovered.npy',arr=R_out)
                                            
                                        else:
                                            np.save('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__biased__processed__' + attr + '__' + pathogen + '_' + str(r).replace('.','_') + '__Overall_Infectious.npy',arr=I_out)
                                            np.save('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_*' + '__biased__processed__' + attr + '__' + pathogen + '_' + str(r).replace('.','_') + '__Overall_Recovered.npy',arr=R_out)

                                    else:
                                        for i in range(len(list_of_files)):
                                            cm = np.load(list_of_files[i])
                                            if groundtruth:
                                                ind = list_of_files[i].find('gt')
                                            else:
                                                ind = list_of_files[i].find('biased')
                                            run_no = list_of_files[i][ind - 3]

                                            S_out, I_out, R_out = run_SIR(c=cm, N=N_pop, n_g=group_lens[attr], pop_split=pop_dist, pathogen=pathogen, target_groups=target_groups, target_R=r)
                                            if groundtruth:
                                                np.save('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_' + run_no +  '__gt__processed__' + attr + '__' + pathogen + '_' + str(r).replace('.','_') + '__Overall_Infectious.npy',arr=I_out)
                                                np.save('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_' + run_no +  '__gt__processed__' + attr + '__' + pathogen + '_' + str(r).replace('.','_') + '__Overall_Recovered.npy',arr=R_out)
                                            else:
                                                np.save('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_' + run_no +  '__biased__processed__' + attr + '__' + pathogen + '_' + str(r).replace('.','_') + '__Overall_Infectious.npy',arr=I_out)
                                                np.save('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '__' + target_groups_input + '_' + run_no +  '__biased__processed__' + attr + '__' + pathogen + '_' + str(r).replace('.','_') + '__Overall_Recovered.npy',arr=R_out)