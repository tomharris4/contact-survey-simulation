import glob
import numpy as np
from scipy.integrate import odeint
import pickle 

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

def get_pop_dist(G, attr):

    bins = np.linspace(0,85,18)
    R = 7
    E = 2
    S = 3

    dist = np.empty(len(G.nodes()),dtype=np.int32)
    dist_dict = {}

    k = 0

    if attr == 'a':
        for node in G.nodes:
            i = np.digitize(G.nodes[node]['age'],bins) - 1
            dist[k] = i
            k += 1
    elif attr == 'r':
        for node in G.nodes:
            i = G.nodes[node]['race']
            dist[k] = i
            k += 1
    elif attr == 'e':
        for node in G.nodes:
            i = G.nodes[node]['ethnicity']
            dist[k] = i
            k += 1
    elif attr == 's':
        for node in G.nodes:
            i = G.nodes[node]['ses'] - 1
            dist[k] = i
            k += 1
    elif attr == 'ar':
        for node in G.nodes:
            i_age = np.digitize(G.nodes[node]['age'],bins) - 1
            i_race = G.nodes[node]['race']
            dist[k] = i_age * R + i_race
            k += 1
    elif attr == 'ae':
        for node in G.nodes:
            i_age = np.digitize(G.nodes[node]['age'],bins) - 1
            i_eth = G.nodes[node]['ethnicity']
            dist[k] = i_age * E + i_eth
            k += 1
    elif attr == 'as':
        for node in G.nodes:
            i_age = np.digitize(G.nodes[node]['age'],bins) - 1
            i_ses = G.nodes[node]['ses'] - 1
            dist[k] = i_age * S + i_ses
            k += 1

    dist_uniq = np.unique(dist, return_counts=True)

    for i in range(len(dist_uniq[0])):
        dist_dict[dist_uniq[0][i]] = dist_uniq[1][i]

    return dist_dict
    
def run_SIR(c,N,n_g,pop_split,pathogen,target_groups):

    I0 = []
    R0 = []
    S0 = []

    for i in range(n_g):
        if i in pop_split:
            I0.append(1)
            R0.append(0)
            S0.append(pop_split[i] - 1)
        else:
            I0.append(0)
            R0.append(0)
            S0.append(0)

    if pathogen == 'C':
        target_R = 2.9
        rel_susc = [0.45, 0.45, 0.43, 0.43, 0.90, 0.90, 0.98, 0.98, 0.91, 0.91, 0.93, 0.93, 1, 1, 0.84, 0.84, 0.84, 0.84]
        gamma = 0.2
    elif pathogen == 'I':
        target_R = 1.5
        # rel_susc = [1, 0.33, 0.33, 0.33, 0.36, 0.36, 0.36, 0.36, 0.36 , 0.36, 0.44, 0.44, 0.44, 0.31, 0.31, 0.31, 0.31, 0.31]
        # rel_susc = [0.01, 0.01, 0.01, 0.01, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 100, 100]
        rel_susc = [1, 0.57, 0.57, 0.57, 0.42, 0.42, 0.42, 0.42, 0.42 , 0.42, 0.34, 0.34, 0.34, 0.16, 0.16, 0.16, 0.16, 0.16]
        gamma = 0.2
    elif pathogen == 'X':
        target_R = 1.4
        rel_susc = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1]
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


    t = np.linspace(0, 300, 300)

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
            temp_R += out[(2*n_g) + i][t_]
        S_.append(temp_S)
        I_.append(temp_I)
        R_.append(temp_R)

    return S_, I_, R_

if __name__ == '__main__':

    input_network = 'NM_network_v3'
    date = '2025-06-25'
    # input_params = ''
    # G = pickle.load(open('../Data/Contact network/' + input_network + '.pickle', 'rb'))
    experiment = 'exp2b'
    N_pop = 2089388

    # attr = 'a'
    group_lens = {'a':18, 'r':7, 'e':2, 's':3, 'ae':36, 'ar':126, 'as':54}
    # raw = False
    # groundtruth = False
    # average = True
    pathogen = 'X'

    input_params_all = [str(h) for h in range(3)]#['state','tract','random','dominant']#['0']#['0-0','0-5','1-0']#['A_0-0','A_0-5','A_1-0']#['']
    attr_all = [ 'ar']
    raw_all = [False]
    groundtruth_all = [False,True]
    average_all = [False, True]

    for input_params in input_params_all:
        for attr in attr_all:
            for average in average_all:
                for raw in raw_all:
                    for groundtruth in groundtruth_all:

                        if groundtruth:
                            search_string = '../Data/Contact matrices/' + input_network +  '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__Overall.npy'
                        else:
                            search_string = '../Data/Contact matrices/' + input_network +  '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__Overall.npy'

                        if average:
                            if groundtruth:
                                list_of_files = ['../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__Overall.npy']
                            else:
                                list_of_files = ['../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__Overall.npy']
                        else:
                            list_of_files = glob.glob(search_string)


                        # pop_dist = get_pop_dist(G=G,attr=attr)
                        # print(pop_dist)
                        
                        #EXP 1: age
                        if attr == 'a':
                            target_groups = [x for x in list(range(group_lens[attr])) if x > 12] # elderly
                            pop_dist = {0: 120467, 1: 133396, 2: 141965, 3: 136600, 4: 130735, 5: 134971, 6: 138814, 7: 128595, 8: 118190, 9: 122808, 10: 133518, 11: 143473, 12: 139797, 13: 124096, 14: 94146, 15: 64665, 16: 43637, 17: 39515}
                        #EXP 1: ethnicity
                        if attr == 'ae':
                            target_groups = [x for x in list(range(group_lens[attr])) if x % 2 != 0] # hispanic
                            pop_dist = {0: 48810, 1: 71657, 2: 54460, 3: 78936, 4: 56836, 5: 85129, 6: 53764, 7: 82836, 8: 58819, 9: 71916, 10: 64166, 11: 70805, 12: 68870, 13: 69944, 14: 62894, 15: 65701, 16: 57878, 17: 60312, 18: 62080, 19: 60728, 20: 71078, 21: 62440, 22: 84997, 23: 58476, 24: 87833, 25: 51964, 26: 82300, 27: 41796, 28: 63948, 29: 30198, 30: 43126, 31: 21539, 32: 28585, 33: 15052, 34: 26199, 35: 13316}

                        #EXP 2
                        if attr == 'ar':
                            target_groups = [x for x in list(range(group_lens[attr])) if x % 7 != 0] # non-white
                            pop_dist = {0: 85398, 1: 2367, 2: 1842, 3: 13452, 4: 141, 5: 9503, 6: 7764, 7: 92728, 8: 2700, 9: 1380, 10: 16448, 11: 107, 12: 11203, 13: 8830, 14: 100459, 15: 2249, 16: 1704, 17: 16850, 18: 103, 19: 13359, 20: 7241, 21: 97282, 22: 3197, 23: 1845, 24: 15171, 25: 57, 26: 13428, 27: 5620, 28: 91478, 29: 4336, 30: 2031, 31: 14757, 32: 290, 33: 12565, 34: 5278, 35: 95819, 36: 3371, 37: 3031, 38: 14397, 39: 139, 40: 13394, 41: 4820, 42: 102244, 43: 3667, 44: 3065, 45: 14671, 46: 121, 47: 11164, 48: 3882, 49: 94305, 50: 2785, 51: 2704, 52: 12424, 53: 75, 54: 12900, 55: 3402, 56: 85656, 57: 2478, 58: 2195, 59: 12740, 60: 40, 61: 12420, 62: 2661, 63: 91060, 64: 2916, 65: 2296, 66: 11293, 67: 189, 68: 12480, 69: 2574, 70: 98412, 71: 2504, 72: 2514, 73: 13349, 74: 22, 75: 13972, 76: 2745, 77: 113279, 78: 2012, 79: 2195, 80: 11183, 81: 44, 82: 11961, 83: 2799, 84: 112633, 85: 2722, 86: 2129, 87: 10752, 88: 50, 89: 8811, 90: 2700, 91: 103649, 92: 2559, 93: 1745, 94: 7415, 95: 13, 96: 6950, 97: 1765, 98: 78994, 99: 1225, 100: 1142, 101: 5656, 102: 32, 103: 5903, 104: 1194, 105: 55788, 106: 568, 107: 340, 108: 3739, 110: 3224, 111: 1006, 112: 36144, 113: 452, 114: 304, 115: 2773, 116: 31, 117: 2966, 118: 967, 119: 33507, 120: 735, 121: 206, 122: 2177, 124: 2250, 125: 640}

                        #EXP 3
                        if attr == 'as':
                            target_groups = [x for x in list(range(group_lens[attr])) if x % 3 == 0]
                            pop_dist = {0: 70917, 1: 44331, 2: 5219, 3: 76846, 4: 50304, 5: 6246, 6: 75339, 7: 58841, 8: 7785, 9: 74227, 10: 55413, 11: 6960, 12: 72959, 13: 52356, 14: 5420, 15: 64633, 16: 63976, 17: 6362, 18: 61985, 19: 67422, 20: 9407, 21: 57531, 22: 60122, 23: 10942, 24: 46923, 25: 59701, 26: 11566, 27: 45636, 28: 61132, 29: 16040, 30: 49150, 31: 63878, 32: 20490, 33: 49946, 34: 68360, 35: 25167, 36: 51769, 37: 65372, 38: 22656, 39: 45096, 40: 59699, 41: 19301, 42: 37941, 43: 43820, 44: 12385, 45: 28675, 46: 28969, 47: 7021, 48: 21250, 49: 18242, 50: 4145, 51: 20637, 52: 15597, 53: 3281}

                        if average:
                            cm = np.load(list_of_files[0])
                            S_out, I_out, R_out = run_SIR(c=cm, N=N_pop, n_g=group_lens[attr], pop_split=pop_dist, pathogen=pathogen, target_groups=target_groups)
                            if groundtruth:
                                np.save('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy',arr=I_out)
                                np.save('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy',arr=R_out)
                            else:
                                np.save('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy',arr=I_out)
                                np.save('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy',arr=R_out)
                        else:
                            for i in range(len(list_of_files)):
                                cm = np.load(list_of_files[i])
                                S_out, I_out, R_out = run_SIR(c=cm, N=N_pop, n_g=group_lens[attr], pop_split=pop_dist, pathogen=pathogen, target_groups=target_groups)
                                if groundtruth:
                                    np.save('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy',arr=I_out)
                                    np.save('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__gt__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy',arr=R_out)
                                else:
                                    np.save('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Infectious.npy',arr=I_out)
                                    np.save('../Data/SIR trajectories/' + input_network + '__' + experiment + '__' + input_params + '_' + str(i) + '_' + date + '__biased__processed__' + attr + '__' + pathogen + '__Overall_Recovered.npy',arr=R_out)
                            