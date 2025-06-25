import glob
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint
import pickle 
import seaborn as sb
import copy

# plt.rcParams.update({'font.size': 22})

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
    
def run_SIR(c,N,part_dist,pop_split,pathogen,rel_susc_in):

    I0 = []
    R0 = []
    S0 = []

    for i in range(2):
        if i in pop_split:
            I0.append(1)
            R0.append(0)
            S0.append(pop_split[i] - 1)
        else:
            I0.append(0)
            R0.append(0)
            S0.append(0)

    # I0 = [0,100000]
    # R0 = [0, 0]
    # S0 = [pop_split[0], pop_split[1]- 100000]

    if pathogen == 'C':
        target_R = 2.9
        rel_susc = rel_susc_in
        gamma = 0.2
    elif pathogen == 'I':
        target_R = 1.5
        rel_susc = rel_susc_in
        gamma = 0.2
    elif pathogen == 'X':
        target_R = 1.2
        rel_susc = rel_susc_in
        gamma = 0.2

    beta_scaler = 1

    beta = [beta_scaler * h for h in rel_susc]

    for i in range(2):
        for j in range(2):
            c[i][j] = c[i][j] / part_dist[j]

    c_ = copy.deepcopy(c)

    for i in range(2):
        for j in range(2):
            c[i][j] = (1 / pop_split[j]) * ((c_[i][j] * pop_split[j]) + (c_[j][i] * pop_split[i])) / 2

    for i in range(2):
        for j in range(2):
            c[i][j] = c[i][j] / pop_split[i]

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


    t = np.linspace(0, 500, 500)

    y0 = S0 + I0 + R0

    ret = odeint(deriv, y0, t, args=(N, beta, gamma, c, 2))
    out = ret.T

    S_ = []
    I_ = []
    R_ = []

    for t_ in range(len(t)):
        temp_S = 0
        temp_I = 0
        temp_R = 0
        for i in [1]:
            temp_S += out[i][t_]
            temp_I += out[2 + i][t_]
            temp_R += out[(2*2) + i][t_]
        S_.append(temp_S)
        I_.append(temp_I)
        R_.append(temp_R)

    return S_, I_, R_

if __name__ == '__main__':

    input_network = 'NM_network_v3'
    date = '2025-06-05'
    # input_params = ''
    # G = pickle.load(open('../Data/Contact network/' + input_network + '.pickle', 'rb'))
    experiment = 'exp1'
    N_pop = 2089388

    overall_target = 0
    overall_non_target = 0
    part_target = []
    part_non_target = []
    cont_target = []
    cont_non_target = []

    group_lens = {'a':18, 'r':7, 'e':2, 's':3, 'ae':36, 'ar':126, 'as':54}
    rel_susc_binary = {'C':{'a':[0.784,0.894], 'r':[0.811,0.782], 'e':[0.831,0.775], 's':[0.829,0.774], 'ae':[1,1], 'ar':[1,1], 'as':[1,1]},
                       'I':{'a':[0.477,0.16], 'r':[0.412,0.451], 'e':[0.392,0.453], 's':[0.402,0.444], 'ae':[1,1], 'ar':[1,1], 'as':[1,1]},
                       'X':{'a':[1,4], 'r':[0.1,0.2], 'e':[1,1.5], 's':[0.1,0.2], 'ae':[1,1], 'ar':[1,1], 'as':[1,1]}}
    pathogen = 'X'

    biased_x_all = {}
    biased_y_all = {}

    gt_x_all = {}
    gt_y_all = {}

    # input_params_all = ['A_0-0','A_0-5','A_1-0']
    # input_params_all = ['0-0','0-5','1-0']
    # input_params_all = ['state','tract','random']
    input_params_all = ['']
    # race_input_params_all_colors = {'A_0-0':'blue','A_0-5':'purple','A_1-0':'pink'}
    race_input_params_all_colors = {'0-0':'blue','0-5':'purple','1-0':'pink'}
    default_key = '0-0'
    ses_input_params_all_colors = {'state':'blue','tract':'purple','random':'pink'}
    attr = 'e'
    groundtruth_all = [False,True]

    if attr == 'a':
        assort_contact_change = np.linspace(0.13,0.32,19)
        disassort_contact_change = np.linspace(0.01,0.2,19)
    elif attr == 'e':
        assort_contact_change = np.linspace(0.52,0.70,19)
        disassort_contact_change = np.linspace(0.25,0.43,19)
    elif attr == 'r':
        assort_contact_change = np.linspace(0.44,0.8,19)
        disassort_contact_change = np.linspace(0,0.36,19)
    elif attr == 's':
        assort_contact_change = np.linspace(0.5,0.69,19)
        disassort_contact_change = np.linspace(0.2,0.39,19)

    for input_params in input_params_all:
        biased_x = []
        biased_y = []

        gt_x = []
        gt_y = []
        for groundtruth in groundtruth_all:
            if groundtruth:
                search_string_binary = '../Data/Contact matrices/' + input_network +  '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__Binary.npy'
            else:
                search_string_binary = '../Data/Contact matrices/' + input_network +  '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__Binary.npy'

            list_of_files_binary = glob.glob(search_string_binary)

            for i in range(len(list_of_files_binary)):
                cm = np.load(list_of_files_binary[i])

                if i == 0:
                    overall_target = cm[1][1][1]
                    overall_non_target = cm[1][0][1]

                total_non_target_contacts = 0
                total_target_contacts = 0

                for i in range(2):
                    for j in range(2):
                        if j == 1:
                            total_target_contacts += cm[0][i][j]
                        else:
                            total_non_target_contacts += cm[0][i][j]

                if not groundtruth:
                    biased_y.append((1/(max(assort_contact_change) - min(assort_contact_change))) * len(assort_contact_change) * ((cm[0][1][1] / total_target_contacts) - min(assort_contact_change)))
                    biased_x.append((1/(max(disassort_contact_change) - min(disassort_contact_change))) * len(disassort_contact_change) * ((cm[0][1][0] / total_non_target_contacts) - min(disassort_contact_change)))
                else:
                    gt_y.append((1/(max(assort_contact_change) - min(assort_contact_change))) * len(assort_contact_change) * ((cm[0][1][1] / total_target_contacts) - min(assort_contact_change)))
                    gt_x.append((1/(max(disassort_contact_change) - min(disassort_contact_change))) * len(disassort_contact_change) * ((cm[0][1][0] / total_non_target_contacts) - min(disassort_contact_change)))

                part_target.append(cm[1][1][0])
                part_non_target.append(cm[1][0][0])
                cont_target.append(total_target_contacts)
                cont_non_target.append(total_non_target_contacts)

        biased_x_all[input_params] = biased_x
        biased_y_all[input_params] = biased_y

        gt_x_all[input_params] = gt_x
        gt_y_all[input_params] = gt_y


    part_target = np.mean(part_target)
    part_non_target = np.mean(part_non_target)
    cont_target = np.mean(cont_target)
    cont_non_target = np.mean(cont_non_target)


    # Binary analysis - parameter sweep
    peak_prev = []
    final_size = []
    peak_time = []
    # x_labels = []
    # y_labels = []

    for i in range(len(assort_contact_change)):
        peak_prev.append([])
        final_size.append([])
        peak_time.append([])
        for j in range(len(disassort_contact_change)):
            peak_prev[i].append(0)
            final_size[i].append(0)
            peak_time[i].append(0)

    for a in range(len(assort_contact_change)):
        # x_labels.append(cont_non_target * disassort_contact_change[a]/part_non_target)
        # y_labels.append(cont_target * assort_contact_change[a]/part_target)
        for d in range(len(disassort_contact_change)):
            c = [[cont_non_target * (1 - disassort_contact_change[d]),
              cont_target * (1 - assort_contact_change[a])],
              [cont_non_target * disassort_contact_change[d],
              cont_target * assort_contact_change[a]]]
            
            S_temp, I_temp, R_temp = run_SIR(c=c,N=N_pop,part_dist={0:part_non_target,1:part_target}, pop_split={0:overall_non_target,1:overall_target},pathogen=pathogen,
                                          rel_susc_in=rel_susc_binary[pathogen][attr])
        
            peak_prev[a][d] = max(I_temp)
            final_size[a][d] = max(R_temp)
            peak_time[a][d] = np.argmax(I_temp)
            
    fig, ax = plt.subplots(nrows=1,ncols=1)

    sb.heatmap(peak_prev, annot=False, ax=ax, fmt='g')
    CS = ax.contour(np.arange(.5, 19), np.arange(.5, 19), peak_prev, colors='yellow')
    ax.clabel(CS, CS.levels, fontsize=10,colors='yellow',use_clabeltext=True)
    if attr == 'r':
        for i in biased_x_all:
            ax.scatter(biased_x_all[i], biased_y_all[i], marker='o', s=50, color=race_input_params_all_colors[i],label='Biased (l=' + i.replace('-','.') + ')') 
            ax.scatter(np.mean(biased_x_all[i]), np.mean(biased_y_all[i]), marker='*', s=300, color=race_input_params_all_colors[i],label='Biased (mean, l=' + i.replace('-','.') + ')',edgecolors='white') 
        ax.scatter(gt_x_all[default_key], gt_y_all[default_key], marker='o', s=50, color='green',label='True') 
        ax.scatter(np.mean(gt_x_all[default_key]), np.mean(gt_y_all[default_key]), marker='*', s=300, color='green',label='True (mean)',edgecolors='white') 
    elif attr == 's':
        for i in biased_x_all:
            ax.scatter(biased_x_all[i], biased_y_all[i], marker='o', s=50, color=ses_input_params_all_colors[i],label='Biased (' + i.replace('-','.') + ')') 
            ax.scatter(np.mean(biased_x_all[i]), np.mean(biased_y_all[i]), marker='*', s=300, color=ses_input_params_all_colors[i],label='Biased (mean, ' + i.replace('-','.') + ')',edgecolors='white') 
        ax.scatter(gt_x_all['state'], gt_y_all['state'], marker='o', s=50, color='green',label='True') 
        ax.scatter(np.mean(gt_x_all['state']), np.mean(gt_y_all['state']), marker='*', s=300, color='green',label='True (mean)',edgecolors='white') 
    else:
        ax.scatter(biased_x, biased_y, marker='o', s=50, color='blue',label='Biased') 
        ax.scatter(np.mean(biased_x), np.mean(biased_y), marker='*', s=300, color='blue',label='Biased (mean)',edgecolors='white') 
        ax.scatter(gt_x, gt_y, marker='o', s=50, color='green',label='True') 
        ax.scatter(np.mean(gt_x), np.mean(gt_y), marker='*', s=300, color='green',label='True (mean)',edgecolors='white') 
    ax.invert_yaxis()
    ax.set_xticklabels([round(h,3) for h in disassort_contact_change])
    # ax.set_xticklabels([round(h,3) for h in x_labels])
    ax.set_yticklabels([round(h,3) for h in assort_contact_change])
    # ax.set_yticklabels([round(h,3) for h in y_labels])
    if attr == 'a':
        plt.ylabel('Proportion of daily contacts that are with Elderly individuals among Elderly participants')
        # plt.ylabel('Contact rate (contacts/day) with Elderly individuals for Elderly participants')
        plt.xlabel('Proportion of daily contacts that are with Elderly individuals among non-Elderly participants')
        # plt.xlabel('Contact rate (contacts/day) with Elderly individuals for Non-Elderly participants')
        if pathogen == 'C':
            plt.title('Peak prevalence among Elderly (65+) individuals (SARS-CoV-2 - $R_0$ = 2.9)')
        elif pathogen == 'I':
            plt.title('Peak prevalence among Elderly (65+) individuals (H1N1 - $R_0$ = 1.5)')
    elif attr == 'e':
        plt.ylabel('Proportion of daily contacts that are with Hispanic individuals among Hispanic participants')
        # plt.ylabel('Contact rate (contacts/day) with Hispanic individuals for Hispanic participants')
        plt.xlabel('Proportion of daily contacts that are with Hispanic individuals among non-Hispanic participants')
        # plt.xlabel('Contact rate (contacts/day) with Hispanic individuals for non-Hispanic participants')
        if pathogen == 'C':
            plt.title('Peak prevalence among Hispanic individuals (SARS-CoV-2 - $R_0$ = 2.9)')
        elif pathogen == 'I':
            plt.title('Peak prevalence among Hispanic individuals (H1N1 - $R_0$ = 1.5)')
    elif attr == 'r':
        # plt.ylabel('Proportion of non-White contact with non-White')
        plt.ylabel('Contact rate (contacts/day) with non-White individuals for non-White participants')
        # plt.xlabel('Proportion of White contact with non-White')
        plt.xlabel('Contact rate (contacts/day) with non-White individuals for White participants')
        if pathogen == 'C':
            plt.title('Peak prevalence among non-White individuals (SARS-CoV-2 - $R_0$ = 2.9)')
        elif pathogen == 'I':
            plt.title('Peak prevalence among non-White individuals (H1N1 - $R_0$ = 1.5)')
    elif attr == 's':
        # plt.ylabel('Proportion of lower class contact with lower class')
        plt.ylabel('Contact rate (contacts/day) with lower income individuals for lower income participants')
        # plt.xlabel('Proportion of middle/upper class contact with lower class')
        plt.xlabel('Contact rate (contacts/day) with lower income individuals for middle/upper income participants')
        if pathogen == 'C': 
            plt.title('Peak prevalence among lower income individuals (SARS-CoV-2 - $R_0$ = 2.9)')
        elif pathogen == 'I':
            plt.title('Peak prevalence among lower income individuals (H1N1 - $R_0$ = 1.5)')

    plt.legend()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.savefig('../Figures/Supplementary Material/Supp_SIR_binary_' + attr + '_' + pathogen + '_peak_prevalence.pdf')

    fig, ax = plt.subplots(nrows=1,ncols=1)

    sb.heatmap(final_size, annot=False, ax=ax, fmt='g')
    CS = ax.contour(np.arange(.5, 19), np.arange(.5, 19), final_size, colors='yellow')
    ax.clabel(CS, CS.levels, fontsize=10,colors='yellow',use_clabeltext=True)
    if attr == 'r':
        for i in biased_x_all:
            ax.scatter(biased_x_all[i], biased_y_all[i], marker='o', s=50, color=race_input_params_all_colors[i],label='Biased (l=' + i.replace('-','.') + ')') 
            ax.scatter(np.mean(biased_x_all[i]), np.mean(biased_y_all[i]), marker='*', s=300, color=race_input_params_all_colors[i],label='Biased (mean, l=' + i.replace('-','.') + ')',edgecolors='white') 
        ax.scatter(gt_x_all[default_key], gt_y_all[default_key], marker='o', s=50, color='green',label='True') 
        ax.scatter(np.mean(gt_x_all[default_key]), np.mean(gt_y_all[default_key]), marker='*', s=300, color='green',label='True (mean)',edgecolors='white') 
    elif attr == 's':
        for i in biased_x_all:
            ax.scatter(biased_x_all[i], biased_y_all[i], marker='o', s=50, color=ses_input_params_all_colors[i],label='Biased (' + i.replace('-','.') + ')') 
            ax.scatter(np.mean(biased_x_all[i]), np.mean(biased_y_all[i]), marker='*', s=300, color=ses_input_params_all_colors[i],label='Biased (mean, ' + i.replace('-','.') + ')',edgecolors='white') 
        ax.scatter(gt_x_all['state'], gt_y_all['state'], marker='o', s=50, color='green',label='True') 
        ax.scatter(np.mean(gt_x_all['state']), np.mean(gt_y_all['state']), marker='*', s=300, color='green',label='True (mean)',edgecolors='white') 
    else:
        ax.scatter(biased_x, biased_y, marker='o', s=50, color='blue',label='Biased') 
        ax.scatter(np.mean(biased_x), np.mean(biased_y), marker='*', s=300, color='blue',label='Biased (mean)',edgecolors='white') 
        ax.scatter(gt_x, gt_y, marker='o', s=50, color='green',label='True') 
        ax.scatter(np.mean(gt_x), np.mean(gt_y), marker='*', s=300, color='green',label='True (mean)',edgecolors='white') 
    ax.invert_yaxis()
    ax.set_xticklabels([round(h,3) for h in disassort_contact_change])
    # ax.set_xticklabels([round(h,3) for h in x_labels])
    ax.set_yticklabels([round(h,3) for h in assort_contact_change])
    # ax.set_yticklabels([round(h,3) for h in y_labels])
    if attr == 'a':
        plt.ylabel('Proportion of daily contacts that are with Elderly individuals among Elderly participants')
        # plt.ylabel('Contact rate (contacts/day) with Elderly individuals for Elderly participants')
        plt.xlabel('Proportion of daily contacts that are with Elderly individuals among non-Elderly participants')
        # plt.xlabel('Contact rate (contacts/day) with Elderly individuals for Non-Elderly participants')
        if pathogen == 'C':
            plt.title('Final size among Elderly (65+) individuals (SARS-CoV-2 - $R_0$ = 2.9)')
        elif pathogen == 'I':
            plt.title('Final size among Elderly (65+) individuals (H1N1 - $R_0$ = 1.5)')
    elif attr == 'e':
        plt.ylabel('Proportion of daily contacts that are with Hispanic individuals among Hispanic participants')
        # plt.ylabel('Contact rate (contacts/day) with Hispanic individuals for Hispanic participants')
        plt.xlabel('Proportion of daily contacts that are with Hispanic individuals among non-Hispanic participants')
        # plt.xlabel('Contact rate (contacts/day) with Hispanic individuals for non-Hispanic participants')
        if pathogen == 'C':
            plt.title('Final size among Hispanic individuals (SARS-CoV-2 - $R_0$ = 2.9)')
        elif pathogen == 'I':
            plt.title('Final size among Hispanic individuals (H1N1 - $R_0$ = 1.5)')
    elif attr == 'r':
        # plt.ylabel('Proportion of non-White contact with non-White')
        plt.ylabel('Contact rate (contacts/day) with non-White individuals for non-White participants')
        # plt.xlabel('Proportion of White contact with non-White')
        plt.xlabel('Contact rate (contacts/day) with non-White individuals for White participants')
        if pathogen == 'C':
            plt.title('Final size among non-White individuals (SARS-CoV-2 - $R_0$ = 2.9)')
        elif pathogen == 'I':
            plt.title('Final size among non-White individuals (H1N1 - $R_0$ = 1.5)')
    elif attr == 's':
        # plt.ylabel('Proportion of lower class contact with lower class')
        plt.ylabel('Contact rate (contacts/day) with lower income individuals for lower income participants')
        # plt.xlabel('Proportion of middle/upper class contact with lower class')
        plt.xlabel('Contact rate (contacts/day) with lower income individuals for middle/upper income participants')
        if pathogen == 'C': 
            plt.title('Final size among lower income individuals (SARS-CoV-2 - $R_0$ = 2.9)')
        elif pathogen == 'I':
            plt.title('Final size among lower income individuals (H1N1 - $R_0$ = 1.5)')

    plt.legend()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.savefig('../Figures/Supplementary Material/Supp_SIR_binary_' + attr + '_' + pathogen + '_final_size.pdf')

    
    fig, ax = plt.subplots(nrows=1,ncols=1)

    sb.heatmap(peak_time, annot=False, ax=ax, fmt='g')
    CS = ax.contour(np.arange(.5, 19), np.arange(.5, 19), peak_time, colors='yellow')
    ax.clabel(CS, CS.levels, fontsize=10,colors='yellow',use_clabeltext=True)
    if attr == 'r':
        for i in biased_x_all:
            ax.scatter(biased_x_all[i], biased_y_all[i], marker='o', s=50, color=race_input_params_all_colors[i],label='Biased (l=' + i.replace('-','.') + ')') 
            ax.scatter(np.mean(biased_x_all[i]), np.mean(biased_y_all[i]), marker='*', s=300, color=race_input_params_all_colors[i],label='Biased (mean, l=' + i.replace('-','.') + ')',edgecolors='white') 
        ax.scatter(gt_x_all[default_key], gt_y_all[default_key], marker='o', s=50, color='green',label='True') 
        ax.scatter(np.mean(gt_x_all[default_key]), np.mean(gt_y_all[default_key]), marker='*', s=300, color='green',label='True (mean)',edgecolors='white') 
    elif attr == 's':
        for i in biased_x_all:
            ax.scatter(biased_x_all[i], biased_y_all[i], marker='o', s=50, color=ses_input_params_all_colors[i],label='Biased (' + i.replace('-','.') + ')') 
            ax.scatter(np.mean(biased_x_all[i]), np.mean(biased_y_all[i]), marker='*', s=300, color=ses_input_params_all_colors[i],label='Biased (mean, ' + i.replace('-','.') + ')',edgecolors='white') 
        ax.scatter(gt_x_all['state'], gt_y_all['state'], marker='o', s=50, color='green',label='True') 
        ax.scatter(np.mean(gt_x_all['state']), np.mean(gt_y_all['state']), marker='*', s=300, color='green',label='True (mean)',edgecolors='white') 
    else:
        ax.scatter(biased_x, biased_y, marker='o', s=50, color='blue',label='Biased') 
        ax.scatter(np.mean(biased_x), np.mean(biased_y), marker='*', s=300, color='blue',label='Biased (mean)',edgecolors='white') 
        ax.scatter(gt_x, gt_y, marker='o', s=50, color='green',label='True') 
        ax.scatter(np.mean(gt_x), np.mean(gt_y), marker='*', s=300, color='green',label='True (mean)',edgecolors='white') 
    ax.invert_yaxis()
    ax.set_xticklabels([round(h,3) for h in disassort_contact_change])
    # ax.set_xticklabels([round(h,3) for h in x_labels])
    ax.set_yticklabels([round(h,3) for h in assort_contact_change])
    # ax.set_yticklabels([round(h,3) for h in y_labels])
    if attr == 'a':
        plt.ylabel('Proportion of daily contacts that are with Elderly individuals among Elderly participants')
        # plt.ylabel('Contact rate (contacts/day) with Elderly individuals for Elderly participants')
        plt.xlabel('Proportion of daily contacts that are with Elderly individuals among non-Elderly participants')
        # plt.xlabel('Contact rate (contacts/day) with Elderly individuals for Non-Elderly participants')
        if pathogen == 'C':
            plt.title('Peak prevalence time among Elderly (65+) individuals (SARS-CoV-2 - $R_0$ = 2.9)')
        elif pathogen == 'I':
            plt.title('Peak prevalence time among Elderly (65+) individuals (H1N1 - $R_0$ = 1.5)')
    elif attr == 'e':
        plt.ylabel('Proportion of daily contacts that are with Hispanic individuals among Hispanic participants')
        # plt.ylabel('Contact rate (contacts/day) with Hispanic individuals for Hispanic participants')
        plt.xlabel('Proportion of daily contacts that are with Hispanic individuals among non-Hispanic participants')
        # plt.xlabel('Contact rate (contacts/day) with Hispanic individuals for non-Hispanic participants')
        if pathogen == 'C':
            plt.title('Peak prevalence time among Hispanic individuals (SARS-CoV-2 - $R_0$ = 2.9)')
        elif pathogen == 'I':
            plt.title('Peak prevalence time among Hispanic individuals (H1N1 - $R_0$ = 1.5)')
    elif attr == 'r':
        # plt.ylabel('Proportion of non-White contact with non-White')
        plt.ylabel('Contact rate (contacts/day) with non-White individuals for non-White participants')
        # plt.xlabel('Proportion of White contact with non-White')
        plt.xlabel('Contact rate (contacts/day) with non-White individuals for White participants')
        if pathogen == 'C':
            plt.title('Peak prevalence time among non-White individuals (SARS-CoV-2 - $R_0$ = 2.9)')
        elif pathogen == 'I':
            plt.title('Peak prevalence time among non-White individuals (H1N1 - $R_0$ = 1.5)')
    elif attr == 's':
        # plt.ylabel('Proportion of lower class contact with lower class')
        plt.ylabel('Contact rate (contacts/day) with lower income individuals for lower income participants')
        # plt.xlabel('Proportion of middle/upper class contact with lower class')
        plt.xlabel('Contact rate (contacts/day) with lower income individuals for middle/upper income participants')
        if pathogen == 'C': 
            plt.title('Peak prevalence time among lower income individuals (SARS-CoV-2 - $R_0$ = 2.9)')
        elif pathogen == 'I':
            plt.title('Peak prevalence time among lower income individuals (H1N1 - $R_0$ = 1.5)')

    plt.legend()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.savefig('../Figures/Supplementary Material/Supp_SIR_binary_' + attr + '_' + pathogen + '_peak_time.pdf')
