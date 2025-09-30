# Script for constructing contact matrices from simulated contact surveys or full contact networks

import glob
import numpy as np
import pickle
import copy

# Generate population distribution by sociodemographic atrribute ('attr') among participants in a contact survey and overall population in contact network
def get_pop_dist(G, attr, contact_survey):

    # Define age brackets - 0-85years by 5year age groups
    bins = np.linspace(0,85,18)

    # Define constants for total number of groups by race (R), ethnicity (E) and income (S)
    R = 7
    E = 2
    S = 3
    
    # If contact_survey == None, return overall population distribution; otherwise, return survey participants distribution & overall population distribution
    if not contact_survey:

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

        return dist_dict, dist_dict
    
    else:

        contact_survey_dict = pickle.load(open(contact_survey, 'rb'))

        dist = np.empty(len(G.nodes()),dtype=np.int32)
        dist_dict = {}

        dist_sampled = np.empty(len(contact_survey_dict),dtype=np.int32)
        dist_sampled_dict = {}

        k = 0

        if attr == 'a':
            for node in G.nodes:
                i = np.digitize(G.nodes[node]['age'],bins) - 1
                dist[k] = i
                k += 1
            k = 0
            for node in contact_survey_dict:
                i = np.digitize(G.nodes[node]['age'],bins) - 1
                dist_sampled[k] = i
                k += 1
        elif attr == 'r':
            for node in G.nodes:
                i = G.nodes[node]['race']
                dist[k] = i
                k += 1
            k = 0
            for node in contact_survey_dict:
                i = G.nodes[node]['race']
                dist_sampled[k] = i
                k += 1
        elif attr == 'e':
            for node in G.nodes:
                i = G.nodes[node]['ethnicity']
                dist[k] = i
                k += 1
            k = 0
            for node in contact_survey_dict:
                i = G.nodes[node]['ethnicity']
                dist_sampled[k] = i
                k += 1
        elif attr == 's':
            for node in G.nodes:
                i = G.nodes[node]['ses'] - 1
                dist[k] = i
                k += 1
            k = 0
            for node in contact_survey_dict:
                i = G.nodes[node]['ses'] - 1
                dist_sampled[k] = i
                k += 1
        elif attr == 'ar':
            for node in G.nodes:
                i_age = np.digitize(G.nodes[node]['age'],bins) - 1
                i_race = G.nodes[node]['race']
                dist[k] = i_age * R + i_race
                k += 1
            k = 0
            for node in contact_survey_dict:
                i_age = np.digitize(G.nodes[node]['age'],bins) - 1
                i_race = G.nodes[node]['race']
                dist_sampled[k] = i_age * R + i_race
                k += 1
        elif attr == 'ae':
            for node in G.nodes:
                i_age = np.digitize(G.nodes[node]['age'],bins) - 1
                i_eth = G.nodes[node]['ethnicity']
                dist[k] = i_age * E + i_eth
                k += 1
            k = 0
            for node in contact_survey_dict:
                i_age = np.digitize(G.nodes[node]['age'],bins) - 1
                i_eth = G.nodes[node]['ethnicity']
                dist_sampled[k] = i_age * E + i_eth
                k += 1
        elif attr == 'as':
            for node in G.nodes:
                i_age = np.digitize(G.nodes[node]['age'],bins) - 1
                i_ses = G.nodes[node]['ses'] - 1
                dist[k] = i_age * S + i_ses
                k += 1
            k = 0
            for node in contact_survey_dict:
                i_age = np.digitize(G.nodes[node]['age'],bins) - 1
                i_ses = G.nodes[node]['ses'] - 1
                dist_sampled[k] = i_age * S + i_ses
                k += 1

        dist_uniq = np.unique(dist, return_counts=True)
        dist_sampled_uniq = np.unique(dist_sampled, return_counts=True)

        for i in range(len(dist_uniq[0])):
            dist_dict[dist_uniq[0][i]] = dist_uniq[1][i]

        for i in range(len(dist_sampled_uniq[0])):
            dist_sampled_dict[dist_sampled_uniq[0][i]] = dist_sampled_uniq[1][i]

        return dist_dict, dist_sampled_dict

# Build contact matrix from simulated contact survey or full contact network
def build_contact_matrix(G, attr, contact_survey=None, groundtruth=True, symmetry=True, per_capita=True):

    # Define number of groups by stratification and stratification names
    bins = np.linspace(0,85,18)
    A = 18

    race_dist_labels = ['White', 'Black', 'Asian', 'AIAN',  'NHPI','Other','Multi']
    R = len(race_dist_labels)

    eth_dist_labels = ['Non-Hispanic','Hispanic']
    E = len(eth_dist_labels)

    ses_dist_labels = ['Lower','Middle', 'Upper']
    S = len(ses_dist_labels)

    # Define number of groups in multiple stratification scenarios (e.g. age + ethnicity)
    A_E = A*E
    A_R = A*R
    A_S = A*S

    group_lens = {'a':A, 'r':R, 'e':E, 's':S, 'ae':A_E, 'ar':A_R, 'as':A_S}

    group_cm = group_lens[attr]
    
    # Get population distribution of survey sample and total population
    full_pop_dist, sample_pop_dist = get_pop_dist(G, attr, contact_survey)

    # If survey not specified, compute contact matrix for whole population; else, compute contact matrix for sampled population
    if not contact_survey:

        cm = []
        cm_c = []
        cm_h = []
        cm_w = []
        cm_s = []

        # Initialise contact matrix structures
        for i in range(group_cm):
            cm.append([])
            cm_c.append([])
            cm_h.append([])
            cm_s.append([])
            cm_w.append([])

            for j in range(group_cm):
                cm[i].append(0)
                cm_c[i].append(0)
                cm_h[i].append(0)
                cm_w[i].append(0)
                cm_s[i].append(0)

        # Iterate through contacts (u->v) and attribute them to contact matrix structures
        for u,v,data in G.edges(data=True):
            if attr == 'a':
                u_group = np.digitize(G.nodes[u]['age'],bins) - 1
                v_group = np.digitize(G.nodes[v]['age'],bins) - 1
            elif attr == 'r':
                u_group = G.nodes[u]['race']
                v_group = G.nodes[v]['race']
            elif attr == 'e':
                u_group = G.nodes[u]['ethnicity']
                v_group = G.nodes[v]['ethnicity']
            elif attr == 's':
                u_group = int(G.nodes[u]['ses']) - 1
                v_group = int(G.nodes[v]['ses']) - 1
            elif attr == 'ar':
                u_group_age = np.digitize(G.nodes[u]['age'],bins) - 1
                v_group_age = np.digitize(G.nodes[v]['age'],bins) - 1

                u_group_race = G.nodes[u]['race']
                v_group_race = G.nodes[v]['race']

                u_group = u_group_age * R + u_group_race
                v_group = v_group_age * R + v_group_race
            elif attr == 'ae':
                u_group_age = np.digitize(G.nodes[u]['age'],bins) - 1
                v_group_age = np.digitize(G.nodes[v]['age'],bins) - 1

                u_group_eth = G.nodes[u]['ethnicity']
                v_group_eth = G.nodes[v]['ethnicity']

                u_group = u_group_age * E + u_group_eth
                v_group = v_group_age * E + v_group_eth
            elif attr == 'as':
                u_group_age = np.digitize(G.nodes[u]['age'],bins) - 1
                v_group_age = np.digitize(G.nodes[v]['age'],bins) - 1

                u_group_ses = int(G.nodes[u]['ses']) - 1
                v_group_ses = int(G.nodes[v]['ses']) - 1

                u_group = u_group_age * S + u_group_ses
                v_group = v_group_age * S + v_group_ses

            cm[v_group][u_group] += 1
            cm[u_group][v_group] += 1

            if data['context'] == 'C_N' or data['context'] == 'C_D':
                cm_c[v_group][u_group] += 1
                cm_c[u_group][v_group] += 1
            elif data['context'] == 'H':
                cm_h[v_group][u_group] += 1
                cm_h[u_group][v_group] += 1
            elif data['context'] == 'W':
                cm_w[v_group][u_group] += 1
                cm_w[u_group][v_group] += 1
            elif data['context'] == 'S':
                cm_s[v_group][u_group] += 1
                cm_s[u_group][v_group] += 1

    else:
        sampled_nodes = pickle.load(open(contact_survey, 'rb'))

        cm = []
        cm_c = []
        cm_h = []
        cm_w = []
        cm_s = []

        # Initialise contact matrix structures
        for i in range(group_cm):
            cm.append([])
            cm_c.append([])
            cm_h.append([])
            cm_s.append([])
            cm_w.append([])

            for j in range(group_cm):
                cm[i].append(0)
                cm_c[i].append(0)
                cm_h[i].append(0)
                cm_w[i].append(0)
                cm_s[i].append(0)

        # Iterate through contacts (u->v) and attribute them to contact matrix structures; if groundtruth, use real attributes
        for n in sampled_nodes:
            for u,v,data in G.edges(n, data=True):
                if attr == 'a':
                    if groundtruth:
                        u_group = np.digitize(G.nodes[u]['age'],bins) - 1
                        v_group = np.digitize(G.nodes[v]['age'],bins) - 1
                    else:
                        u_group = np.digitize(G.nodes[u]['age'],bins) - 1
                        v_group = np.digitize(sampled_nodes[u][v]['recall_age_estimate'],bins) - 1
                elif attr == 'r':
                    if groundtruth:
                        u_group = G.nodes[u]['race']
                        v_group = G.nodes[v]['race']
                    else:
                        u_group = G.nodes[u]['race']
                        v_group = sampled_nodes[u][v]['recall_race_estimate']
                elif attr == 'e':
                    if groundtruth:
                        u_group = G.nodes[u]['ethnicity']
                        v_group = G.nodes[v]['ethnicity']
                    else:
                        u_group = G.nodes[u]['ethnicity']
                        v_group = sampled_nodes[u][v]['recall_eth_estimate']
                elif attr == 's':
                    if groundtruth:
                        u_group = int(G.nodes[u]['ses']) - 1
                        v_group = int(G.nodes[v]['ses']) - 1
                    else:
                        u_group = int(G.nodes[u]['ses']) - 1
                        v_group = int(sampled_nodes[u][v]['recall_ses_estimate']) - 1
                elif attr == 'ar':
                    u_group_age = np.digitize(G.nodes[u]['age'],bins) - 1
                    v_group_age = np.digitize(G.nodes[v]['age'],bins) - 1

                    if groundtruth:
                        u_group_race = G.nodes[u]['race']
                        v_group_race = G.nodes[v]['race']
                    else:
                        u_group_race = G.nodes[u]['race']
                        v_group_race = sampled_nodes[u][v]['recall_race_estimate']

                    u_group = u_group_age * R + u_group_race
                    v_group = v_group_age * R + v_group_race
                elif attr == 'ae':
                    u_group_age = np.digitize(G.nodes[u]['age'],bins) - 1
                    v_group_age = np.digitize(G.nodes[v]['age'],bins) - 1

                    if groundtruth:
                        u_group_eth = G.nodes[u]['ethnicity']
                        v_group_eth = G.nodes[v]['ethnicity']
                    else:
                        u_group_eth = G.nodes[u]['ethnicity']
                        v_group_eth = sampled_nodes[u][v]['recall_eth_estimate']

                    u_group = u_group_age * E + u_group_eth
                    v_group = v_group_age * E + v_group_eth
                elif attr == 'as':
                    u_group_age = np.digitize(G.nodes[u]['age'],bins) - 1
                    v_group_age = np.digitize(G.nodes[v]['age'],bins) - 1

                    if groundtruth:
                        u_group_ses = int(G.nodes[u]['ses']) - 1
                        v_group_ses = int(G.nodes[v]['ses']) - 1
                    else:
                        u_group_ses = int(G.nodes[u]['ses']) - 1
                        v_group_ses = int(sampled_nodes[u][v]['recall_ses_estimate']) - 1

                    u_group = u_group_age * S + u_group_ses
                    v_group = v_group_age * S + v_group_ses

                cm[u_group][v_group] += 1

                if data['context'] == 'C_N' or data['context'] == 'C_D':
                    cm_c[u_group][v_group] += 1
                elif data['context'] == 'H':
                    cm_h[u_group][v_group] += 1
                elif data['context'] == 'W':
                    cm_w[u_group][v_group] += 1
                elif data['context'] == 'S':
                    cm_s[u_group][v_group] += 1

    # Compute contact rates by dividing by sample population size
    for i in range(group_cm):
        if i in sample_pop_dist:
            for j in range(group_cm):
                cm[i][j] = cm[i][j] / sample_pop_dist[i]
                cm_c[i][j] = cm_c[i][j] / sample_pop_dist[i]
                cm_h[i][j] = cm_h[i][j] / sample_pop_dist[i]
                cm_w[i][j] = cm_w[i][j] / sample_pop_dist[i]
                cm_s[i][j] = cm_s[i][j] / sample_pop_dist[i]

    # Adjust contact matrix for symmetry condition
    if symmetry:
        cm_temp = copy.deepcopy(cm)
        cm_c_temp = copy.deepcopy(cm_c)
        cm_h_temp = copy.deepcopy(cm_h)
        cm_w_temp = copy.deepcopy(cm_w)
        cm_s_temp = copy.deepcopy(cm_s)

        for i in range(group_cm):
            for j in range(group_cm):
                if i in full_pop_dist and j in full_pop_dist:
                    cm[i][j] = (1 / full_pop_dist[i]) * ((cm_temp[i][j] * full_pop_dist[i]) + (cm_temp[j][i] * full_pop_dist[j])) / 2
                    cm_c[i][j] = (1 / full_pop_dist[i]) * ((cm_c_temp[i][j] * full_pop_dist[i]) + (cm_c_temp[j][i] * full_pop_dist[j])) / 2
                    cm_h[i][j] = (1 / full_pop_dist[i]) * ((cm_h_temp[i][j] * full_pop_dist[i]) + (cm_h_temp[j][i] * full_pop_dist[j])) / 2
                    cm_w[i][j] = (1 / full_pop_dist[i]) * ((cm_w_temp[i][j] * full_pop_dist[i]) + (cm_w_temp[j][i] * full_pop_dist[j])) / 2
                    cm_s[i][j] = (1 / full_pop_dist[i]) * ((cm_s_temp[i][j] * full_pop_dist[i]) + (cm_s_temp[j][i] * full_pop_dist[j])) / 2

    # Adjust contact matrox for per-capita contact rates
    if per_capita:
        for i in range(group_cm):
            for j in range(group_cm):
                if i in full_pop_dist and j in full_pop_dist:
                    cm[i][j] = cm[i][j] / full_pop_dist[j]
                    cm_c[i][j] = cm_c[i][j] / full_pop_dist[j]
                    cm_h[i][j] = cm_h[i][j] / full_pop_dist[j]
                    cm_w[i][j] = cm_w[i][j] / full_pop_dist[j]
                    cm_s[i][j] = cm_s[i][j] / full_pop_dist[j]
            

    return [cm, cm_c, cm_h, cm_w, cm_s]

if __name__ == '__main__':
    
    # Load input contact network; assign to G
    input_network =  'NM_network_v3'
    G = pickle.load(open('../Data/Contact network/' + input_network + '.pickle', 'rb'))

    # Define date of survey simulation (i.e. computer time date when survey_simulator.py was run)
    date = '2025-09-18'

    # Define experiment name
    experiment = 'supp_exp_context'

    # Define context labels for contact matrices
    cm_out_labels = ['Overall','Community','Household','Workplace','School']

    # Define attributes for stratifying population (age, ethnicity, race, SES/income)
    attr = ['a','e','r','s']

    # Define matrix indices of subpopulations of interest
    target_groups_attr = {'a':[13,14,15,16,17],'e':[1],'r':[1,2,3,4,5,6],'s':[0],
                          'ae':[h for h in range(36) if h % 2 == 1],
                          'ar':[h for h in range(126) if h % 7 != 0],
                          'as':[h for h in range(54) if h % 3 == 0]}


    # FULL POPULATION GROUNDTRUTH MATRICES
    if experiment == 'groundtruth':
        for a in attr:
            cm_out = build_contact_matrix(G = G, attr = a, contact_survey=None, 
                                                        groundtruth=True, symmetry=False, per_capita=False)
        
            for i in range(len(cm_out)):
                np.save('../Data/Contact matrices/' + input_network + '__full_pop__'  + a + '__' + cm_out_labels[i] + '.npy',arr=cm_out[i])

            cm_out = build_contact_matrix(G = G, attr = a, contact_survey=None, 
                                                        groundtruth=True, symmetry=True, per_capita=True)
        
            for i in range(len(cm_out)):
                np.save('../Data/Contact matrices/' + input_network + '__full_pop__processed__'  + a + '__' + cm_out_labels[i] + '.npy',arr=cm_out[i])

    # EXPERIMENT 1: Age
    if experiment == 'exp1':
        attr = ['a']
        input_survey = glob.glob('../Data/Contact survey data/' + input_network + '__exp1__survey*' + date + '.pickle')
        input_survey = [h.split('__exp1__')[1][:-7] for h in input_survey]
        
        for a in attr:
            for survey_in in input_survey:

                cm_out = build_contact_matrix(G = G, attr = a, contact_survey = '../Data/Contact survey data/' + input_network + '__exp1__' + survey_in + '.pickle', 
                                                            groundtruth=True, symmetry=False, per_capita=False, target_groups=target_groups_attr[a])
                np.save('../Data/Contact matrices/' + input_network + '__exp1__' + survey_in[7:] + '__gt__raw__' + a + '__' + cm_out_labels[0] + '.npy',arr=cm_out[0])
                np.save('../Data/Contact matrices/' + input_network + '__exp1__' + survey_in[7:] + '__gt__raw__' + a + '__Binary.npy',arr=cm_out[5])

                cm_out = build_contact_matrix(G = G, attr = a, contact_survey = '../Data/Contact survey data/' + input_network + '__exp1__' + survey_in + '.pickle', 
                                                            groundtruth=True, symmetry=True, per_capita=True, target_groups=target_groups_attr[a])
                np.save('../Data/Contact matrices/' + input_network + '__exp1__' + survey_in[7:] + '__gt__processed__' + a + '__' + cm_out_labels[0] + '.npy',arr=cm_out[0])
                np.save('../Data/Contact matrices/' + input_network + '__exp1__' + survey_in[7:] + '__gt__processed__' + a + '__Binary.npy',arr=cm_out[5])

                cm_out = build_contact_matrix(G = G, attr = a, contact_survey = '../Data/Contact survey data/' + input_network + '__exp1__' + survey_in + '.pickle', 
                                                            groundtruth=False, symmetry=False, per_capita=False, target_groups=target_groups_attr[a])
                np.save('../Data/Contact matrices/' + input_network + '__exp1__' + survey_in[7:] + '__biased__raw__' + a + '__' + cm_out_labels[0] + '.npy',arr=cm_out[0])
                np.save('../Data/Contact matrices/' + input_network + '__exp1__' + survey_in[7:] + '__biased__raw__' + a + '__Binary.npy',arr=cm_out[5])

                cm_out = build_contact_matrix(G = G, attr = a, contact_survey = '../Data/Contact survey data/' + input_network + '__exp1__' + survey_in + '.pickle', 
                                                            groundtruth=False, symmetry=True, per_capita=True, target_groups=target_groups_attr[a])
                np.save('../Data/Contact matrices/' + input_network + '__exp1__' + survey_in[7:] + '__biased__processed__' + a + '__' + cm_out_labels[0] + '.npy',arr=cm_out[0])
                np.save('../Data/Contact matrices/' + input_network + '__exp1__' + survey_in[7:] + '__biased__processed__' + a + '__Binary.npy',arr=cm_out[5])

    # EXPERIMENT 2: Race
    if experiment == 'exp2':
        attr = ['r','ar']
        input_survey = glob.glob('../Data/Contact survey data/'+ input_network + '__' + experiment + '__survey_tract*' + date + '.pickle')
        input_survey = [h.split('__' + experiment + '__')[1][:-7] for h in input_survey]
        
        for a in attr:
            for survey_in in input_survey:
                cm_out = build_contact_matrix(G = G, attr = a, contact_survey = '../Data/Contact survey data/' + input_network + '__' + experiment + '__' + survey_in + '.pickle', 
                                                            groundtruth=True, symmetry=False, per_capita=False, target_groups=target_groups_attr[a])
                np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + survey_in[7:] + '__gt__raw__' + a + '__' + cm_out_labels[0] + '.npy',arr=cm_out[0])
                np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + survey_in[7:] + '__gt__raw__' + a + '__' + cm_out_labels[1] + '.npy',arr=cm_out[1])
                np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + survey_in[7:] + '__gt__raw__' + a + '__Binary.npy',arr=cm_out[5])

                cm_out = build_contact_matrix(G = G, attr = a, contact_survey = '../Data/Contact survey data/' + input_network + '__' + experiment + '__' + survey_in + '.pickle', 
                                                            groundtruth=True, symmetry=True, per_capita=True, target_groups=target_groups_attr[a])
                np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + survey_in[7:] + '__gt__processed__' + a + '__' + cm_out_labels[0] + '.npy',arr=cm_out[0])
                np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + survey_in[7:] + '__gt__processed__' + a + '__' + cm_out_labels[1] + '.npy',arr=cm_out[1])
                np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + survey_in[7:] + '__gt__processed__' + a + '__Binary.npy',arr=cm_out[5])

                cm_out = build_contact_matrix(G = G, attr = a, contact_survey = '../Data/Contact survey data/' + input_network + '__' + experiment + '__' + survey_in + '.pickle', 
                                                            groundtruth=False, symmetry=False, per_capita=False, target_groups=target_groups_attr[a])
                np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + survey_in[7:] + '__biased__raw__' + a + '__' + cm_out_labels[0] + '.npy',arr=cm_out[0])
                np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + survey_in[7:] + '__biased__raw__' + a + '__' + cm_out_labels[1] + '.npy',arr=cm_out[1])
                np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + survey_in[7:] + '__biased__raw__' + a + '__Binary.npy',arr=cm_out[5])

                cm_out = build_contact_matrix(G = G, attr = a, contact_survey = '../Data/Contact survey data/' + input_network + '__' + experiment + '__' + survey_in + '.pickle', 
                                                            groundtruth=False, symmetry=True, per_capita=True, target_groups=target_groups_attr[a])
                np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + survey_in[7:] + '__biased__processed__' + a + '__' + cm_out_labels[0] + '.npy',arr=cm_out[0])
                np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + survey_in[7:] + '__biased__processed__' + a + '__' + cm_out_labels[1] + '.npy',arr=cm_out[1])
                np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + survey_in[7:] + '__biased__processed__' + a + '__Binary.npy',arr=cm_out[5])

    
     # EXPERIMENT SAMPLE SIZE: Age, race

    # SUPPLEMENTRAL EXPERIMENTS
    if experiment == 'supp_exp_context' or experiment == 'supp_wg':
        attr = ['r','ar']
        input_survey = glob.glob('../Data/Contact survey data/'+ input_network + '__' + experiment + '__*' + date + '.pickle')
        input_survey = [h.split('__' + experiment + '__')[1][:-7] for h in input_survey]
        
        for a in attr:
            for survey_in in input_survey:

                cm_out = build_contact_matrix(G = G, attr = a, contact_survey = '../Data/Contact survey data/' + input_network + '__' + experiment + '__' + survey_in + '.pickle', 
                                                            groundtruth=True, symmetry=False, per_capita=False, target_groups=None)
                np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + survey_in[7:] + '__gt__raw__' + a + '__' + cm_out_labels[0] + '.npy',arr=cm_out[0])

                cm_out = build_contact_matrix(G = G, attr = a, contact_survey = '../Data/Contact survey data/' + input_network + '__' + experiment + '__' + survey_in + '.pickle', 
                                                            groundtruth=True, symmetry=True, per_capita=True, target_groups=None)
                np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + survey_in[7:] + '__gt__processed__' + a + '__' + cm_out_labels[0] + '.npy',arr=cm_out[0])
                
                cm_out = build_contact_matrix(G = G, attr = a, contact_survey = '../Data/Contact survey data/' + input_network + '__' + experiment + '__' + survey_in + '.pickle', 
                                                            groundtruth=False, symmetry=False, per_capita=False, target_groups=None)
                np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + survey_in[7:] + '__biased__raw__' + a + '__' + cm_out_labels[0] + '.npy',arr=cm_out[0])

                cm_out = build_contact_matrix(G = G, attr = a, contact_survey = '../Data/Contact survey data/' + input_network + '__' + experiment + '__' + survey_in + '.pickle', 
                                                            groundtruth=False, symmetry=True, per_capita=True, target_groups=None)
                np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + survey_in[7:] + '__biased__processed__' + a + '__' + cm_out_labels[0] + '.npy',arr=cm_out[0])

    
    