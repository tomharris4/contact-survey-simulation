import datetime
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import copy

def sample_age(u,v,context,age_attr):

    mean = v['age'] + (age_attr['sigma_map'][context] * age_attr['b_age'] * age_attr['bias_mean'][v['age']])

    variance = age_attr['sigma_map'][context] * age_attr['h']

    if mean <= 0:
        mean = 0.05
    
    if variance == 0:
        return mean
    
    scale = variance / mean
    shape = mean / scale
    
    return int(np.floor(np.random.gamma(shape=shape,scale=scale)))

def sample_race(u,v,context,race_attr):

    l_ind = int(u['race'] != 0)
    r_ind = int(v['race'] != 0)

    if v['race'] != u['race']:
        ind = 0 
    else:
        ind = 1

    if ind == 1:
        if race_attr['pop_race'] == 'state':
            race_weights = [race_attr['l'][l_ind] * race_attr['pop_prop_race'][h] * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context] for h in range(7)]
            race_weights[v['race']] = 1 - (race_attr['l'][l_ind] - (race_attr['l'][l_ind] * race_attr['pop_prop_race'][v['race']])) * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context]
        elif race_attr['pop_race'] == 'tract':
            if context in ['H','C_N']:
                race_weights = [race_attr['l'][l_ind] * race_attr['pop_prop_race']['N_' + str(v['night_tract'])[:-1]][h] * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context] for h in range(7)]
                race_weights[v['race']] = 1 - (race_attr['l'][l_ind] - (race_attr['l'][l_ind] * race_attr['pop_prop_race']['N_' + str(v['night_tract'])[:-1]][v['race']])) * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context]
            else:
                race_weights = [race_attr['l'][l_ind] * race_attr['pop_prop_race']['D_' + str(v['day_tract'])[:-1]][h] * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context] for h in range(7)]
                race_weights[v['race']] = 1 - (race_attr['l'][l_ind] - (race_attr['l'][l_ind] * race_attr['pop_prop_race']['D_' + str(v['day_tract'])[:-1]][v['race']])) * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context]
        elif race_attr['pop_race'] == 'dominant':
            if r_ind:
                race_weights = [0,0,0,0,0,0,0]
                race_weights[v['race']] = 1 - race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context]
                race_weights[0] = race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context]
            else:
                race_weights = [1,0,0,0,0,0,0]
        else:
            race_weights = [race_attr['l'][l_ind] * (1/7) * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context] for h in range(7)]
            race_weights[v['race']] = 1 - (race_attr['l'][l_ind] - (race_attr['l'][l_ind] * (1/7))) * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context]
    else:
        if race_attr['pop_race'] == 'state':
            race_weights = [race_attr['l'][l_ind] * race_attr['pop_prop_race'][h] * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context] for h in range(7)]
            race_weights[v['race']] = 1 - (1 - (race_attr['l'][l_ind] * race_attr['pop_prop_race'][v['race']])) * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context]
            race_weights[u['race']] = (1 - race_attr['l'][l_ind] + (race_attr['l'][l_ind] * race_attr['pop_prop_race'][u['race']])) * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context]
        elif race_attr['pop_race'] == 'tract':
            if context in ['H','C_N']:
                race_weights = [race_attr['l'][l_ind] * race_attr['pop_prop_race']['N_' + str(v['night_tract'])[:-1]][h] * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context] for h in range(7)]
                race_weights[v['race']] = 1 - (1 - (race_attr['l'][l_ind] * race_attr['pop_prop_race']['N_' + str(v['night_tract'])[:-1]][v['race']])) * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context]
                race_weights[u['race']] = (1 - race_attr['l'][l_ind] + (race_attr['l'][l_ind] * race_attr['pop_prop_race']['N_' + str(v['night_tract'])[:-1]][u['race']])) * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context]
            else:
                race_weights = [race_attr['l'][l_ind] * race_attr['pop_prop_race']['D_' + str(v['day_tract'])[:-1]][h] * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context] for h in range(7)]
                race_weights[v['race']] = 1 - (1 - (race_attr['l'][l_ind] * race_attr['pop_prop_race']['D_' + str(v['day_tract'])[:-1]][v['race']])) * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context]
                race_weights[u['race']] = (1 - race_attr['l'][l_ind] + (race_attr['l'][l_ind] * race_attr['pop_prop_race']['D_' + str(v['day_tract'])[:-1]][u['race']])) * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context]
        elif race_attr['pop_race'] == 'dominant':
            if r_ind:
                race_weights = [0,0,0,0,0,0,0]
                race_weights[v['race']] = 1 - race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context]
                race_weights[0] = race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context]
            else:
                race_weights = [1,0,0,0,0,0,0]
        else:
            race_weights = [race_attr['l'][l_ind] * (1/7) * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context] for h in range(7)]
            race_weights[v['race']] = 1 - (1 - (race_attr['l'][l_ind] * (1/7))) * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context]
            race_weights[u['race']] = (1 - race_attr['l'][l_ind] + (race_attr['l'][l_ind] * (1/7))) * race_attr['anchors_race'][r_ind] * race_attr['sigma_map'][context]

    return rng.choice(a=range(0,7), p=race_weights)

def sample_ethnicity(u,v,context,eth_attr):
    
    # Check if individual being contacted is Hispanic - if so, use r_3 error weighting 
    if v['ethnicity'] == 1:
        ind = 0
    else:
        ind = 1

    if eth_attr['pop_eth'] == 'state':
        eth_weights = [eth_attr['pop_prop_eth'][h] * eth_attr['anchors_eth'][ind] * eth_attr['sigma_map'][context] for h in range(2)]
        eth_weights[v['ethnicity']] = 1 - (1 - eth_attr['pop_prop_eth'][v['ethnicity']]) * eth_attr['anchors_eth'][ind] * eth_attr['sigma_map'][context]
    elif eth_attr['pop_eth'] == 'tract':
        if context in ['H','C_N']:
            eth_weights = [eth_attr['pop_prop_eth']['N_' + str(v['night_tract'])[:-1]][h] * eth_attr['anchors_eth'][ind] * eth_attr['sigma_map'][context] for h in range(2)]
            eth_weights[v['ethnicity']] = 1 - (1 - eth_attr['pop_prop_eth']['N_' + str(v['night_tract'])[:-1]][v['ethnicity']]) * eth_attr['anchors_eth'][ind] * eth_attr['sigma_map'][context]
        else:
            eth_weights = [eth_attr['pop_prop_eth']['D_' + str(v['day_tract'])[:-1]][h] * eth_attr['anchors_eth'][ind] * eth_attr['sigma_map'][context] for h in range(2)]
            eth_weights[v['ethnicity']] = 1 - (1 - eth_attr['pop_prop_eth']['D_' + str(v['day_tract'])[:-1]][v['ethnicity']]) * eth_attr['anchors_eth'][ind] * eth_attr['sigma_map'][context]
    else:
        eth_weights = [(1/2) * eth_attr['anchors_eth'][ind] * eth_attr['sigma_map'][context] for h in range(2)]
        eth_weights[v['ethnicity']] = 1 - (1/2) * eth_attr['anchors_eth'][ind] * eth_attr['sigma_map'][context]
    

    return rng.choice(a=range(0,2), p=eth_weights)

def sample_ses(u,v,context,ses_attr):

    if ses_attr['pop_ses'] == 'state':
        ses_weights = [ses_attr['pop_prop_ses'][h+1] * ses_attr['anchor_ses'] * ses_attr['sigma_map'][context] for h in range(3)]
        ses_weights[int(v['ses'])-1] = 1 - (1 - ses_attr['pop_prop_ses'][int(v['ses'])]) * ses_attr['anchor_ses'] * ses_attr['sigma_map'][context]
    elif ses_attr['pop_ses'] == 'tract':
        if context in ['H','C_N']:
            ses_weights = [ses_attr['pop_prop_ses']['N_' + str(v['night_tract'])[:-1]][h+1] * ses_attr['anchor_ses'] * ses_attr['sigma_map'][context] for h in range(3)]
            ses_weights[int(v['ses'])-1] = 1 - (1 - ses_attr['pop_prop_ses']['N_' + str(v['night_tract'])[:-1]][int(v['ses'])]) * ses_attr['anchor_ses'] * ses_attr['sigma_map'][context]
        else:
            ses_weights = [ses_attr['pop_prop_ses']['D_' + str(v['day_tract'])[:-1]][h+1] * ses_attr['anchor_ses'] * ses_attr['sigma_map'][context] for h in range(3)]
            ses_weights[int(v['ses'])-1] = 1 - (1 - ses_attr['pop_prop_ses']['D_' + str(v['day_tract'])[:-1]][int(v['ses'])]) * ses_attr['anchor_ses'] * ses_attr['sigma_map'][context]
    else:
        ses_weights = [(1/3) * ses_attr['anchor_ses'] * ses_attr['sigma_map'][context] for h in range(3)]
        ses_weights[int(v['ses'])-1] = 1 - (2/3) * ses_attr['anchor_ses'] * ses_attr['sigma_map'][context]
    
    return rng.choice(a=range(1,4), p=ses_weights)

def add_survey_attributes(G,e):

    recall_init = (-1,-1)

    G.edges()[e]['recall_age_estimate'] = recall_init
    G.edges()[e]['recall_age_var_estimate'] = recall_init

    G.edges()[e]['recall_race_estimate'] = recall_init

    G.edges()[e]['recall_eth_estimate'] = recall_init
    G.edges()[e]['recall_eth_var_estimate'] = recall_init

    G.edges()[e]['recall_ses_estimate'] = recall_init

def get_race_lookup(geo,G,filename=None):

    if geo == 'state':
        agents_race = np.unique(list(nx.get_node_attributes(G,'race').values()),return_counts=True)
        agents_race = dict(zip(agents_race[0],agents_race[1]))

        pop_prop_race = {}

        for i in range(7):
            pop_prop_race[i] = agents_race[i] / len(G.nodes())

    elif geo == 'tract':
        if filename:
            return pickle.load(open(filename, 'rb'))
        
        agents_tract_N = np.unique(['N_' + str(h[1]['night_tract'])[:-1] for h in G.nodes(data=True)],return_counts=True)
        agents_tract_N = dict(zip(agents_tract_N[0],agents_tract_N[1]))

        agents_tract_D = np.unique(['D_' + str(h[1]['day_tract'])[:-1] for h in G.nodes(data=True)],return_counts=True)
        agents_tract_D = dict(zip(agents_tract_D[0],agents_tract_D[1]))
        
        agents_race_N = np.unique([('N_' + str(h[1]['night_tract'])[:-1] + str(h[1]['race'])) for h in G.nodes(data=True)],return_counts=True)
        agents_race_N = dict(zip(agents_race_N[0],agents_race_N[1]))

        agents_race_D = np.unique([('D_' + str(h[1]['day_tract'])[:-1] + str(h[1]['race'])) for h in G.nodes(data=True)],return_counts=True)
        agents_race_D = dict(zip(agents_race_D[0],agents_race_D[1]))

        pop_prop_race = {}

        for i in agents_tract_N:
            pop_prop_race[i] = {}
            for j in range(0,7):
                if (i + str(j)) in agents_race_N:
                    pop_prop_race[i][j] = agents_race_N[(i + str(j))] / agents_tract_N[i]
                else:
                    pop_prop_race[i][j] = 0

        for i in agents_tract_D:
            pop_prop_race[i] = {}
            for j in range(0,7):
                if (i + str(j)) in agents_race_D:
                    pop_prop_race[i][j] = agents_race_D[(i + str(j))] / agents_tract_D[i]
                else: 
                    pop_prop_race[i][j] = 0


        pickle.dump(pop_prop_race,open('../Data/Misc/Tract_race_demographics.pickle', 'wb' ))

    else:
        pop_prop_race = None

    return pop_prop_race

def get_eth_lookup(geo,G,filename=None):
    if geo == 'state':
        agents_eth = np.unique(list(nx.get_node_attributes(G,'ethnicity').values()),return_counts=True)
        agents_eth = dict(zip(agents_eth[0],agents_eth[1]))
        pop_prop_eth = {}

        for i in range(2):
            pop_prop_eth[i] = agents_eth[i] / len(G.nodes())

    elif geo == 'tract':
        if filename:
            return pickle.load(open(filename, 'rb'))
        
        agents_tract_N = np.unique(['N_' + str(h[1]['night_tract'])[:-1] for h in G.nodes(data=True)],return_counts=True)
        agents_tract_N = dict(zip(agents_tract_N[0],agents_tract_N[1]))

        agents_tract_D = np.unique(['D_' + str(h[1]['day_tract'])[:-1] for h in G.nodes(data=True)],return_counts=True)
        agents_tract_D = dict(zip(agents_tract_D[0],agents_tract_D[1]))
        
        agents_eth_N = np.unique([('N_' + str(h[1]['night_tract'])[:-1] + str(h[1]['ethnicity'])) for h in G.nodes(data=True)],return_counts=True)
        agents_eth_N = dict(zip(agents_eth_N[0],agents_eth_N[1]))

        agents_eth_D = np.unique([('D_' + str(h[1]['day_tract'])[:-1] + str(h[1]['ethnicity'])) for h in G.nodes(data=True)],return_counts=True)
        agents_eth_D = dict(zip(agents_eth_D[0],agents_eth_D[1]))

        pop_prop_eth = {}

        for i in agents_tract_N:
            pop_prop_eth[i] = {}
            for j in range(0,2):
                if (i + str(j)) in agents_eth_N:
                    pop_prop_eth[i][j] = agents_eth_N[(i + str(j))] / agents_tract_N[i]
                else:
                    pop_prop_eth[i][j] = 0

        for i in agents_tract_D:
            pop_prop_eth[i] = {}
            for j in range(0,2):
                if (i + str(j)) in agents_eth_D:
                    pop_prop_eth[i][j] = agents_eth_D[(i + str(j))] / agents_tract_D[i]
                else:
                    pop_prop_eth[i][j] = 0


        pickle.dump(pop_prop_eth,open('../Data/Misc/Tract_eth_demographics.pickle', 'wb' ))

    else:
        pop_prop_eth = None

    return pop_prop_eth

def get_ses_lookup(geo,G,filename=None):
    if geo == 'state':
        agents_ses = np.unique(list(nx.get_node_attributes(G,'ses').values()),return_counts=True)
        agents_ses = dict(zip(agents_ses[0],agents_ses[1]))
        pop_prop_ses = {}

        for i in range(1,4):
            pop_prop_ses[i] = agents_ses[i] / len(G.nodes())

    elif geo == 'tract':
        if filename:
            return pickle.load(open(filename, 'rb'))
        
        agents_tract_N = np.unique(['N_' + str(h[1]['night_tract'])[:-1] for h in G.nodes(data=True)],return_counts=True)
        agents_tract_N = dict(zip(agents_tract_N[0],agents_tract_N[1]))

        agents_tract_D = np.unique(['D_' + str(h[1]['day_tract'])[:-1] for h in G.nodes(data=True)],return_counts=True)
        agents_tract_D = dict(zip(agents_tract_D[0],agents_tract_D[1]))
        
        agents_ses_N = np.unique([('N_' + str(h[1]['night_tract'])[:-1] + str(int(h[1]['ses']))) for h in G.nodes(data=True)],return_counts=True)
        agents_ses_N = dict(zip(agents_ses_N[0],agents_ses_N[1]))

        agents_ses_D = np.unique([('D_' + str(h[1]['day_tract'])[:-1] + str(int(h[1]['ses']))) for h in G.nodes(data=True)],return_counts=True)
        agents_ses_D = dict(zip(agents_ses_D[0],agents_ses_D[1]))

        pop_prop_ses = {}

        for i in agents_tract_N:
            pop_prop_ses[i] = {}
            for j in range(1,4):
                if (i + str(j)) in agents_ses_N:
                    pop_prop_ses[i][j] = agents_ses_N[(i + str(j))] / agents_tract_N[i]
                else:
                    pop_prop_ses[i][j] = 0

        for i in agents_tract_D:
            pop_prop_ses[i] = {}
            for j in range(1,4):
                if (i + str(j)) in agents_ses_D:
                    pop_prop_ses[i][j] = agents_ses_D[(i + str(j))] / agents_tract_D[i]
                else:
                    pop_prop_ses[i][j] = 0

        pickle.dump(pop_prop_ses,open('../Data/Misc/Tract_ses_demographics.pickle', 'wb' ))

    else:
        pop_prop_ses = None

    return pop_prop_ses

def get_age_bias_mean():

    bias_mean = []

    for i in np.linspace(0, 100, 100):
        bias_mean.append(((-0.00413*i**2 + 1.216*i) - i))

    return bias_mean

def get_age_distribution(G):
    age_dist_network_binned = []

    for node in G.nodes:
        a = np.digitize(G.nodes[node]['age'],np.linspace(0,85,18)) - 1
        age_dist_network_binned.append(a)
        

    age_dist_dict = {}

    for i in range(len(np.unique(age_dist_network_binned, return_counts=True)[0])):
        age_dist_dict[np.unique(age_dist_network_binned, return_counts=True)[0][i]] = np.unique(age_dist_network_binned, return_counts=True)[1][i] / len(G.nodes())

    return age_dist_dict
    
def simulate_survey(G, edge_context_dic, age_dist, age_attr, race_attr, eth_attr, ses_attr, sample_size=10000, f=[0,0], g=[0,0]):

    # sampled_nodes = np.empty(sample_size,dtype=np.uint32)

    # for a in age_dist:
    #     age_dist[a] = age_dist[a] * sample_size

    # age_tally = {}

    G_out = {}

    node_list = list(G.nodes())

    sampled_nodes = rng.choice(node_list,size=sample_size,replace=False)

    for node in sampled_nodes:
        G_out[node] = {}
        # for e in G.edges(node):
        #     add_survey_attributes(G,e)

    # k = 0

    # while k < sample_size:
    #     node = np.random.choice(node_list)
    #     node_group = np.digitize(G.nodes[node]['age'],np.linspace(0,85,18)) - 1

    #     if node_group not in age_tally:
    #         age_tally[node_group] = 1
    #         sampled_nodes[k] = node
    #         for e in G.edges(node):
    #             add_survey_attributes(G,e)
    #         k += 1
    #     else:
    #         if age_tally[node_group] < age_dist[node_group]:
    #             if node not in sampled_nodes:
    #                 age_tally[node_group] += 1
    #                 sampled_nodes[k] = node
    #                 for e in G.edges(node):
    #                     add_survey_attributes(G,e)
    #                 k += 1

    for n in sampled_nodes:
        for u,v in G.edges(n):
            if (u,v) in edge_context_dic:
                context = edge_context_dic[(u,v)]
            else:
                context = edge_context_dic[(v,u)]
            
            e_a = sample_age(G.nodes[u],G.nodes[v],context,age_attr)
            if abs(e_a - G.nodes[v]['age']) > 3 and rng.random() < g[1]:
                e_a_var = 1
            elif abs(e_a - G.nodes[v]['age']) <= 3 and rng.random() < g[0]:
                e_a_var = 1
            else:
                e_a_var = 0
            e_r = sample_race(G.nodes[u],G.nodes[v],context,race_attr)
            e_e = sample_ethnicity(G.nodes[u],G.nodes[v],context,eth_attr)
            if e_e != G.nodes[v]['ethnicity'] and rng.random() < f[1]:
                e_e_var = 1
            elif e_e == G.nodes[v]['ethnicity'] and rng.random() < f[0]:
                e_e_var = 1
            else:
                e_e_var = 0
            e_s = sample_ses(G.nodes[u],G.nodes[v],context,ses_attr)

            G_out[u][v] = {}
            
            G_out[u][v]['recall_age_estimate'] = e_a
            G_out[u][v]['recall_age_var_estimate'] = e_a_var
            G_out[u][v]['recall_race_estimate'] = e_r
            G_out[u][v]['recall_eth_estimate'] = e_e
            G_out[u][v]['recall_eth_var_estimate'] = e_e_var
            G_out[u][v]['recall_ses_estimate'] = e_s

    return G_out#, sampled_nodes

    
if __name__ == '__main__':

    rng = np.random.RandomState(10)

    input_network = 'NM_network_v3'

    G = pickle.load(open('../Data/Contact network/' + input_network + '.pickle', 'rb'))
    edge_context_dic = nx.get_edge_attributes(G,'context')

    bias_mean = get_age_bias_mean()

    age_dist_dict = get_age_distribution(G)

    experiment = 'exp2b'

    print(datetime.datetime.now(), 'Initialisation complete')

    # b_age = 2.564
    # sigma_map = {'H':0,'W':0.44,'S':0.25,'C_N':0.39,'C_D':0.39}
    # anchors_race = [1.282]
    # anchors_eth = [1.282,0.256]
    # anchor_ses = 1.718

    # Experiment 1
    if experiment == 'exp1':
        h = 10 
        b_age = np.linspace(0, 1.6*2.564, 9)
        sigma_map = {'H':0,'W':0.44,'S':0.25,'C_N':0.39,'C_D':0.39}
        anchors_race = np.linspace(0,0.8*2.564,9)
        anchors_eth = np.linspace(0,0.8*2.564,9)
        anchor_ses = np.linspace(0,0.8*2.564,9)
        l = 1
        geo_level_weighting = 'dominant'
        pop_prop_race = get_race_lookup(geo_level_weighting,G)
        pop_prop_eth = get_eth_lookup(geo_level_weighting,G)
        pop_prop_ses = get_ses_lookup(geo_level_weighting,G)

        for j in range(9):
            for i in range(10):
                g_out = simulate_survey(G,sample_size=10000, age_dist=age_dist_dict, edge_context_dic = edge_context_dic,
                                age_attr={'sigma_map':sigma_map, 'h':h, 'b_age':b_age[j], 'bias_mean':bias_mean},
                                race_attr={'sigma_map':sigma_map,'anchors_race':[0,anchors_race[j]], 'pop_race':geo_level_weighting, 'pop_prop_race': pop_prop_race, 'l':[l,l]},
                                eth_attr={'sigma_map':sigma_map,'anchors_eth':[anchors_eth[j],0], 'pop_eth':geo_level_weighting, 'pop_prop_eth': pop_prop_eth},
                                ses_attr={'sigma_map':sigma_map,'anchor_ses':anchor_ses[j], 'pop_ses':geo_level_weighting, 'pop_prop_ses':pop_prop_ses},
                                f = [0,0], g = [0,0])
                
                pickle.dump(g_out,open('../Data/Contact survey data/' + input_network + '__' + experiment + '__' + 'survey_' + str(j) + '_' +
                                        str(i) + '_' + str(datetime.date.today()) + '.pickle', 'wb' ))

                if i % 5 == 0:
                    print(datetime.datetime.now(), 'Survey complete: ', i)


    # Experiment 2
    # if experiment == 'exp2a':
    #     h = 10
    #     b_age = 2.564
    #     sigma_map = {'H':0,'W':0.44,'S':0.25,'C_N':0.39,'C_D':0.39}
    #     anchors_race = [0,0.8*2.564]
    #     anchors_eth = 0.8*2.564
    #     anchor_ses = 0.8*2.564
    #     l = [1]#np.linspace(0,1,10)
    #     geo_level_weighting = 'state'
    #     pop_prop_race = get_race_lookup(geo_level_weighting,G)
    #     pop_prop_eth = get_eth_lookup(geo_level_weighting,G)
    #     pop_prop_ses = get_ses_lookup(geo_level_weighting,G)

    #     for j in range(1):
    #         for i in range(10):
    #             g_out = simulate_survey(G,sample_size=10000, age_dist=age_dist_dict, edge_context_dic = edge_context_dic,
    #                             age_attr={'sigma_map':sigma_map, 'h':h, 'b_age':b_age, 'bias_mean':bias_mean},
    #                             race_attr={'sigma_map':sigma_map,'anchors_race':anchors_race, 'pop_race':geo_level_weighting, 'pop_prop_race': pop_prop_race, 'l':[0,l[j]]},
    #                             eth_attr={'sigma_map':sigma_map,'anchors_eth':[anchors_eth,anchors_eth], 'pop_eth':geo_level_weighting, 'pop_prop_eth': pop_prop_eth},
    #                             ses_attr={'sigma_map':sigma_map,'anchor_ses':anchor_ses, 'pop_ses':geo_level_weighting, 'pop_prop_ses':pop_prop_ses},
    #                             f = [0,0], g = [0,0])
                
    #             pickle.dump(g_out,open('../Data/Contact survey data/' + input_network + '__' + experiment + '__' + 'survey_' + str(j) + '_' +
    #                                     str(i) + '_' + str(datetime.date.today()) + '.pickle', 'wb' ))

    #             if i % 5 == 0:
    #                 print(datetime.datetime.now(), 'Survey complete: ', i)


    if experiment == 'exp2b':
        h = 10
        b_age = 2.564
        sigma_map = {'H':0,'W':0.44,'S':0.25,'C_N':0.39,'C_D':0.39}
        anchors_race = [0.8*2.564,0.8*2.564]
        anchors_eth = 0.8*2.564
        anchor_ses = 0.8*2.564
        l = np.linspace(0,1,3)
        geo_level_weighting = 'state'
        pop_prop_race = get_race_lookup(geo_level_weighting,G)
        pop_prop_eth = get_eth_lookup(geo_level_weighting,G)
        pop_prop_ses = get_ses_lookup(geo_level_weighting,G)

        for j in range(3):
            for i in range(10):
                g_out = simulate_survey(G,sample_size=10000, age_dist=age_dist_dict, edge_context_dic = edge_context_dic,
                                age_attr={'sigma_map':sigma_map, 'h':h, 'b_age':b_age, 'bias_mean':bias_mean},
                                race_attr={'sigma_map':sigma_map,'anchors_race':anchors_race, 'pop_race':geo_level_weighting, 'pop_prop_race': pop_prop_race, 'l':[1,l[j]]},
                                eth_attr={'sigma_map':sigma_map,'anchors_eth':[anchors_eth,anchors_eth], 'pop_eth':geo_level_weighting, 'pop_prop_eth': pop_prop_eth},
                                ses_attr={'sigma_map':sigma_map,'anchor_ses':anchor_ses, 'pop_ses':geo_level_weighting, 'pop_prop_ses':pop_prop_ses},
                                f = [0,0], g = [0,0])
                
                pickle.dump(g_out,open('../Data/Contact survey data/' + input_network + '__' + experiment + '__' + 'survey_' + str(j) + '_' +
                                        str(i) + '_' + str(datetime.date.today()) + '.pickle', 'wb' ))

                if i % 5 == 0:
                    print(datetime.datetime.now(), 'Survey complete: ', i)

    if experiment == 'exp3':
        h = 10
        b_age = 2.564
        sigma_map = {'H':0,'W':0.44,'S':0.25,'C_N':0.39,'C_D':0.39}
        anchors_race = [0.8*2.564,0.8*2.564]
        anchors_eth = 0.8*2.564
        anchor_ses = 0.8*2.564
        l = 1 #np.linspace(0,1,10)
        geo_level_weighting = ['state','tract','random','dominant']

        for j in range(3):
            pop_prop_race = get_race_lookup(geo_level_weighting[j],G)
            pop_prop_eth = get_eth_lookup(geo_level_weighting[j],G)
            pop_prop_ses = get_ses_lookup(geo_level_weighting[j],G)
            for i in range(10):
                g_out = simulate_survey(G,sample_size=10000, age_dist=age_dist_dict, edge_context_dic = edge_context_dic,
                                age_attr={'sigma_map':sigma_map, 'h':h, 'b_age':b_age, 'bias_mean':bias_mean},
                                race_attr={'sigma_map':sigma_map,'anchors_race':anchors_race, 'pop_race':geo_level_weighting[j], 'pop_prop_race': pop_prop_race, 'l':[l,l]},
                                eth_attr={'sigma_map':sigma_map,'anchors_eth':[anchors_eth,anchors_eth], 'pop_eth':geo_level_weighting[j], 'pop_prop_eth': pop_prop_eth},
                                ses_attr={'sigma_map':sigma_map,'anchor_ses':anchor_ses, 'pop_ses':geo_level_weighting[j], 'pop_prop_ses':pop_prop_ses},
                                f = [0,0], g = [0,0])
                
                pickle.dump(g_out,open('../Data/Contact survey data/' + input_network + '__' + experiment + '__' + 'survey_' + str(geo_level_weighting[j]) + '_' +
                                        str(i) + '_' + str(datetime.date.today()) + '.pickle', 'wb' ))

                if i % 5 == 0:
                    print(datetime.datetime.now(), 'Survey complete: ', i)

    # for i in range(10):
    #             g_out = simulate_survey(G,sample_size=10000, age_dist=age_dist_dict, edge_context_dic = edge_context_dic,
    #                             age_attr={'sigma_map':sigma_map, 'h':h, 'b_age':b_age, 'bias_mean':bias_mean},
    #                             race_attr={'sigma_map':sigma_map,'anchors_race':anchors_race, 'pop_race':geo_level_weighting, 'pop_prop_race': pop_prop_race, 'l':[l,l]},
    #                             eth_attr={'sigma_map':sigma_map,'anchors_eth':anchors_eth, 'pop_eth':geo_level_weighting, 'pop_prop_eth': pop_prop_eth},
    #                             ses_attr={'sigma_map':sigma_map,'anchor_ses':anchor_ses, 'pop_ses':geo_level_weighting, 'pop_prop_ses':pop_prop_ses},
    #                             f = [0,0], g = [0,0])
                
    #             pickle.dump(g_out,open('../Data/Contact survey data/' + input_network + '__' + experiment + '__' + str(j) + 'survey__' + 
    #                                     str(i) + '_' + str(datetime.date.today()) + '.pickle', 'wb' ))

    #             if i % 5 == 0:
    #                 print(datetime.datetime.now(), 'Survey complete: ', i)

    # f_right_param_space = np.linspace(0,0.2,3)
    # f_wrong_param_space = np.linspace(0.8,1,3)
    # l_param_space = np.linspace(0,1,3)
    # sym = 'A'
    # n_pop_param_space = ['tract','state','random']


    # for geo_level_weighting in n_pop_param_space:
    #     pop_prop_race = get_race_lookup(geo_level_weighting,G)#,filename='../Data/Misc/Tract_race_demographics.pickle')
    #     pop_prop_eth = get_eth_lookup(geo_level_weighting,G)#,filename='../Data/Misc/Tract_eth_demographics.pickle')
    #     pop_prop_ses = get_ses_lookup(geo_level_weighting,G)#,filename='../Data/Misc/Tract_ses_demographics.pickle')
    # for l in l_param_space:

    #EXP 2
    # pickle.dump(g_out,open('../Data/Contact survey data/' + input_network + '__' + experiment + '__' + 'survey_' + sym + '_' + str(round(l,2)).replace('.','-') + '_' + 
    #                         str(i) + '_' + str(datetime.date.today()) + '.pickle', 'wb' ))

    # #EXP 3
    # # pickle.dump(g_out,open('../Data/Contact survey data/' + input_network + '__' + experiment + '__' + 'survey_' + geo_level_weighting + '_' +
    # #                         str(i) + '_' + str(datetime.date.today()) + '.pickle', 'wb' ))



