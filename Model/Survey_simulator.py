# Script for simulating contact survey on contact network

import datetime
import numpy as np
import networkx as nx
import pickle

# Sample age of person 'v' given they are a contact of participant 'u', context of contact event, and setting specific recall accuracy scalers
def sample_age(u,v,context,age_attr):

    # Define mean & variance of Gamma distribution for age sampling
    mean = v['age'] + (age_attr['sigma_map'][context] * age_attr['b_age'] * age_attr['bias_mean'][v['age']])
    variance = age_attr['sigma_map'][context] * age_attr['h']

    # If mean == 0, re-assign to value just greater one to avoid division by zero
    if mean <= 0:
        mean = 0.05
    
    # If variance == 0, return mean
    if variance == 0:
        return mean
    
    # Convert distribution mean and variance to scale and shape parameters
    scale = variance / mean
    shape = mean / scale
    
    # Return rounded sample from distribution
    return int(np.floor(rng.gamma(shape=shape,scale=scale)))

# Sample race of person 'v' given they are a contact of participant 'u', context of contact event, and setting specific recall accuracy scalers
def sample_race(u,v,context,race_attr):

    # Check if participant or contact are non-White
    l_ind = int(u['race'] != 0)
    r_ind = int(v['race'] != 0)

    # Check if participant and contact identify as same race
    if v['race'] != u['race']:
        ind = 0 
    else:
        ind = 1

    # Compute categorical probability distribution for racial estimation - see Main Text: Methods & Supplementary Material
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
        elif race_attr['pop_race'] == 'majority':
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
        elif race_attr['pop_race'] == 'majority':
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

# Sample ethnicity of person 'v' given they are a contact of participant 'u', context of contact event, and setting specific recall accuracy scalers
def sample_ethnicity(u,v,context,eth_attr): 
    
    # Check if individual being contacted is Hispanic - if so, use r_h error weighting 
    if v['ethnicity'] == 1:
        ind = 1
    else:
        ind = 0

    # Use contact ethnicity and population distribution to inform ethnicity estimation
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

# Sample income stratum of person 'v' given they are a contact of participant 'u', context of contact event, and setting specific recall accuracy scalers
def sample_income(u,v,context,income_attr):

    # Use population distribution to inform income estimation
    if income_attr['pop_income'] == 'state':
        income_weights = [income_attr['pop_prop_income'][h+1] * income_attr['anchor_income'] * income_attr['sigma_map'][context] for h in range(3)]
        income_weights[int(v['income'])-1] = 1 - (1 - income_attr['pop_prop_income'][int(v['income'])]) * income_attr['anchor_income'] * income_attr['sigma_map'][context]
    elif income_attr['pop_income'] == 'tract':
        if context in ['H','C_N']:
            income_weights = [income_attr['pop_prop_income']['N_' + str(v['night_tract'])[:-1]][h+1] * income_attr['anchor_income'] * income_attr['sigma_map'][context] for h in range(3)]
            income_weights[int(v['income'])-1] = 1 - (1 - income_attr['pop_prop_income']['N_' + str(v['night_tract'])[:-1]][int(v['income'])]) * income_attr['anchor_income'] * income_attr['sigma_map'][context]
        else:
            income_weights = [income_attr['pop_prop_income']['D_' + str(v['day_tract'])[:-1]][h+1] * income_attr['anchor_income'] * income_attr['sigma_map'][context] for h in range(3)]
            income_weights[int(v['income'])-1] = 1 - (1 - income_attr['pop_prop_income']['D_' + str(v['day_tract'])[:-1]][int(v['income'])]) * income_attr['anchor_income'] * income_attr['sigma_map'][context]
    else:
        income_weights = [(1/3) * income_attr['anchor_income'] * income_attr['sigma_map'][context] for h in range(3)]
        income_weights[int(v['income'])-1] = 1 - (2/3) * income_attr['anchor_income'] * income_attr['sigma_map'][context]
    
    return rng.choice(a=range(1,4), p=income_weights)

# Generate race distribution at state or tract level
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

# Generate ethnicity distribution at state or tract level
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

# Generate income distribution at state or tract level
def get_income_lookup(geo,G,filename=None):
    if geo == 'state':
        agents_income = np.unique(list(nx.get_node_attributes(G,'income').values()),return_counts=True)
        agents_income = dict(zip(agents_income[0],agents_income[1]))
        pop_prop_income = {}

        for i in range(1,4):
            pop_prop_income[i] = agents_income[i] / len(G.nodes())

    elif geo == 'tract':
        if filename:
            return pickle.load(open(filename, 'rb'))
        
        agents_tract_N = np.unique(['N_' + str(h[1]['night_tract'])[:-1] for h in G.nodes(data=True)],return_counts=True)
        agents_tract_N = dict(zip(agents_tract_N[0],agents_tract_N[1]))

        agents_tract_D = np.unique(['D_' + str(h[1]['day_tract'])[:-1] for h in G.nodes(data=True)],return_counts=True)
        agents_tract_D = dict(zip(agents_tract_D[0],agents_tract_D[1]))
        
        agents_income_N = np.unique([('N_' + str(h[1]['night_tract'])[:-1] + str(int(h[1]['income']))) for h in G.nodes(data=True)],return_counts=True)
        agents_income_N = dict(zip(agents_income_N[0],agents_income_N[1]))

        agents_income_D = np.unique([('D_' + str(h[1]['day_tract'])[:-1] + str(int(h[1]['income']))) for h in G.nodes(data=True)],return_counts=True)
        agents_income_D = dict(zip(agents_income_D[0],agents_income_D[1]))

        pop_prop_income = {}

        for i in agents_tract_N:
            pop_prop_income[i] = {}
            for j in range(1,4):
                if (i + str(j)) in agents_income_N:
                    pop_prop_income[i][j] = agents_income_N[(i + str(j))] / agents_tract_N[i]
                else:
                    pop_prop_income[i][j] = 0

        for i in agents_tract_D:
            pop_prop_income[i] = {}
            for j in range(1,4):
                if (i + str(j)) in agents_income_D:
                    pop_prop_income[i][j] = agents_income_D[(i + str(j))] / agents_tract_D[i]
                else:
                    pop_prop_income[i][j] = 0

        pickle.dump(pop_prop_income,open('../Data/Misc/Tract_income_demographics.pickle', 'wb' ))

    else:
        pop_prop_income = None

    return pop_prop_income

# Generate bias mean for all integer ages between 0-100 (i.e. apply B(x) as described in main text)
def get_age_bias_mean():

    bias_mean = []

    for i in np.linspace(0, 100, 100):
        bias_mean.append(((-0.0035*i**2 + 1.188*i) - i))

    return bias_mean

# Generate age distribution (aggregated at 5year age bands) at state level
def get_age_distribution(G):
    age_dist_network_binned = []

    for node in G.nodes:
        a = np.digitize(G.nodes[node]['age'],np.linspace(0,85,18)) - 1
        age_dist_network_binned.append(a)
        

    age_dist_dict = {}

    for i in range(len(np.unique(age_dist_network_binned, return_counts=True)[0])):
        age_dist_dict[np.unique(age_dist_network_binned, return_counts=True)[0][i]] = np.unique(age_dist_network_binned, return_counts=True)[1][i] / len(G.nodes())

    return age_dist_dict

# Simulate contact survey    
def simulate_survey(G, edge_context_dic, age_attr, race_attr, eth_attr, income_attr, sample_size=10000):

    # Output dictionary holding true and sampled values for every contact pair
    G_out = {}

    node_list = list(G.nodes())

    # Choose survey participants through uniform random sample of population
    sampled_nodes = rng.choice(node_list,size=sample_size,replace=False)

    for node in sampled_nodes:
        G_out[node] = {}

    # For each survey participant sample attributes for each contacted individual
    for n in sampled_nodes:
        for u,v in G.edges(n):
            if (u,v) in edge_context_dic:
                context = edge_context_dic[(u,v)]
            else:
                context = edge_context_dic[(v,u)]
            
            e_a = sample_age(G.nodes[u],G.nodes[v],context,age_attr)
            e_r = sample_race(G.nodes[u],G.nodes[v],context,race_attr)
            e_e = sample_ethnicity(G.nodes[u],G.nodes[v],context,eth_attr)
            e_s = sample_income(G.nodes[u],G.nodes[v],context,income_attr)

            G_out[u][v] = {}
            
            G_out[u][v]['recall_age_estimate'] = e_a
            G_out[u][v]['recall_race_estimate'] = e_r
            G_out[u][v]['recall_eth_estimate'] = e_e
            G_out[u][v]['recall_income_estimate'] = e_s

    return G_out

    
if __name__ == '__main__':

    # Load input contact network; assign to G
    input_network = 'NM_network'
    G = pickle.load(open('../Data/Contact network/' + input_network + '.pickle', 'rb'))

    # Get transmission setting ('context') of all contacts in G
    edge_context_dic = nx.get_edge_attributes(G,'context')

    # Define age-related bias for all integer ages between 0-100 (i.e. apply B(x) as described in main text) 
    bias_mean = get_age_bias_mean()

    # Define age distibution at state level (5year age bands)
    age_dist_dict = get_age_distribution(G)

    # Define experimental conditions for contact survey simulation
    # Main analysis: Experiment 1 ('exp1'), Experiment 2 ('exp2')
    # Supplemental analysis: within-group bias ('supp_wg'), transmissing setting SA ('supp_exp_context')
    experiments = ['exp1', 'exp2']

    print(datetime.datetime.now(), 'Initialisation complete')

    for experiment in experiments:
        rng = np.random.RandomState(10)

        # Experiment 1 - age-related perception bias
        if experiment == 'exp1':
            h = 10 
            b_age = np.linspace(0, 1.5*2.564, 7)
            b_age = np.append(b_age,1.5*2.564)
            sigma_map = {'H':0,'W':0.44,'S':0.25,'C_N':0.39,'C_D':0.39}
            anchors_race = np.linspace(0.1*2.564,0.8*2.564,8)
            anchors_eth = np.linspace(0.1*2.564,0.8*2.564,8)
            anchor_income = np.linspace(0.1*2.564,0.8*2.564,8)
            l = 1
            geo_level_weighting = 'tract'
            pop_prop_race = get_race_lookup(geo_level_weighting,G)
            pop_prop_eth = get_eth_lookup(geo_level_weighting,G)
            pop_prop_income = get_income_lookup(geo_level_weighting,G)

            for j in range(7):
                for i in range(10):
                    g_out = simulate_survey(G,sample_size=10000, edge_context_dic = edge_context_dic,
                                    age_attr={'sigma_map':sigma_map, 'h':h, 'b_age':b_age[j], 'bias_mean':bias_mean},
                                    race_attr={'sigma_map':sigma_map,'anchors_race':[0,anchors_race[j]], 'pop_race':geo_level_weighting, 'pop_prop_race': pop_prop_race, 'l':[l,l]},
                                    eth_attr={'sigma_map':sigma_map,'anchors_eth':[anchors_eth[j],0], 'pop_eth':geo_level_weighting, 'pop_prop_eth': pop_prop_eth},
                                    income_attr={'sigma_map':sigma_map,'anchor_income':anchor_income[j], 'pop_income':geo_level_weighting, 'pop_prop_income':pop_prop_income},
                                    )
                    
                    pickle.dump(g_out,open('../Data/Contact survey data/' + input_network + '__' + experiment + '__' + 'survey_' + str(j) + '_' +
                                            str(i) + '.pickle', 'wb' ))

                    if i % 5 == 0:
                        print(datetime.datetime.now(), 'Survey complete: ', i)

        # Experiment 2 - race-related perception bias
        if experiment == 'exp2':
            h = 10
            b_age = 0
            sigma_map = {'H':0,'W':0.44,'S':0.25,'C_N':0.39,'C_D':0.39}
            anchors_race = np.linspace(0*2.564,0.8*2.564,9)
            anchors_eth = np.linspace(0*2.564,0.8*2.564,9)
            anchor_income = np.linspace(0*2.564,0.8*2.564,9)
            l = 1
            geo_level_weighting = ['tract'] # ['state','tract','random','majority']

            for j in range(len(geo_level_weighting)):
                pop_prop_race = get_race_lookup(geo_level_weighting[j],G)
                pop_prop_eth = get_eth_lookup(geo_level_weighting[j],G)
                pop_prop_income = get_income_lookup(geo_level_weighting[j],G)
                for k in range(9):
                    for i in range(10):
                        g_out = simulate_survey(G,sample_size=10000, edge_context_dic = edge_context_dic,
                                        age_attr={'sigma_map':sigma_map, 'h':h, 'b_age':b_age, 'bias_mean':bias_mean},
                                        race_attr={'sigma_map':sigma_map,'anchors_race':[0,anchors_race[k]], 'pop_race':geo_level_weighting[j], 'pop_prop_race': pop_prop_race, 'l':[l,l]},
                                        eth_attr={'sigma_map':sigma_map,'anchors_eth':[0,anchors_eth[k]], 'pop_eth':geo_level_weighting[j], 'pop_prop_eth': pop_prop_eth},
                                        income_attr={'sigma_map':sigma_map,'anchor_income':anchor_income[k], 'pop_income':'random', 'pop_prop_income':pop_prop_income},
                                        )
                        
                        pickle.dump(g_out,open('../Data/Contact survey data/' + input_network + '__' + experiment + '__' + 'survey_' + str(geo_level_weighting[j]) + '_' +
                                                str(k) + '_' + str(i) + '.pickle', 'wb' ))

                        if i % 5 == 0:
                            print(datetime.datetime.now(), 'Survey complete: ', i)

        # Supplemetal experiment - race-related perception bias (within-group bias)
        if experiment == 'supp_wg':
            h = 10
            b_age = 2.564
            sigma_map = {'H':0,'W':0.44,'S':0.25,'C_N':0.39,'C_D':0.39}
            anchors_race = [0.8*2.564,0.8*2.564]
            anchors_eth = 0.8*2.564
            anchor_income = 0.8*2.564
            l = np.linspace(0,1,5)
            geo_level_weighting = 'tract'
            pop_prop_race = get_race_lookup(geo_level_weighting,G)
            pop_prop_eth = get_eth_lookup(geo_level_weighting,G)
            pop_prop_income = get_income_lookup(geo_level_weighting,G)

            for j in range(5):
                for i in range(10):
                    g_out = simulate_survey(G,sample_size=10000, edge_context_dic = edge_context_dic,
                                    age_attr={'sigma_map':sigma_map, 'h':h, 'b_age':b_age, 'bias_mean':bias_mean},
                                    race_attr={'sigma_map':sigma_map,'anchors_race':anchors_race, 'pop_race':geo_level_weighting, 'pop_prop_race': pop_prop_race, 'l':[l[j],1]},
                                    eth_attr={'sigma_map':sigma_map,'anchors_eth':[anchors_eth,anchors_eth], 'pop_eth':geo_level_weighting, 'pop_prop_eth': pop_prop_eth},
                                    income_attr={'sigma_map':sigma_map,'anchor_income':anchor_income, 'pop_income':geo_level_weighting, 'pop_prop_income':pop_prop_income},
                                    )
                    
                    pickle.dump(g_out,open('../Data/Contact survey data/' + input_network + '__' + experiment + '__' + 'survey_' + str(j) + '_' +
                                            str(i) + '.pickle', 'wb' ))

                    if i % 5 == 0:
                        print(datetime.datetime.now(), 'Survey complete: ', i)

        # Supplemetal experiment - transmission setting sensitivity analysis
        if experiment == 'supp_exp_context':
            h = 10
            b_age = 2.564
            anchors_race = 0.7*2.564
            anchors_eth = 0.7*2.564
            anchor_income = 0.7*2.564
            l = 1
            geo_level_weighting = 'tract'
            pop_prop_race = get_race_lookup(geo_level_weighting,G)
            pop_prop_eth = get_eth_lookup(geo_level_weighting,G)
            pop_prop_income = get_income_lookup(geo_level_weighting,G)

            sigma_map = [
                {'H':0,'W':0,'S':0,'C_N':0,'C_D':0},
                {'H':0,'W':0,'S':0,'C_N':0.39,'C_D':0.39},
                {'H':0,'W':0,'S':0.25,'C_N':0,'C_D':0},
                {'H':0,'W':0.44,'S':0,'C_N':0,'C_D':0},
                {'H':0,'W':0,'S':0.25,'C_N':0.39,'C_D':0.39},
                {'H':0,'W':0.44,'S':0,'C_N':0.39,'C_D':0.39},
                {'H':0,'W':0.44,'S':0.25,'C_N':0,'C_D':0},
                {'H':0,'W':0.44,'S':0.25,'C_N':0.39,'C_D':0.39}
                ]

            for k in range(8):
                for i in range(10):
                    g_out = simulate_survey(G,sample_size=10000, edge_context_dic = edge_context_dic,
                                    age_attr={'sigma_map':sigma_map[k], 'h':h, 'b_age':b_age, 'bias_mean':bias_mean},
                                    race_attr={'sigma_map':sigma_map[k],'anchors_race':[0,anchors_race], 'pop_race':geo_level_weighting, 'pop_prop_race': pop_prop_race, 'l':[l,1]},
                                    eth_attr={'sigma_map':sigma_map[k],'anchors_eth':[0,anchors_eth], 'pop_eth':geo_level_weighting, 'pop_prop_eth': pop_prop_eth},
                                    income_attr={'sigma_map':sigma_map[k],'anchor_income':anchor_income, 'pop_income':geo_level_weighting, 'pop_prop_income':pop_prop_income},
                                    )
                    
                    pickle.dump(g_out,open('../Data/Contact survey data/' + input_network + '__' + experiment + '__' + 'survey_' + str(k) + '_' +
                                            str(i) + '.pickle', 'wb' ))

                    if i % 5 == 0:
                        print(datetime.datetime.now(), 'Survey complete: ', i)