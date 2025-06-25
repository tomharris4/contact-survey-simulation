# Script for constructing contact network from synthetic population data

import random
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import itertools
import datetime
import gc

def build_graph(context='H', filename=None, pop_night_file=None, pop_day_file=None):

    random.seed(10)
    
    agents = pd.read_csv('../Data/Synthetic population/urbanpop_network_nm_001_processed.csv')
    agents = agents.sort_values(by=['nighttime_blockgroup','household_id'])

    N = len(agents)

    if not filename:
        G = nx.Graph()

        for row in agents.iterrows():
            G.add_node(row[0], day_tract=row[1]['daytime_blockgroup'], night_tract=row[1]['nighttime_blockgroup'], age = row[1]['age'], ethnicity = row[1]['ethnicity'], race = row[1]['race'], ses=row[1]['social_class'])

    else:
        G = pickle.load(open(filename, 'rb'))

    
    # POLYMOD contact rate distribution (daytime/nighttime community/household contacts assumed evenly split)
    prop_h_contact = 0.23
    prop_w_contact = 0.21
    prop_s_contact = 0.14
    prop_c_day_contact = 0.21
    prop_c_night_contact = 0.21
    
    total_possible_hh_contacts = 2347917

    # Overall daily contact rate 
    c_total = total_possible_hh_contacts / (0.5 * N * prop_h_contact)

    # Context-specific contact rates
    c_h = prop_h_contact * c_total
    c_w = prop_w_contact * c_total
    c_s = prop_s_contact * c_total
    c_c_day = prop_c_day_contact * c_total 
    c_c_night = prop_c_night_contact * c_total 

    # Nighttime community
    if 'H' == context or 'C_N' == context:
        comms_night, comms_night_index, community_sizes_night = np.unique(agents['nighttime_blockgroup'], return_counts=True, return_index=True)
        comms_night = comms_night[np.argsort(comms_night_index)]
        community_sizes_night = community_sizes_night[np.argsort(comms_night_index)]
        p_size_night = int(sum([(c-1)*c/2 for c in community_sizes_night]))
        cum_comm_sizes_night = [0] + list(np.cumsum(community_sizes_night))

        hh_lookup = dict(zip(agents.index,agents['household_id']))
        wg_lookup = dict(zip(agents.index,agents['work_group']))
        sg_lookup = dict(zip(agents.index,agents['school_group']))

        if 'H' == context:
            p_home = np.empty(p_size_night, dtype=np.uint8)
        if 'C_N' == context:
            p_comm_night = np.empty(p_size_night, dtype=np.uint8)
    
        if not pop_night_file:
            pop_night = np.empty(p_size_night, dtype='2i')

            k = 0
            for i in range(len(comms_night)):
                if community_sizes_night[i] > 1:
                    temp = k + int((community_sizes_night[i]-1)*community_sizes_night[i]/2)
                    agent_ids = agents.index[cum_comm_sizes_night[i]:cum_comm_sizes_night[i+1]]
                    pop_night[k:temp] = list(itertools.combinations(agent_ids, 2))
                    k = temp
            np.save(file='../Data/Synthetic population/pop_night.npy', arr=pop_night)
        else:
            pop_night = np.load(file=pop_night_file)

        print(datetime.datetime.now(), 'Nighttime edge set ready')

        community_sizes_night = [h for h in community_sizes_night if h > 1]

        if 'H' == context:
            i = 0
            temp_hh = hh_lookup[pop_night[0][0]]
            comm_tracker = 0
            agent_tracker = 0
            hh_size = 0
            while i < p_size_night:
                if temp_hh == hh_lookup[pop_night[i][1]]:
                    p_home[i] = 1
                    hh_size += 1
                    i += 1
                else:
                    skip_len = community_sizes_night[comm_tracker] - 1 - hh_size - agent_tracker

                    #skip i/p_home/p_comm_night forward
                    p_home[i:i+skip_len] = 0
                    i += skip_len

                    agent_tracker += 1
                    if agent_tracker + 1 == community_sizes_night[comm_tracker]:
                        comm_tracker += 1
                        agent_tracker = 0

                    #update temp_hh
                    if i < p_size_night:
                        temp_hh = hh_lookup[pop_night[i][0]]
                        hh_size = 0

            print(datetime.datetime.now(), 'Household contacts probability distribution ready')

            p_home = np.nonzero(p_home)[0]

            for i in p_home:
                sample = pop_night[i]
                G.add_edge(sample[0],sample[1],context='H')

            del p_home
            del pop_night
            gc.collect()

            print(datetime.datetime.now(), 'Household contacts finished sampling')

        if 'C_N' == context:
            i = 0
            temp_hh = hh_lookup[pop_night[0][0]]
            comm_tracker = 0
            agent_tracker = 0
            hh_size = 0
            while i < p_size_night:
                if temp_hh == hh_lookup[pop_night[i][1]]:
                    p_comm_night[i] = 1
                    hh_size += 1
                    i += 1
                else:
                    skip_len = community_sizes_night[comm_tracker] - 1 - hh_size - agent_tracker

                    #skip i/p_home/p_comm_night forward
                    p_comm_night[i:i+skip_len] = 1
                    i += skip_len

                    agent_tracker += 1
                    if agent_tracker + 1 == community_sizes_night[comm_tracker]:
                        comm_tracker += 1
                        agent_tracker = 0

                    #update temp_hh
                    if i < p_size_night:
                        temp_hh = hh_lookup[pop_night[i][0]]
                        hh_size = 0

            print(datetime.datetime.now(), 'Nighttime community probability distribution ready')

            p_comm_night = np.cumsum(p_comm_night, dtype=np.uint64)

            i = 0
            while i < int(0.5*N*c_c_night):
                sample = random.choices(pop_night,cum_weights=p_comm_night,k=1)[0]
                
                if (not G.has_edge(sample[0],sample[1])) and (not G.has_edge(sample[1],sample[0])):
                    G.add_edge(sample[0],sample[1],context='C_N')
                    i += 1

            del pop_night
            del p_comm_night
            gc.collect()

            print(datetime.datetime.now(), 'Nighttime community contacts finished sampling')

    # Daytime community
    if 'W' == context or 'S' == context or 'C_D' == context:
        attr_assort = 'race'
        rho = 1
        agents = agents.sort_values(by=['daytime_blockgroup','school_group', 'work_group'])

        # Daytime community
        comms_day, comms_day_index, community_sizes_day = np.unique(agents['daytime_blockgroup'], return_counts=True, return_index=True)
        comms_day = comms_day[np.argsort(comms_day_index)]
        community_sizes_day = community_sizes_day[np.argsort(comms_day_index)]
        p_size_day = int(sum([(c-1)*c/2 for c in community_sizes_day]))
        cum_comm_sizes_day = [0] + list(np.cumsum(community_sizes_day))

        wg_lookup = dict(zip(agents.index,agents['work_group']))
        sg_lookup = dict(zip(agents.index,agents['school_group']))
        attr_lookup = dict(zip(agents.index,agents[attr_assort]))

        if 'W' == context:
            p_work = np.empty(p_size_day, dtype=np.uint8)
        if 'S' == context:
            p_school = np.empty(p_size_day, dtype=np.uint8)
        if 'C_D' == context:
            p_comm_day = np.empty(p_size_day, dtype=np.uint8)

        if not pop_day_file:
            pop_day = np.empty(p_size_day, dtype='2i')
            k = 0
            for i in range(len(comms_day)):
                if community_sizes_day[i] > 1:
                    temp = k + int((community_sizes_day[i]-1)*community_sizes_day[i]/2)
                    agent_ids = agents.index[cum_comm_sizes_day[i]:cum_comm_sizes_day[i+1]]
                    pop_day[k:temp] = list(itertools.combinations(agent_ids, 2))
                    k = temp
            np.save(file='../Data/Synthetic population/pop_day.npy', arr=pop_day)
        else:
            pop_day = np.load(file=pop_day_file, mmap_mode='r')

        print(datetime.datetime.now(), 'Daytime edge set ready')

        community_sizes_day = [h for h in community_sizes_day if h > 1]

        if 'W' == context:
            i = 0
            comm_tracker = 0
            agent_tracker = 0
            temp_wg = wg_lookup[pop_day[0][0]]
            temp_sg = sg_lookup[pop_day[0][0]]
            g_size = 0
            while i < p_size_day:
                if temp_wg != '0' and temp_wg == wg_lookup[pop_day[i][1]]:
                    if attr_lookup[pop_day[i][0]] == attr_lookup[pop_day[i][1]]:
                        p_work[i] = rho
                    else:
                        p_work[i] = 1
                    g_size += 1
                    i += 1
                elif temp_sg != '0' and temp_sg == sg_lookup[pop_day[i][1]]:
                    p_work[i] = 0
                    g_size += 1
                    i += 1
                else:
                    skip_len = community_sizes_day[comm_tracker] - 1 - g_size - agent_tracker

                    # skip i/p_home/p_comm_day forward
                    p_work[i:i+skip_len] = 0
                    i += skip_len

                    agent_tracker += 1
                    if agent_tracker + 1 >= community_sizes_day[comm_tracker]:
                        comm_tracker += 1
                        agent_tracker = 0

                    # update temps
                    if i < p_size_day:
                        temp_wg = wg_lookup[pop_day[i][0]]
                        temp_sg = sg_lookup[pop_day[i][0]]
                        g_size = 0

            print(datetime.datetime.now(), 'Workplace probability distribution ready')

            p_work = np.cumsum(p_work, dtype=np.uint32)

            i = 0
            while i < int(0.5*N*c_w):
                sample = random.choices(pop_day,cum_weights=p_work,k=1)[0]

                if (not G.has_edge(sample[0],sample[1])) and (not G.has_edge(sample[1],sample[0])):
                    G.add_edge(sample[0],sample[1],context='W')
                    i += 1

            del p_work
            gc.collect()
            print(datetime.datetime.now(), 'Workplace contacts finished sampling')

        if context == 'S':
            i = 0
            comm_tracker = 0
            agent_tracker = 0
            temp_wg = wg_lookup[pop_day[0][0]]
            temp_sg = sg_lookup[pop_day[0][0]]
            g_size = 0
            while i < p_size_day:
                if temp_wg != '0' and temp_wg == wg_lookup[pop_day[i][1]]:
                    p_school[i] = 0
                    g_size += 1
                    i += 1
                elif temp_sg != '0' and temp_sg == sg_lookup[pop_day[i][1]]:
                    if attr_lookup[pop_day[i][0]] == attr_lookup[pop_day[i][1]]:
                        p_school[i] = rho
                    else:
                        p_school[i] = 1
                    g_size += 1
                    i += 1
                else:
                    skip_len = community_sizes_day[comm_tracker] - 1 - g_size - agent_tracker

                    # skip i/p_home/p_comm_day forward
                    p_school[i:i+skip_len] = 0
                    i += skip_len

                    agent_tracker += 1
                    if agent_tracker + 1 >= community_sizes_day[comm_tracker]:
                        comm_tracker += 1
                        agent_tracker = 0

                    # update temps
                    if i < p_size_day:
                        temp_wg = wg_lookup[pop_day[i][0]]
                        temp_sg = sg_lookup[pop_day[i][0]]
                        g_size = 0

            print(datetime.datetime.now(), 'School probability distribution ready')

            p_school = np.cumsum(p_school, dtype=np.uint32)

            i = 0
            while i < int(0.5*N*c_s):
                sample = random.choices(pop_day,cum_weights=p_school,k=1)[0]

                if (not G.has_edge(sample[0],sample[1])) and (not G.has_edge(sample[1],sample[0])):
                    G.add_edge(sample[0],sample[1],context='S')
                    i += 1

            del p_school
            gc.collect()
            print(datetime.datetime.now(), 'School contacts finished sampling')

        if 'C_D' == context:
            i = 0
            comm_tracker = 0
            agent_tracker = 0
            temp_wg = wg_lookup[pop_day[0][0]]
            temp_sg = sg_lookup[pop_day[0][0]]
            g_size = 0
            while i < p_size_day:
                if temp_wg != '0' and temp_wg == wg_lookup[pop_day[i][1]]:
                    p_comm_day[i] = 1
                    g_size += 1
                    i += 1
                elif temp_sg != '0' and temp_sg == sg_lookup[pop_day[i][1]]:
                    p_comm_day[i] = 1
                    g_size += 1
                    i += 1
                else:
                    skip_len = community_sizes_day[comm_tracker] - 1 - g_size - agent_tracker

                    # skip i/p_home/p_comm_day forward
                    p_comm_day[i:i+skip_len] = 1
                    i += skip_len

                    agent_tracker += 1
                    if agent_tracker + 1 >= community_sizes_day[comm_tracker]:
                        comm_tracker += 1
                        agent_tracker = 0

                    # update temps
                    if i < p_size_day:
                        temp_wg = wg_lookup[pop_day[i][0]]
                        temp_sg = sg_lookup[pop_day[i][0]]
                        g_size = 0

            print(datetime.datetime.now(), 'Daytime community probability distribution ready')

            p_comm_day = np.cumsum(p_comm_day, dtype=np.uint64)

            i = 0
            while i < int(0.5*N*c_c_day):
                sample = random.choices(pop_day,cum_weights=p_comm_day,k=1)[0]
                
                if (not G.has_edge(sample[0],sample[1])) and (not G.has_edge(sample[1],sample[0])):
                    G.add_edge(sample[0],sample[1],context='C_D')
                    i += 1


            del pop_day
            del p_comm_day
            gc.collect()

            print(datetime.datetime.now(), 'Daytime community contacts finished sampling')



    # Output graph
    pickle.dump(G, open('../Data/Contact network/NM_network_v3.pickle', 'wb'))
    return G

if __name__ == '__main__':

    # build_graph(context='H', filename=None, pop_day_file=None, pop_night_file=None)
    build_graph(context='H', filename=None, pop_day_file=None, pop_night_file='../Data/Synthetic population/pop_night.npy')

    # build_graph(context='W', filename='../Data/Contact network/NM_network_v3.pickle', pop_day_file=None, pop_night_file=None)
    build_graph(context='W', filename='../Data/Contact network/NM_network_v3.pickle', pop_day_file='../Data/Synthetic population/pop_day.npy', pop_night_file=None)

    build_graph(context='S', filename='../Data/Contact network/NM_network_v3.pickle', pop_day_file='../Data/Synthetic population/pop_day.npy', pop_night_file=None)

    build_graph(context='C_N', filename='../Data/Contact network/NM_network_v3.pickle', pop_day_file=None, pop_night_file='../Data/Synthetic population/pop_night.npy')

    build_graph(context='C_D', filename='../Data/Contact network/NM_network_v3.pickle', pop_day_file='../Data/Synthetic population/pop_day.npy', pop_night_file=None)