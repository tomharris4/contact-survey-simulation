# Script for pre-processing urbanpop synthetic population data - generates epicast contact groupings (schoolgroups & workgroups), 
# determines agent social class and re-assigns individuals with daytime locations outside of New Mexico

import pandas as pd
import numpy as np

rng = np.random.RandomState(10)

# Avg. uni classroom size - https://publicuniversityhonors.com/2015/10/20/estimated-class-sizes-more-than-90-national-universities/
uni_class = 37.4135

# Read in raw urbanpop synthetic population data
urbanpop = pd.read_csv('../Data/Synthetic population/urbanpop_network_nm_001.csv')

# Remove NaN rows at end of csv
urbanpop = urbanpop[urbanpop['household_id'].notnull()]

# Re-assign Texan workers/students
daygroups_counter = np.unique(urbanpop['daytime_blockgroup'],return_counts=True)
daygroups = list(zip(daygroups_counter[0],daygroups_counter[1]))
lc_bgs = [h[0] for h in daygroups if str(h[0])[0:5] == '35013' and h[1] < 5000] # lc -> Las Cruces
texas_bgs = [h[0] for h in daygroups if str(h[0])[0:2] == '48']
lc_naics_tracker = {} # lc -> Las Cruces
texas_naics_tracker ={}

for row in urbanpop.iterrows():
    if row[1]['daytime_blockgroup'] in lc_bgs:
        if row[1]['employment_industry'] in lc_naics_tracker:
            if row[1]['daytime_blockgroup'] in lc_naics_tracker[row[1]['employment_industry']]:
                lc_naics_tracker[row[1]['employment_industry']][row[1]['daytime_blockgroup']] += 1
            else:
                lc_naics_tracker[row[1]['employment_industry']][row[1]['daytime_blockgroup']] = 1
        else:
            lc_naics_tracker[row[1]['employment_industry']] = {}
            lc_naics_tracker[row[1]['employment_industry']][row[1]['daytime_blockgroup']] = 1
    elif row[1]['daytime_blockgroup'] in texas_bgs:
        if row[1]['daytime_blockgroup'] in texas_naics_tracker:
            if row[1]['employment_industry'] in texas_naics_tracker[row[1]['daytime_blockgroup']]:
                texas_naics_tracker[row[1]['daytime_blockgroup']][row[1]['employment_industry']] += 1
            else:
                texas_naics_tracker[row[1]['daytime_blockgroup']][row[1]['employment_industry']] = 1
        else:
            texas_naics_tracker[row[1]['daytime_blockgroup']] = {}
            texas_naics_tracker[row[1]['daytime_blockgroup']][row[1]['employment_industry']] = 1

texas_bg_mapping = {}

for i in texas_naics_tracker:
    max_naics = max(texas_naics_tracker[i], key=texas_naics_tracker[i].get)
    texas_bg_mapping[i] = max(lc_naics_tracker[max_naics], key=lc_naics_tracker[max_naics].get)
    lc_naics_tracker[max_naics][texas_bg_mapping[i]] = 0

new_daygroups = np.empty(len(urbanpop),dtype='<U35')
k = 0

for row in urbanpop.iterrows():
    if row[1]['daytime_blockgroup'] in texas_bgs:
        new_daygroups[k] = texas_bg_mapping[row[1]['daytime_blockgroup']]
    else:
        new_daygroups[k] = row[1]['daytime_blockgroup']
    k += 1

urbanpop['daytime_blockgroup'] = new_daygroups

daygroups_counter = np.unique(urbanpop['daytime_blockgroup'],return_counts=True)
daygroups = list(zip(daygroups_counter[0],daygroups_counter[1]))
lc_bgs = [h for h in daygroups if str(h[0])[0:5] == '35013']

daytime_comm = urbanpop['daytime_blockgroup']

urbanpop_schools = urbanpop[urbanpop['school_id'].notnull()]

school_to_grades = urbanpop_schools.groupby(by=['school_id','daytime_blockgroup']).school_grade.apply(list)
school_to_grades_dict = dict()

sg_sizes = pd.read_csv('../Data/Misc/counties_schoolgroups.csv')

sg_sizes['counties'] = [h[1:-2] for h in sg_sizes['counties']]

sg_sizes = sg_sizes[[h[0:2] == '35' for h in sg_sizes['counties']]] # New Mexico

sg_sizes['pre_k'] = [h[1:-2] for h in sg_sizes['pre_k']]
sg_sizes['kind'] = [h[1:-2] for h in sg_sizes['kind']]
sg_sizes['elem'] = [h[1:-2] for h in sg_sizes['elem']]
sg_sizes['sec'] = [h[1:-2] for h in sg_sizes['sec']]

grade_sg_sizes = {}
for row in sg_sizes.iterrows():
    grade_sg_sizes[(row[1]['counties'],1)] = row[1]['pre_k']
    grade_sg_sizes[(row[1]['counties'],2)] = row[1]['kind']

    for j in range(3,8):
        grade_sg_sizes[(row[1]['counties'],j)] = row[1]['elem']

    for j in range(8,15):
        grade_sg_sizes[(row[1]['counties'],j)] = row[1]['sec']

    for j in range(15,17):
        grade_sg_sizes[(row[1]['counties'],j)] = uni_class

school_to_grades = dict(school_to_grades)
teacher_to_grades_dict = dict()

for i in school_to_grades:
    sg_dict = {}
    sg_teacher_dict = {}
    uniq = np.unique(school_to_grades[i],return_counts=True)
    for j in range(len(uniq[0])):
        num_sgs = max(1,round(uniq[1][j] / int(grade_sg_sizes[(str(i[1])[0:5],uniq[0][j])])))
        sg_dict[uniq[0][j]] = num_sgs
    school_to_grades_dict[i] = sg_dict

school_ids = urbanpop['school_id']
school_grades = urbanpop['school_grade']
work_naics = urbanpop['employment_industry'].copy()

sgs = np.empty(len(school_ids),dtype='<U35')

for i in range(len(school_ids)):
    s_id = school_ids[i]
    grade_id = school_grades[i]
    dt_id = daytime_comm[i]

    if not pd.isnull(school_ids[i]):
        sg_temp = str(dt_id) + '_' + str(s_id)[:13] + '_' + str(grade_id) + '_' + str(rng.randint(1,school_to_grades_dict[(s_id,dt_id)][grade_id]+1))
        sgs[i] = sg_temp
        if dt_id in teacher_to_grades_dict:
            teacher_to_grades_dict[dt_id][sg_temp] = False
        else:
            teacher_to_grades_dict[dt_id] = {sg_temp: False}
    else:
        sgs[i] = 0

for i in range(len(work_naics)):
    naics_id = work_naics[i]
    dt_id = daytime_comm[i]

    if naics_id == 611 and dt_id in teacher_to_grades_dict and (not all(list(teacher_to_grades_dict[dt_id].values()))):
        for j in teacher_to_grades_dict[dt_id]:
            if not teacher_to_grades_dict[dt_id][j]:
                sgs[i] = j
                teacher_to_grades_dict[dt_id][j] = True
                work_naics[i] = 0
                break

urbanpop['school_group'] = sgs
urbanpop['work_naics_temp'] = work_naics

urbanpop_work = urbanpop[urbanpop['work_naics_temp'] > 0]

naics_to_wg_size = pd.read_csv('../Data/Misc/workgroups.csv')

naics_to_wg_size = naics_to_wg_size[naics_to_wg_size['FIPS'] == 35] # New Mexico

naics_to_wg_size['NAICS'] = [h.ljust(3,'0') for h in naics_to_wg_size['NAICS']]

naics_to_wg_size['NAICS'] = [h.replace('M','0') for h in naics_to_wg_size['NAICS']]

naics_to_wg_size['NAICS'] = [h.replace('S','0') for h in naics_to_wg_size['NAICS']]

naics_to_wg_size = dict(zip(naics_to_wg_size['NAICS'],naics_to_wg_size['workgroup size']))

dt_to_industry = urbanpop_work.groupby(by='daytime_blockgroup').work_naics_temp.apply(list)
industry_to_wgs = dict(dt_to_industry)
industry_to_wgs_dict = {}

for i in industry_to_wgs:
    wg_dict = {}
    uniq = np.unique(industry_to_wgs[i],return_counts=True)
    for j in range(len(uniq[0])):
        num_wgs = max(1,round(uniq[1][j] / naics_to_wg_size[str(uniq[0][j])]))
        wg_dict[uniq[0][j]] = num_wgs
    industry_to_wgs_dict[i] = wg_dict

industries = urbanpop['work_naics_temp']

wgs = np.empty(len(daytime_comm),dtype='<U25')

for i in range(len(daytime_comm)):
    d_id = daytime_comm[i]
    naics_id = industries[i]

    if industries[i] > 0:
        wgs[i] = str(d_id) + '_' + str(naics_id) + '_' + str(rng.randint(1,industry_to_wgs_dict[d_id][naics_id]+1))
    else:
        wgs[i] = 0


urbanpop['work_group'] = wgs

urbanpop = urbanpop.drop('work_naics_temp', axis=1)

# Assign social class based on household income
agents_hh = urbanpop.groupby(by=['household_id']).count()
hh_size_lookup = dict(zip(agents_hh.index,agents_hh['household_income']))

urbanpop['hh_size_scaler'] = [hh_size_lookup[h]**0.5 for h in urbanpop['household_id']]

urbanpop['household_income_adjusted'] = urbanpop['household_income'] / urbanpop['hh_size_scaler']

median_national_income = 70784
mean_hh_size = 2.51

med_us_adj_hh_income = median_national_income / mean_hh_size**0.5
ses_quantile_low = (2/3) * med_us_adj_hh_income 
ses_quantile_high = 2 * med_us_adj_hh_income
bins = [0,ses_quantile_low,ses_quantile_high]

social_class = np.empty(len(urbanpop))
k = 0

for row in urbanpop.iterrows():
    social_class[k] = int(np.digitize(row[1]['household_income_adjusted'],bins=bins))
    k += 1

urbanpop['social_class'] = social_class

urbanpop = urbanpop.drop('hh_size_scaler', axis=1)
urbanpop = urbanpop.drop('household_income_adjusted', axis=1)

urbanpop.to_csv('../Data/Synthetic population/urbanpop_network_nm_001_processed.csv')