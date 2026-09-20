# Simulating the impact of perception bias on social contact surveys for infectious disease modelling
This data and code accompanies the journal article, 'Simulating the impact of perception bias on social contact surveys for infectious disease modelling' by Harris et al. (2025). A preprint is available at: doi.org/10.48550/arXiv.2511.03897

Abstract: Social contact patterns are a key input to many infectious disease models. Contact surveys, where participants are asked to provide information on their recent close and casual contacts with others, are one of the standard methods to measure contact patterns in a population. Surveys that require detailed sociodemographic descriptions of contacts allow for the specification of fine-grained contact rates between subpopulations in models. However, perception biases affecting a surveyed person's ability to estimate sociodemographic attributes (e.g., age, race) of others could affect contact rates derived from survey data. Here, we simulate contact surveys using a synthetic contact network of New Mexico, USA to investigate the impact of these biases on survey accuracy and infectious disease model projections. We found that perception biases affecting the estimation of another individual's age and race substantially decreased the accuracy of the derived contact patterns. Using these biased patterns in a Susceptible-Infectious-Recovered compartmental model lead to an underestimation of cumulative incidence among older people (65+ years) and individuals identifying as races other than White. Our study shows that perception biases can impact contact patterns estimated from surveys in ways that systematically underestimate disease burden in minority populations when used in transmission models.

## File hierarchy 
- Model:
    - Population_preprocessing.py - script for pre-processing urbanpop synthetic population data - assigns contact groups to individuals (schoolgroups & workgroups), determines agent income stratum and re-assigns individuals with daytime locations outside of New Mexico
    - Network_constructor.py - script for constructing contact network from synthetic population data
    - Survey_simulator.py - script for simulating contact survey on contact network
    - Matrix_constructor.py - script for constructing contact matrices from simulated contact surveys or full contact networks
    - Matrix_analyser.py - script for computing average matrices and visualations of contact matrices
    - SIR_simulator.py - script for running SIR epidemic model given a contact matrix
    - Figure generators - folder containing scripts for generating main text figures
    - Misc analysis:
        - Age_bias_fit.py - script for fitting quadratic to age-related estimation bias data (see Supplementary Material)
        - Contact_matrix_bias_analysis.py - script for measuring perception bias impact on contact matrices (see Supplementary Material)
        - Context_analysis.py - script for running bias contrbution by transmission setting (see Supplementary Material)
        - Ethnicity_analysis.py - script for generating ethnicity bias analysis (see Supplementary Material)
        - Income_analysis.py - script for generating income bias analysis (see Supplementary Material)
        - Population_analysis.py - script for generating population summary (see Supplementary Material)
        - Race_contact_comparison.py - script for comparing racial bias results to experimental estimates (see Supplementary Material)
        - Within_group_analysis.py - script for analysing alternative model of racial bias with within-group bias (see Supplementary Material)
- Data:
    - Synthetic population - folder containing synthetic population data stored in .csv file
    - Contact network - folder containing contact networks stored in .pickle files
    - Contact survey data - folder containing contact surveys stored in .pickle files
    - Contact matrices - folder containing contact matrices derived from contact surveys or whole contact networks stored in .npy files & .pdf visualisations
    - SIR trajectories - folder containing infectious & recovered curves from SIR model runs stored in .npy files
    - Misc - folder contains supporting files for network generation and survey simulation
- Figures - folder contains figures from main text and supplementary material

## Requirements
- python3 (3.9.7):
    - pandas (2.2.3), numpy (1.26.4), networkx (3.2.1), scipy (1.13.1), seaborn (0.13.2), matplotlib (3.9.2), random, itertools, gc, datetime, pickle, glob, copy

## Setup
The main workflow for reproducing main text figures:
1.  Run Population_preprocessing.py to assign contact groups and income stratum to individuals.
2.  Run Network_constructor.py to construct contact network from processed synthetic population.
3.  Run Survey_simulator.py to simulate contact surveys on constructed contact network.
4.  Run Matrix_constructor.py to construct contact matrices from simulated contact surveys or full contact networks.
5.  Run Matrix_analyser.py to compute average matrices over simulated set. 
6.  Run SIR_simulator.py to execute SIR model for given contact matrix under different disease spread assumptions.
7.  Run figure generator files (Figure_2_gen.py, Figure_3_gen.py, Figure_4_gen.py).

## Notes:
- Due to the GitHub file size limit, the synthetic population data has been included in compressed format and the contact network data has been ommitted. See 10.5281/zenodo.22845383 for the full dataset.
- UrbanPop data sourced from: Tuccillo, J., & Gaboardi, J. (2026). UrbanPop Nighttime/Daytime: New Mexico 2019 (Version V1.0.0) [Dataset]. Zenodo. https://doi.org/10.48690/1532537

## Release:
These data are approved for public distribution by Los Alamos National Laboratory under LA-UR-25-31216.
