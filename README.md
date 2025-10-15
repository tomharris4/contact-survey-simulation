# Simulating the impact of perception bias on detailed sociodemographic surveys of social contact behaviour
This data and code accompanies the journal article, 'Simulating the impact of perception bias on detailed sociodemographic surveys of social contact behaviour' by Harris et al. (2025)
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
        - Context_analysis.py - script for running bias contrbution by transmission setting (see Supplementary Material)
        - Ethnicity_analysis.py - script for generating ethnicity bias analysis (see Supplementary Material)
        - Income_analysis.py - script for generating income bias analysis (see Supplementary Material)
        - Population_analysis.py - script for generating population summary (see Supplementary Material)
        - Race_contact_comparison.py - script for comparing racial bias results to experimental estimates (see Supplementary Material)
        - Within_group_analysis.py - script for analysing alternative model of racial bias with within-group bias (see Supplementary Material)
- Data:
    - Synthetic population
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
