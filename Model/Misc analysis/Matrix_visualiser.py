from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter, MultipleLocator
import numpy as np
import seaborn as sb
import glob
import os


if __name__ == '__main__':
    input_network = 'NM_network_v3'
    date = '2025-06-25'


    input_params_all = [str(h) for h in range(3)]#['state','tract','random','dominant']#['A_0-0','A_0-5','A_1-0']#['0-0','0-5','1-0']#
    experiment = 'exp2b'

    attr_all = ['r','ar']
    raw_all = [False,True]
    groundtruth_all = [False,True]

    for input_params in input_params_all:
        for attr in attr_all:
            for raw in raw_all:
                for groundtruth in groundtruth_all:

                    if raw:
                        if groundtruth:
                            search_string = '../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__raw__' + attr + '__Overall.npy'
                        else:
                            search_string = '../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__raw__' + attr + '__Overall.npy'
                    else:
                        if groundtruth:
                            search_string = '../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__Overall.npy'
                        else:
                            search_string = '../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__Overall.npy'

                    list_of_files = glob.glob(search_string) 
                    # latest_file = max(list_of_files, key=os.path.getctime)

                    if len(list_of_files) > 1:
                        cm_total = np.load(list_of_files[0])
                        for i in list_of_files[1:]:
                            cm = np.load(i)
                            cm_total = np.add(cm_total,cm)

                        cm_total = cm_total / len(list_of_files)
                    else:
                        cm_total = np.load(list_of_files[0])


                    fig, ax = plt.subplots(nrows=1, ncols=1)

                    sb.heatmap(cm_total, vmin=0,ax=ax, cbar_kws={'label': 'Mean # unique contacts \n per day'})

                    ax.invert_yaxis()

                    if attr == 'a':
                        ax.set(title='Age', xlabel="Participant Age", ylabel="Contact Age")
                        ax.xaxis.set_major_locator(MultipleLocator(2))
                        ax.xaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,10))))
                        ax.yaxis.set_major_locator(MultipleLocator(2))
                        ax.yaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,10))))
                        ax.tick_params(which='major', pad=2, labelsize=7)

                    if attr == 'e':
                        eth_dist_labels = ['.', 'Non-Hispanic','Hispanic']
                        ax.set(title='Ethnicity', xlabel="Participant Ethnicity", ylabel="Contact Ethnicity")
                        ax.xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
                        ax.xaxis.set_major_formatter(FixedFormatter(eth_dist_labels))
                        ax.yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
                        ax.yaxis.set_major_formatter(FixedFormatter(eth_dist_labels))
                        ax.tick_params(which='major', pad=2, labelsize=7,labelrotation=45)
                        ax.set_yticklabels(labels=eth_dist_labels,va='center')

                    if attr == 'r':
                        race_dist_labels = ['.', 'White', 'Black', 'Asian', 'AIAN',  'NHPI','Other','Multi']
                        ax.set(title='Race', xlabel="Participant Race", ylabel="Contact Race")
                        ax.xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
                        ax.xaxis.set_major_formatter(FixedFormatter(race_dist_labels))
                        ax.yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
                        ax.yaxis.set_major_formatter(FixedFormatter(race_dist_labels))
                        ax.tick_params(which='major', pad=2, labelsize=7,labelrotation=45)
                        ax.set_yticklabels(labels=race_dist_labels,va='center')

                    if attr == 's':
                        ses_dist_labels = ['.', 'Lower','Middle','Upper']
                        ax.set(title='Social Class', xlabel="Participant Social Class", ylabel="Contact Social Class")
                        ax.xaxis.set_major_locator(MultipleLocator(1,offset=0.5))
                        ax.xaxis.set_major_formatter(FixedFormatter(ses_dist_labels))
                        ax.yaxis.set_major_locator(MultipleLocator(1,offset=0.5))
                        ax.yaxis.set_major_formatter(FixedFormatter(ses_dist_labels))
                        ax.tick_params(which='major', pad=2, labelsize=7,labelrotation=45)
                        ax.set_yticklabels(labels=ses_dist_labels,va='center')

                    if attr == 'ae':
                        ax.xaxis.set_major_locator(MultipleLocator(2))
                        ax.xaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,5))))
                        ax.xaxis.set_minor_locator(MultipleLocator(1, offset=0.5))
                        ax.xaxis.set_minor_formatter(FixedFormatter([0] + ['NH','H']*18))
                        ax.yaxis.set_major_locator(MultipleLocator(2))
                        ax.yaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,5))))
                        ax.yaxis.set_minor_locator(MultipleLocator(1, offset=0.5))
                        ax.yaxis.set_minor_formatter(FixedFormatter([0] + ['NH','H']*18))
                        ax.tick_params(which='major', pad=10, labelsize=10)
                        ax.tick_params(which='minor', labelsize=5)

                    if attr == 'ar':
                        race_abbr = ['W','B','A','N','H','O','M']
                        ax.xaxis.set_major_locator(MultipleLocator(7))
                        ax.xaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,5))))
                        ax.xaxis.set_minor_locator(MultipleLocator(1, offset=0.5))
                        ax.xaxis.set_minor_formatter(FixedFormatter([0] + race_abbr*18))
                        ax.yaxis.set_major_locator(MultipleLocator(7))
                        ax.yaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,5))))
                        ax.yaxis.set_minor_locator(MultipleLocator(1, offset=0.5))
                        ax.yaxis.set_minor_formatter(FixedFormatter([0] + race_abbr*18))
                        ax.tick_params(which='major', pad=10, labelsize=10)
                        ax.tick_params(which='minor', labelsize=5)
                    
                    if attr == 'as':
                        ax.xaxis.set_major_locator(MultipleLocator(3))
                        ax.xaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,5))))
                        ax.xaxis.set_minor_locator(MultipleLocator(1, offset=0.5))
                        ax.xaxis.set_minor_formatter(FixedFormatter([0] + ['L','M','H']*18))
                        ax.yaxis.set_major_locator(MultipleLocator(3))
                        ax.yaxis.set_major_formatter(FixedFormatter([0] + list(range(0,91,5))))
                        ax.yaxis.set_minor_locator(MultipleLocator(1, offset=0.5))
                        ax.yaxis.set_minor_formatter(FixedFormatter([0] + ['L','M','H']*18))
                        ax.tick_params(which='major', pad=10, labelsize=10)
                        ax.tick_params(which='minor', labelsize=5)

                        
                    if raw:
                        if groundtruth:
                            fig.savefig('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__raw__' + attr + '__Overall_visualisation.pdf')
                            np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__raw__' + attr + '__Overall.npy',arr=cm_total)
                        else:
                            fig.savefig('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__raw__' + attr + '__Overall_visualisation.pdf')
                            np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__raw__' + attr + '__Overall.npy',arr=cm_total)
                    else:
                        if groundtruth:
                            fig.savefig('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt_processed__' + attr + '__Overall_visualisation.pdf')
                            np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__gt__processed__' + attr + '__Overall.npy',arr=cm_total)
                        else:
                            fig.savefig('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__Overall_visualisation.pdf')
                            np.save('../Data/Contact matrices/' + input_network + '__' + experiment + '__' + input_params + '_*_' + date + '__biased__processed__' + attr + '__Overall.npy',arr=cm_total)

