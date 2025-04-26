import sys, json
import pandas as pd
from metric_functions import accuracy #, kappa, logloss, rmse
from tabulate import tabulate
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
tabulate.PRESERVE_WHITESPACE = True

def p_value(file, method='fixed_time', fixed_time=2.5, individual=True):
    
    metric = accuracy

    dfs = pd.read_csv(file)
    subjects = dfs['subject'].unique()
    classifiers = dfs['classifier'].unique()
    
    columns = ['classifier 1', 'classifier 2', 'subject', 'p-value']    
    table = pd.DataFrame(columns=columns)

    if individual:
        for classifier_1 in classifiers:
            for classifier_2 in classifiers:
                if classifier_1 == classifier_2:
                    continue
                for subject in subjects:
                    filtered_1 = dfs[(dfs['classifier'] == classifier_1) & (dfs['subject'] == subject)]
                    filtered_2 = dfs[(dfs['classifier'] == classifier_2) & (dfs['subject'] == subject)]
                    if filtered_1.empty or filtered_2.empty:
                        new_row = [classifier_1, classifier_2, subject, 'NaN']
                    else:
                        file_1 = pd.read_csv(filtered_1['path'].values[0])
                        file_2 = pd.read_csv(filtered_2['path'].values[0])

                        if method == 'fixed_time':
                            file_1 = file_1[file_1['tmin'] == fixed_time]
                            file_2 = file_2[file_2['tmin'] == fixed_time]

                            file_1['metric'] = file_1.iloc[:, 3:].idxmax(axis=1) == file_1['true_label']
                            file_2['metric'] = file_2.iloc[:, 3:].idxmax(axis=1) == file_2['true_label']
                            
                            # p_value
                            p_val = posthoc_dunn([file_1['metric'], file_2['metric']], p_adjust='bonferroni', val_col='metric')
                            # print(p_val)
                            # _, krusk = kruskal(file_1['metric'], file_2['metric'])
                            # if krusk > 0.05:
                            #     print('No significant difference between classifiers {} and {} for subject {}'.format(classifier_1, classifier_2, subject))
                            
                            new_row = [classifier_1, classifier_2, subject, str(round(p_val.iloc[0, 1], 4))]

                        elif method == 'integral':
                            for f in [file_1, file_2]:
                                times = f['tmin'].unique()
                                metrics = []
                                for time in times:        
                                    filtered = f[f['tmin'] == time]
                                    metric_value = metric(filtered['true_label'], filtered.iloc[:, 3:])
                                    metrics.append(metric_value)
                                integral = sum(metrics)
                                f['metric'] = integral

                            # p_value
                            p_val = posthoc_dunn([file_1['metric'], file_2['metric']], p_adjust='bonferroni')
                            # _, krusk = kruskal(file_1['metric'], file_2['metric'])
                            # if krusk > 0.05:
                            #     print('No significant difference between classifiers {} and {} for subject {}'.format(classifier_1, classifier_2, subject))
                            
                            new_row = [classifier_1, classifier_2, subject, str(round(p_val.iloc[0, 1], 4))]

                        elif method == 'oscillation':
                            for f in [file_1, file_2]:
                                times = f['tmin'].unique()
                                metrics = []
                                for time in times:        
                                    filtered = f[f['tmin'] == time]
                                    metric_value = metric(filtered['true_label'], filtered.iloc[:, 3:])
                                    metrics.append(metric_value)
                                oscillation = sum([abs(metrics[i] - metrics[i-1]) for i in range(1, len(metrics))])
                                f['metric'] = oscillation

                            # p_value
                            p_val = posthoc_dunn([file_1['metric'], file_2['metric']], p_adjust='bonferroni')
                            # _, krusk = kruskal(file_1['metric'], file_2['metric'])
                            # if krusk > 0.05:
                            #     print('No significant difference between classifiers {} and {} for subject {}'.format(classifier_1, classifier_2, subject))
                            
                            new_row = [classifier_1, classifier_2, subject, str(round(p_val.iloc[0, 1], 4))]

                        table = pd.concat([table, pd.DataFrame([new_row], columns=columns)], ignore_index=True)
            classifiers = classifiers[1:]
    
    classifiers = dfs['classifier'].unique()
    for classifier_1 in classifiers:
        for classifier_2 in classifiers:
            if classifier_1 == classifier_2:
                continue
            filtered_1 = dfs[dfs['classifier'] == classifier_1]
            filtered_2 = dfs[dfs['classifier'] == classifier_2]
            # print(filtered_1)
            # print(filtered_2)
            if filtered_1.empty or filtered_2.empty:
                new_row = [classifier_1, classifier_2, 'all', 'NaN']
            else:
                file_1 = pd.DataFrame()
                for file_name in filtered_1['path']:
                    read_file = pd.read_csv(file_name)
                    file_1 = pd.concat([file_1, read_file])
                file_2 = pd.DataFrame()
                for file_name in filtered_2['path']:
                    read_file = pd.read_csv(file_name)
                    file_2 = pd.concat([file_2, read_file])
                
                if method == 'fixed_time':
                    file_1 = file_1[file_1['tmin'] == fixed_time]
                    file_2 = file_2[file_2['tmin'] == fixed_time]

                    file_1['metric'] = file_1.iloc[:, 3:].idxmax(axis=1) == file_1['true_label']
                    file_2['metric'] = file_2.iloc[:, 3:].idxmax(axis=1) == file_2['true_label']

                    # p_value
                    p_val = posthoc_dunn([file_1['metric'], file_2['metric']], p_adjust='bonferroni')
                    # _, krusk = kruskal(file_1['metric'], file_2['metric'])
                    # if krusk > 0.05:
                    #     print('No significant difference between classifiers {} and {} for subject {}'.format(classifier_1, classifier_2, 'all'))               
    
                    new_row = [classifier_1, classifier_2, 'all', str(round(p_val.iloc[0, 1], 4))]

                elif method == 'integral':
                    for f in [file_1, file_2]:
                        times = f['tmin'].unique()
                        metrics = []
                        for time in times:        
                            filtered = f[f['tmin'] == time]
                            metric_value = metric(filtered['true_label'], filtered.iloc[:, 3:])
                            metrics.append(metric_value)
                        integral = sum(metrics)
                        f['metric'] = integral

                    # p_value
                    p_val = posthoc_dunn([file_1['metric'], file_2['metric']], p_adjust='bonferroni')
                    # _, krusk = kruskal(file_1['metric'], file_2['metric'])
                    # if krusk > 0.05:
                    #     print('No significant difference between classifiers {} and {} for subject {}'.format(classifier_1, classifier_2, subject))
                    
                    new_row = [classifier_1, classifier_2, 'all', str(round(p_val.iloc[0, 1], 4))]

                elif method == 'oscillation':
                    for f in [file_1, file_2]:
                        times = f['tmin'].unique()
                        metrics = []
                        for time in times:        
                            filtered = f[f['tmin'] == time]
                            metric_value = metric(filtered['true_label'], filtered.iloc[:, 3:])
                            metrics.append(metric_value)
                        oscillation = sum([abs(metrics[i] - metrics[i-1]) for i in range(1, len(metrics))])
                        f['metric'] = oscillation

                    # p_value
                    p_val = posthoc_dunn([file_1['metric'], file_2['metric']], p_adjust='bonferroni')
                    # _, krusk = kruskal(file_1['metric'], file_2['metric'])
                    # if krusk > 0.05:
                    #     print('No significant difference between classifiers {} and {} for subject {}'.format(classifier_1, classifier_2, subject))
                    
                    new_row = [classifier_1, classifier_2, 'all', str(round(p_val.iloc[0, 1], 4))]

                table = pd.concat([table, pd.DataFrame([new_row], columns=columns)], ignore_index=True)
        classifiers = classifiers[1:]
    
    # order by classifier 1, classifier 2, subject
    table = table.sort_values(by=['classifier 1', 'classifier 2', 'subject'])

    # print(table)     
    
    latex_table = tabulate(
        table, 
        tablefmt='latex_booktabs', 
        headers='keys', 
        showindex=False, 
        colalign=tuple(['center' for i in range(len(classifiers) + 1)]),
        floatfmt=tuple(['' if i == 0 else '.4f' for i in range(len(classifiers) + 1)])
    )    
    lines = latex_table.split('\n')
    lines.insert(0, '\\begin{table}[ht!]')
    lines.append('\\end{table}')
    latex_table = '\n'.join(lines)
    with open('p_value_table.tex', 'w') as f:
        f.write(latex_table)

if __name__ == '__main__':
    try:
        config = sys.argv[1]
    except IndexError:
        print('Usage: python one_file.py config.json')
        exit(1)
    
    with open(config) as json_file:
        config_read = json.load(json_file)
    
    try:
        file = config_read['file']
    except KeyError:
        print('Error: file not found')
        exit(1)
    
    #* fixed_time, integral, oscillation
    if 'method' in config_read:
        method = config_read['method']
    else:
        print('Error: method not found')
        exit(1)

    if 'time' in config_read:
        tipe_position = config_read['time']
    else:
        tipe_position = 2.5

    if 'individual' in config_read:
        individual = config_read['individual']
    else:
        individual = True

    p_value(file, method=method, fixed_time=tipe_position, individual=individual)