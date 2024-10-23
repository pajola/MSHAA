import os
import pandas as pd


def tabulate_results(name_dataset, lrs, surrogate_types, poison_rate, VAE_metric, VAE_cost_function):
    """
    Tabulate the results of the attack evaluation
    :param name_dataset: name of the dataset
    :param lrs: learning rates
    :param surrogate_types: clean or surrogate
    :param poison_rate: percentage of poison samples
    :param VAE_metric: l0 or energy
    :param VAE_cost_function: linear or exponential
    :return:
    """
    loss_type = VAE_metric + '_' + VAE_cost_function

    neurons, Lambdas = None, None
    if name_dataset == 'MNIST':
        neurons = [32, 64, 128]
        Lambdas = ['[32,64,128]', '[32]', '[64]', '[128]']
    elif name_dataset == 'CIFAR10':
        neurons = [128, 256, 512]
        Lambdas = ['[128,256,512]', '[128]', '[256]', '[512]']
    elif name_dataset == 'SpeechCommands':
        neurons = [128, 256]
        Lambdas = ['[128,256]', '[128]', '[256]']

    if lrs == ['full_knowledge_vae']:
        Lambdas = ['full_knowledge']

    metric = None
    if VAE_metric == 'l0' and name_dataset == 'SpeechCommands':
        metric = 'l0'
    if (VAE_metric == 'l0' or VAE_metric == 'energy') and name_dataset != 'SpeechCommands':
        metric = 'Energy (pJ)'
    if VAE_metric == 'latency':
        metric = 'Latency (s)'
    if VAE_metric == 'genError':
        metric = 'Validation Loss'

    result_dict = {'name': [], '>0': [], '>0.5': [], '<0': [], '=0': [], 'nan': [], 'inf': [], 'score': []}
    result_dict_tot = {'name': [], '>0': [], '>0.5': [], '<0': [], '=0': [], 'nan': [], 'inf': [], 'score': []}
    result_dict_complete = {'name': [], '>0': [], '>0.5': [], '<0': [], '=0': [], 'nan': [], 'inf': [], 'score': []}

    for train_type in surrogate_types:
        for Lambda in Lambdas:
            attack_evaluation, result_dict = tabulate_results_per_neuron(
                name_dataset, loss_type, lrs, neurons, Lambda, poison_rate, metric, train_type, result_dict
            )
            attack_evaluation, result_dict_tot = tabulate_results_per_lr(
                name_dataset, loss_type, lrs, Lambda, poison_rate, metric, train_type, result_dict_tot
            )
            if lrs == ['full_knowledge_vae']:
                attack_evaluation, result_dict_complete = tabulate_results_complete(
                    name_dataset, loss_type, lrs, Lambda, poison_rate, metric, train_type, result_dict_complete
                )

    # Save the statistics of attack success
    if lrs == ['full_knowledge_vae']:
        path_name = './stats/' + name_dataset + '_' + loss_type + '_complete_attack_success_lr_neuron_stats_100'
        stats_summary(result_dict, path_name, metric)
        path_name = './stats/' + name_dataset + '_' + loss_type + '_complete_attack_success_lr_stats_100'
        stats_summary(result_dict_tot, path_name, metric)
        path_name = './stats/' + name_dataset + '_' + loss_type + '_attack_success_complete_stats_100'
        stats_summary(result_dict_complete, path_name, metric)
    else:
        path_name = './stats/' + name_dataset + '_' + loss_type + '_attack_success_lr_neuron_stats_100'
        stats_summary(result_dict, path_name, metric)
        path_name = './stats/' + name_dataset + '_' + loss_type + '_attack_success_lr_stats_100'
        stats_summary(result_dict_tot, path_name, metric)


def get_effectiveness_score(model_grid, metric):
    """
    Get the Effectiveness Score of the model grid
    :param model_grid: statistics of the model grid
    :param metric: metric used to evaluate the models
    :return: score: Effectiveness Score
    """
    # get the index of the minimum validation poison loss
    min_poison_val_loss_index = model_grid['Validation Poison Loss'].idxmin()
    # get the metric of the minimum validation poison loss
    score_of_min_poison_val_loss = model_grid.loc[min_poison_val_loss_index, metric]
    # get the index of the minimum validation loss
    min_val_loss_index = model_grid['Validation Loss'].idxmin()
    # get the metric of the minimum validation loss
    score_of_min_val_loss = model_grid.loc[min_val_loss_index, metric]
    # get the metric of the maximum achievable score
    max_possible_score = max(model_grid[metric])
    # calculate the score
    score = (score_of_min_poison_val_loss - score_of_min_val_loss) / (
            max_possible_score - score_of_min_val_loss)
    return score


def count_stats(result_dict, evaluation_dict, name):
    """
    Count the statistics of attack success
    :param result_dict: dictionary to store the statistics
    :param evaluation_dict: dictionary of the evaluation results
    :param name: name of the attack
    :return: result_dict: updated dictionary of statistics
    """
    # Count the number of values for the key '100\\%'
    values_100_percent = evaluation_dict['PR 100\\%']
    count_greater_than_zero = sum(1 for value in values_100_percent if (value > 0))
    count_greater_than_pointFive = sum(1 for value in values_100_percent if (value > 0.5))
    count_lower_than_zero = sum(1 for value in values_100_percent if (value < 0))
    count_equal_to_zero = sum(1 for value in values_100_percent if (value == 0 or str(value) == 'nan'))
    count_nan = sum(1 for value in values_100_percent if (str(value) == 'nan'))
    count_inf = sum(1 for value in values_100_percent if (str(value) == '-inf'))
    # get the score
    score = sum(values_100_percent)
    # Update the stats
    result_dict['name'].append(name)
    result_dict['>0'].append(count_greater_than_zero)
    result_dict['>0.5'].append(count_greater_than_pointFive)
    result_dict['<0'].append(count_lower_than_zero)
    result_dict['=0'].append(count_equal_to_zero)
    result_dict['nan'].append(count_nan)
    result_dict['inf'].append(count_inf)
    result_dict['score'].append(score)

    return result_dict


def stats_summary(result_dict, file_name, metric):
    """
    Save in txt file the statistics of attack success
    :param result_dict: statistics of attack
    :param file_name: name of the file to save
    :param metric: metric used to evaluate the models
    :return: None
    """
    names = result_dict['name']
    # Initialize counters
    successes_MLaaS = 0
    h_successes_MLaaS = 0
    fail_MLaaS = 0
    ineff_MLaaS = 0
    nan_MLaaS = 0
    inf_MLaaS = 0
    scores_MLaaS = 0
    successes_Surrogate = 0
    h_successes_Surrogate = 0
    fail_Surrogate = 0
    ineff_Surrogate = 0
    nan_Surrogate = 0
    inf_Surrogate = 0
    scores_Surrogate = 0
    tot_succ = 0
    tot_h_succ = 0
    tot_fail = 0
    tot_ineff = 0
    tot_nan = 0
    tot_inf = 0
    # Compute the statistics
    for i in range(len(names)):
        if 'MLaaS' in names[i]:
            successes_MLaaS += result_dict['>0'][i]
            h_successes_MLaaS += result_dict['>0.5'][i]
            fail_MLaaS += result_dict['<0'][i]
            ineff_MLaaS += result_dict['=0'][i]
            nan_MLaaS += result_dict['nan'][i]
            inf_MLaaS += result_dict['inf'][i]
            scores_MLaaS += result_dict['score'][i]
        if 'Surrogate' in names[i]:
            successes_Surrogate += result_dict['>0'][i]
            h_successes_Surrogate += result_dict['>0.5'][i]
            fail_Surrogate += result_dict['<0'][i]
            ineff_Surrogate += result_dict['=0'][i]
            nan_Surrogate += result_dict['nan'][i]
            inf_Surrogate += result_dict['inf'][i]
            scores_Surrogate += result_dict['score'][i]
        tot_succ += result_dict['>0'][i]
        tot_h_succ += result_dict['>0.5'][i]
        tot_fail += result_dict['<0'][i]
        tot_ineff += result_dict['=0'][i]
        tot_nan += result_dict['nan'][i]
        tot_inf += result_dict['inf'][i]
    # Save all the metrics to a txt file
    with open(f'./' + file_name + '.txt', 'w') as f:
        f.write("METRIC FOR EFFECTIVENESS SCORE FUNCTION: " + metric + '\n\n')

        f.write("Number of successful attacks against MLaaS Grids: " + str(successes_MLaaS) + '\n')
        f.write("Number of highly successful attacks against MLaaS Grids: " + str(h_successes_MLaaS) + '\n')
        f.write("Number of failed attacks against MLaaS Grids: " + str(fail_MLaaS) + '\n')
        f.write("Number of ineffective against MLaaS Grids: " + str(ineff_MLaaS) + '\n')
        f.write("Number of NaN Scores MLaaS Grids: " + str(nan_MLaaS) + '\n')
        f.write("Number of -inf Scores MLaaS Grids: " + str(inf_MLaaS) + '\n')
        f.write("Total Number of attacks against MLaaS Grids: " +
                str(successes_MLaaS + fail_MLaaS + ineff_MLaaS) + '\n'
                )
        f.write("Percentage of successful attacks against MLaaS Grids: " +
                str(100 * (successes_MLaaS / (successes_MLaaS + fail_MLaaS + ineff_MLaaS - nan_MLaaS - inf_MLaaS))) +
                '\\% \n'
                )
        f.write("Mean Effectiveness Score MLaaS: " + str(scores_MLaaS / (
                successes_MLaaS + fail_MLaaS + ineff_MLaaS)) + '\n')
        f.write('\n')
        f.write("Number of successful attacks against Surrogate Grids: " + str(successes_Surrogate) + '\n')
        f.write("Number of highly successful attacks against Surrogate Grids: " + str(h_successes_Surrogate) + '\n')
        f.write("Number of failed attacks against Surrogate Grids: " + str(fail_Surrogate) + '\n')
        f.write("Number of ineffective attacks against Surrogate Grids: " + str(ineff_Surrogate) + '\n')
        f.write("Number of NaN Scores Surrogate Grids: " + str(nan_Surrogate) + '\n')
        f.write("Number of -inf Scores Surrogate Grids: " + str(inf_Surrogate) + '\n')
        f.write("Total Number of attacks against Surrogate Grids: " +
                str(successes_Surrogate + fail_Surrogate + ineff_Surrogate) + '\n'
                )
        f.write("Percentage of successful attacks against Surrogate Grids: " +
                str(100 * successes_Surrogate / (successes_Surrogate + fail_Surrogate + ineff_Surrogate)) + '\\% \n'
                )
        f.write("Percentage of successful attacks against Surrogate Grids removing impossible cases: " +
                str(100 * (successes_Surrogate / (successes_Surrogate + fail_Surrogate + ineff_Surrogate - inf_Surrogate
                    - nan_Surrogate))) + '\\% \n'
                )
        f.write("Mean Effectiveness Score Surrogate: " + str(scores_Surrogate / (
                successes_Surrogate + fail_Surrogate + ineff_Surrogate)) + '\n')
        f.write('\n')
        f.write("Overall number of successful attacks: " + str(tot_succ) + '\n')
        f.write("Overall number of highly successful attacks: " + str(tot_h_succ) + '\n')
        f.write("Overall number of failed attacks: " + str(tot_fail) + '\n')
        f.write("Overall number of ineffective attacks: : " + str(tot_ineff) + '\n')
        f.write("Overall number of NaN Scores: " + str(tot_nan) + '\n')
        f.write("Overall number of -inf Scores:: " + str(tot_inf) + '\n')
        f.write('Total number of Tests: ' + str(tot_succ + tot_fail + tot_ineff) + '\n')
        f.write("Percentage of successful attacks: " + str(100 * tot_succ / (tot_succ + tot_fail + tot_ineff)) +
                '\\% \n'
                )
        f.write("Percentage of successful attacks removing impossible cases: " +
                str(100 * tot_succ / (tot_succ + tot_fail + tot_ineff - tot_nan - tot_inf)) + '\\% \n'
                )


def tabulate_results_per_neuron(name_dataset, loss_type, lrs, neurons, Lambda, prs, metric, train_type, result_dict):
    """
    Tabulate the Effectiveness Scores for Model Grids grouped by Leaning Rate and Neurons
    :param name_dataset: name of the dataset
    :param loss_type: hijack metric and cost function type
    :param lrs: learning rates
    :param neurons: number of neurons
    :param Lambda: knowledge of the Adversary
    :param prs: poison rates
    :param metric: metric used to evaluate the models
    :param train_type: clean or surrogate
    :param result_dict: dictionary to store the statistics
    :return: attack_evaluation: DataFrame of the attack evaluation
             result_dict: updated dictionary of statistics
    """
    path_results = './results/' + name_dataset + '/' + loss_type + '/'

    empty_dict = {
        'lr': [],
        'Neurons': [],
        'PR 10\\%': [],
        'PR 20\\%': [],
        'PR 50\\%': [],
        'PR 80\\%': [],
        'PR 100\\%': []
    }

    att_lrs = lrs
    if att_lrs == ['full_knowledge_vae']:
        if name_dataset == 'MNIST':
            lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        else:
            lrs = [0.001, 0.005]

    for lr in lrs:
        for neuron in neurons:
            empty_dict['Neurons'].append(neuron)
            empty_dict['lr'].append(lr)
            for pr in prs:

                if att_lrs == ['full_knowledge_vae']:
                    result_path = str(
                        path_results + '/results_full_knowledge_vae_VAE_' + Lambda + '_' + train_type + '_' +
                        str(pr) + '.pkl')
                else:
                    result_path = str(
                        path_results + '/results_' + str(lr) + '_VAE_' + Lambda + '_' + train_type + '_' +
                        str(pr) + '.pkl')

                results = pd.read_pickle(result_path)
                # Ensure columns are integers
                results['Neurons'] = results['Neurons'].astype(int)
                results['Layers'] = results['Layers'].astype(int)
                # Group the DataFrame of Models by 'Neurons'
                group = results.groupby('Neurons')
                model_grid = group.get_group(neuron)
                # Get Score for the Model Grid
                score = get_effectiveness_score(model_grid, metric)
                # Tabulate the Score for the specific poison rate considered
                empty_dict['PR ' + str(int(pr * 100)) + '\\%'].append(score)
    # Save the Table as LaTeX file
    attack_evaluation = pd.DataFrame(empty_dict)
    attack_evaluation = attack_evaluation.sort_values(by=['lr', 'Neurons'])
    name = f'./tables/attack_evaluation_{name_dataset}_{loss_type}_{train_type}_{Lambda}'
    save_as_latex(attack_evaluation, name, 'neuron')
    # Update the statistics of attack success
    if train_type == 'clean':
        train_type = 'MLaaS'
    if train_type == 'surrogate':
        train_type = 'Surrogate'
    name = train_type + '_' + Lambda
    result_dict = count_stats(result_dict, empty_dict, name)

    return attack_evaluation, result_dict


def tabulate_results_per_lr(name_dataset, loss_type, lrs, Lambda, prs, metric, train_type, result_dict_tot):
    """
    Tabulate the Effectiveness Scores for Model Grids grouped by Leaning Rate
    :param name_dataset: Name of the dataset, MNIST or CIFAR10
    :param loss_type: Hijack metric and cost function type
    :param lrs: Learning Rates
    :param Lambda: Knowledge of the Adversary
    :param prs: Poison Rates
    :param metric: Metric used to evaluate the models
    :param train_type: Clean or Surrogate
    :param result_dict_tot: Dictionary to store the statistics
    :return: attack_evaluation: DataFrame of the attack evaluation
             result_dict_tot: updated dictionary of statistics
    """
    path_results = './results/' + name_dataset + '/' + loss_type + '/'
    empty_dict = {
        'lr': [],
        'PR 10\\%': [],
        'PR 20\\%': [],
        'PR 50\\%': [],
        'PR 80\\%': [],
        'PR 100\\%': []
    }

    att_lrs = lrs
    if att_lrs == ['full_knowledge_vae']:
        if name_dataset == 'MNIST':
            lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        else:
            lrs = [0.001, 0.005]

    for lr in lrs:
        empty_dict['lr'].append(lr)
        for pr in prs:

            if att_lrs == ['full_knowledge_vae']:
                result_path = str(
                    path_results + '/results_full_knowledge_vae_VAE_' + Lambda + '_' + train_type + '_' +
                    str(pr) + '.pkl')
            else:
                result_path = str(
                    path_results + '/results_' + str(lr) + '_VAE_' + Lambda + '_' + train_type + '_' +
                    str(pr) + '.pkl')

            results = pd.read_pickle(result_path)
            # Ensure columns are integers
            results['Neurons'] = results['Neurons'].astype(int)
            results['Layers'] = results['Layers'].astype(int)
            # Get Score for the Model Grid
            score = get_effectiveness_score(results, metric)
            # Tabulate the Score for the specific poison rate considered
            empty_dict['PR ' + str(int(pr * 100)) + '\\%'].append(score)

    # Save the Table as LaTeX file
    attack_evaluation = pd.DataFrame(empty_dict)
    attack_evaluation = attack_evaluation.drop(columns=['Neurons'], errors='ignore')
    name = f'./tables/attack_evaluation_{name_dataset}_{loss_type}_{train_type}_{Lambda}_total'
    save_as_latex(attack_evaluation, name, 'lr')
    # Update the statistics of attack success
    if train_type == 'clean':
        train_type = 'MLaaS'
    if train_type == 'surrogate':
        train_type = 'Surrogate'
    name = train_type + '_' + Lambda
    result_dict_tot = count_stats(result_dict_tot, empty_dict, name)

    return attack_evaluation, result_dict_tot


def tabulate_results_complete(name_dataset, loss_type, lrs, Lambda, prs, metric, train_type, result_dict_complete):
    """
    Tabulate the Effectiveness Scores for Model Grids of all trained models
    :param name_dataset: Name of the dataset, MNIST or CIFAR10
    :param loss_type: Hijack metric and cost function type
    :param lrs: Learning Rates
    :param Lambda: Knowledge of the Adversary
    :param prs: Poison Rates
    :param metric: Metric used to evaluate the models
    :param train_type: Clean or Surrogate
    :param result_dict_complete: Dictionary to store the statistics
    :return: attack_evaluation: DataFrame of the attack evaluation
             result_dict_complete: updated dictionary of statistics
    """
    path_results = './results/' + name_dataset + '/' + loss_type + '/'
    empty_dict = {
        'PR 10\\%': [],
        'PR 20\\%': [],
        'PR 50\\%': [],
        'PR 80\\%': [],
        'PR 100\\%': []
    }

    for pr in prs:
        result_path = str(
            path_results + '/results_full_knowledge_vae_VAE_' + Lambda + '_' + train_type + '_' +
            str(pr) + '.pkl')
        result_complete = pd.read_pickle(result_path)

        # Ensure columns are integers
        result_complete['Neurons'] = result_complete['Neurons'].astype(int)
        result_complete['Layers'] = result_complete['Layers'].astype(int)
        # Get Score for the Model Grid
        score = get_effectiveness_score(result_complete, metric)
        # Tabulate the Score for the specific poison rate considered
        empty_dict['PR ' + str(int(pr * 100)) + '\\%'].append(score)
    # Save the Table as LaTeX file
    attack_evaluation = pd.DataFrame(empty_dict)
    attack_evaluation = attack_evaluation.drop(columns=['Neurons','lrs'], errors='ignore')
    name = f'./tables/attack_evaluation_{name_dataset}_{loss_type}_{train_type}_{Lambda}_complete'
    save_as_latex(attack_evaluation, name, 'complete')
    # Update the statistics of attack success
    if train_type == 'clean':
        train_type = 'MLaaS'
    if train_type == 'surrogate':
        train_type = 'Surrogate'
    name = train_type + '_' + Lambda
    result_dict_tot = count_stats(result_dict_complete, empty_dict, name)

    return attack_evaluation, result_dict_tot


def save_as_latex(attack_evaluation, name, grouped_by):
    """
    Save the DataFrame as a LaTeX table
    :param attack_evaluation: dictionary of the attack evaluation
    :param name: file name
    :param grouped_by: type of model grid, either 'neuron' or 'lr'
    :return: None
    """
    def color_custom(val):
        """
        Colors elements in a DataFrame:
        - light green if > 0
        - dark green if >= 0.5
        - red if < 0
        - black if  = 0
        :param val: value in the DataFrame
        :return: color of the value
        """
        if val >= 0.5:
            color = 'green'
        elif val > 0:
            color = 'blu'
        elif val < 0:
            color = 'red'
        else:
            color = 'black'
        return 'textcolor: {%s}' % color

    def custom_format(val):
        """
        Format the values in the DataFrame
        :param val: value in the DataFrame
        :return: Formatted value
        """
        if type(val) is int:
            if val == 0:
                return f"{{{int(val):.4f}}}"
            return f"{int(val):.0f}"
        else:
            return f"{{{val:.4f}}}"

    # Round the values to 3 significant figures
    attack_evaluation = attack_evaluation.round(4)

    if grouped_by == 'neuron':
        # Apply the color formatting and integer formatting for the third column
        styled_df = (attack_evaluation.style
                     .applymap(color_custom, subset=attack_evaluation.columns[2:])
                     .format(custom_format, subset=attack_evaluation.columns[2:])
                     .format("{:.4f}", subset=attack_evaluation.columns[0])
                     .hide(axis='index'))
        str_to_remove = '{rrrrrr}'
        str_to_add = \
            ('{\\textwidth}{>{\\raggedleft\\arraybackslash}X >{\\raggedleft\\arraybackslash}X | '
             '>{\\raggedleft\\arraybackslash}X >{\\raggedleft\\arraybackslash}X >{\\raggedleft\\'
             'arraybackslash}X >{\\raggedleft\\arraybackslash}X}')
    elif grouped_by == 'lr':
        # Apply the color formatting and integer formatting for the second column
        styled_df = (attack_evaluation.style
                     .applymap(color_custom, subset=attack_evaluation.columns[1:])
                     .format(custom_format, subset=attack_evaluation.columns[1:])
                     .format("{:.4f}", subset=attack_evaluation.columns[0])
                     .hide(axis='index'))
        str_to_remove = '{rrrrr}'
        str_to_add = \
            ('{\\textwidth}{>{\\raggedleft\\arraybackslash}X | >{\\raggedleft\\arraybackslash}X '
             '>{\\raggedleft\\arraybackslash}X >{\\raggedleft\\arraybackslash}X >{\\raggedleft\\arraybackslash}X}')
    else:
        # Apply the color formatting and integer formatting for the second column
        styled_df = (attack_evaluation.style
                     .applymap(color_custom, subset=attack_evaluation.columns)
                     .format(custom_format, subset=attack_evaluation.columns)
                     .hide(axis='index'))
        str_to_remove = '{rrrr}'
        str_to_add = \
            ('>{\\raggedleft\\arraybackslash}X >{\\raggedleft\\arraybackslash}X >{\\raggedleft\\arraybackslash}X'
             ' >{\\raggedleft\\arraybackslash}X}')

    # Generate the LaTeX string
    latex_str = styled_df.to_latex()

    # Insert \midrule after the header
    lines = latex_str.splitlines()
    lines.insert(1, '\\toprule')
    lines.insert(3, '\\midrule')
    lines.insert(-1, '\\bottomrule')

    # Replace tabular with tabularx and specify the table width
    lines[0] = lines[0].replace('{tabular}' + str_to_remove, '{tabularx}' + str_to_add)
    lines[-1] = lines[-1].replace('\\end{tabular}', '\\end{tabularx}')

    # Join the lines back into a single string
    latex_str = '\n'.join(lines)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(name), exist_ok=True)

    # Save the LaTeX string to a file
    with open(name + '.tex', 'w') as f:
        f.write(latex_str)

    print("Table saved as attack_evaluation.tex")


def tabulate_stats(pr=1):
    """
    Tabulate the statistics of attack success
    :param pr: poison rate
    :return: none
    """
    tested_datasets_names = ['MNIST', 'CIFAR10', 'SpeechCommands']
    model_grid = ['lr_neuron', 'lr', 'complete']
    hijack_metrics = ['l0', 'energy', 'latency', 'genError']
    setting = ['MLaaS', 'Surrogate']

    for grid_type in model_grid:

        empty_dict_WB = {
            'name': ['MNIST', 'CIFAR10', 'SpeechC.'],
            'genError': [],
            'latency': [],
            'energy': [],
            'l0': []
        }
        empty_dict_BB = {
            'name': ['MNIST', 'CIFAR10', 'SpeechC.'],
            'genError': [],
            'latency': [],
            'energy': [],
            'l0': []
        }

        for metric in hijack_metrics:
            for datasets_name in tested_datasets_names:

                if datasets_name == 'SpeechCommands' and metric == 'energy':
                    empty_dict_WB[metric].append('N/A')
                    empty_dict_BB[metric].append('N/A')
                    continue

                path_stats = ('./stats/' + datasets_name + '_' + metric + '_linear_attack_success_' + grid_type +
                              '_stats_' + str(int(pr * 100)) + '.txt')
                with open(path_stats, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'Percentage of successful attacks against MLaaS Grids:' in line:
                            percentage = line.split(': ')[1].split('\\')[0]
                            empty_dict_WB[metric].append(percentage)
                        if 'Percentage of successful attacks against Surrogate Grids removing impossible cases:' in line:
                            percentage = line.split(': ')[1].split('\\')[0]
                            empty_dict_BB[metric].append(percentage)

        # Save the Table as LaTeX file
        attack_evaluation_WB = pd.DataFrame(empty_dict_WB)
        attack_evaluation_BB = pd.DataFrame(empty_dict_BB)
        name_WB = f'./stats/stats_tables/attack_evaluation_WB_' + grid_type
        name_BB = f'./stats/stats_tables/attack_evaluation_BB_' + grid_type
        save_as_latex_stats(attack_evaluation_WB, name_WB)
        save_as_latex_stats(attack_evaluation_BB, name_BB)


def save_as_latex_stats(attack_evaluation, name):
    """
    Save the DataFrame as a LaTeX table
    :param attack_evaluation: dataframe of the statistics
    :param name: file name to save
    :return: None
    """

    def table_format(val):
        if val == 'N/A':
            return val
        return f"{float(val):.2f}" + '\\%'

    # Round the values to 3 significant figures
    attack_evaluation = attack_evaluation.round(4)

    # Apply the color formatting and integer formatting for the second column
    styled_df = (attack_evaluation.style
                 .format(table_format, subset=attack_evaluation.columns[1:])
                 .hide(axis='index'))

    # Generate the LaTeX string
    latex_str = styled_df.to_latex()

    # Insert \midrule after the header
    lines = latex_str.splitlines()
    lines.insert(1, '\\toprule')
    lines.insert(3, '\\midrule')
    lines.insert(-1, '\\bottomrule')

    str_to_add = \
        ('{0.45 \\textwidth}{>{\\raggedleft\\arraybackslash}X | >{\\raggedleft\\arraybackslash}X |'
         '>{\\raggedleft\\arraybackslash}X | >{\\raggedleft\\arraybackslash}X | >{\\raggedleft\\arraybackslash}X}')

    lines[0] = lines[0].replace('{tabular}{lllll}', '{tabularx}' + str_to_add)
    lines[-1] = lines[-1].replace('\\end{tabular}', '\\end{tabularx}')
    lines[2] = lines[2].replace('name', '')
    lines[2] = lines[2].replace('genError', '\\texttt{Gener.}')
    lines[2] = lines[2].replace('latency', '\\texttt{Latency}')
    lines[2] = lines[2].replace('energy', '\\texttt{Energy}')
    lines[2] = lines[2].replace('l0', '\\texttt{$\\ell_0$}')

    # Join the lines back into a single string
    latex_str = '\n'.join(lines)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(name), exist_ok=True)

    # Save the LaTeX string to a file
    with open(name + '.tex', 'w') as f:
        f.write(latex_str)

    print("Table saved as attack_evaluation.tex")


def get_stats(stat_metric):
    """
    Tabulate the statistics of attack success
    :param stat_metric: metric saved
    :return: none
    """
    tested_datasets_names = ['MNIST', 'CIFAR10', 'SpeechCommands']
    model_grid = ['lr_neuron', 'lr', 'complete']
    hijack_metrics = ['l0', 'energy', 'latency', 'genError']
    poison_rates = [0.1, 0.2, 0.5, 0.8, 1]

    line_to_find_WB = None
    line_to_find_BB = None

    if stat_metric == 'success_rate':
        line_to_find_WB = 'Percentage of successful attacks against MLaaS Grids:'
        line_to_find_BB = 'Percentage of successful attacks against Surrogate Grids removing impossible cases:'
    elif stat_metric == 'score':
        line_to_find_WB = 'Mean Effectiveness Score MLaaS:'
        line_to_find_BB = 'Mean Effectiveness Score Surrogate:'

    for grid_type in model_grid:

        for datasets_name in tested_datasets_names:
            WB_dict = {
                'pr': [],
                'l0': [],
                'energy': [],
                'latency': [],
                'genError': []
            }
            BB_dict = {
                'pr': [],
                'l0': [],
                'energy': [],
                'latency': [],
                'genError': []
            }
            for pr in poison_rates:
                WB_dict['pr'].append(pr)
                BB_dict['pr'].append(pr)

                for metric in hijack_metrics:

                    if datasets_name == 'SpeechCommands' and metric == 'energy':
                        WB_dict[metric].append('N/A')
                        BB_dict[metric].append('N/A')
                        continue

                    path_stats = ('./stats/' + datasets_name + '_' + metric + '_linear_attack_success_' + grid_type +
                                  '_stats_' + str(int(pr * 100)) + '.txt')

                    with open(path_stats, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if line_to_find_WB in line:
                                percentage = line.split(': ')[1].split('\\')[0]
                                WB_dict[metric].append(float(percentage))
                            if line_to_find_BB in line:
                                percentage = line.split(': ')[1].split('\\')[0]
                                BB_dict[metric].append(float(percentage))

            # Save the Table as LaTeX file
            attack_evaluation_WB = pd.DataFrame(WB_dict)
            attack_evaluation_BB = pd.DataFrame(WB_dict)
            name_WB = f'./stats/stats_tables/' + stat_metric + '_WB_' + datasets_name + '_' + grid_type
            name_BB = f'./stats/stats_tables/' + stat_metric + '_BB_' + datasets_name + '_' + grid_type

            # Ensure the directory exists
            os.makedirs(os.path.dirname(name_WB), exist_ok=True)
            os.makedirs(os.path.dirname(name_BB), exist_ok=True)

            # Save as .pkl
            attack_evaluation_WB.to_pickle(name_WB + '.pkl')
            attack_evaluation_BB.to_pickle(name_BB + '.pkl')
