import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#######################################################
## HELPER FUNCTIONS
#######################################################

def generator_score(df: pd.DataFrame, metric: str):
    """
    returns an array containing the generator score (its efficiency) for all test cases found.
    It supposes that df has been already reduced so that each planner result corresponds to a test case.
    """
    results = []
    for planner in df['planner'].unique().tolist():
        planner_df = df.query('planner==@planner')
        results.append(100 * (len(planner_df[planner_df[metric] == 1]) / len(planner_df)))
    return results

def heuristic_score(df: pd.DataFrame):
    """
    returns an array containing the heuristic score (its efficiency) for all test cases found.
    It supposes that df has been already reduced so that each planner result corresponds to a test case.
    """
    results = []
    for planner in df['planner'].unique().tolist():
        planner_df = df.query('planner==@planner').reset_index(drop=True)
        if planner_df.query('failure==1').empty == False:
            # results.append(100 * (planner_df.index[planner_df['failure']==1].tolist()[0] + 1) / len(planner_df))
            results.append(planner_df.index[planner_df['failure']==1].tolist()[0] + 1)
    return results

#######################################################
## FUNCTIONS USED TO EXPORT / PLOT RESULTS
## THEY MAY USE GLOBAL VARIABLES DECLARED IN SCRIPT SECTION
## (TO SEE BELOW)
#######################################################

def csv_to_latex(df_filepath: str, n: 'list[int]', transpose=False):
    """
    reads a DataFrame from the .csv file and writes it as a latex table
    Parameters
    ----------
    n :
        The list of the column indexes to use as row labels
    """
    df = pd.read_csv(df_filepath, header=0, index_col=n)

    filename = re.match('(.+).csv', df_filepath).group(1)

    f = open(filename + '.tex', mode='w')
    with pd.option_context("max_colwidth", None):
        if transpose:
            f.write(df.transpose().to_latex(index=(n != [])))
        else:
            f.write(df.to_latex(index=(n != [])))
    f.close()

def save_csv_non_optimal_detection(filepath: str, df: pd.DataFrame, planners: 'list[str]', generator: str, heuristic: str):
    """
    saves a csv where each :
        - columns headers are domains
        - rows headers are planners found in DataFrame
        - cells are boolean values : whether some test cases violated the metamorphic relation
    """
    f = open(filepath, mode='w')
    # writes header
    f.write(',' + ','.join([f'{k} ({len(v)})' for (k, v) in domains_dict.items()]) + '\n')
    for planner in planners:
            # cells of a line
            cells = [planners_dict[planner] if planner in planners_dict.keys() else planner]
            for domain in domains:
                # reduces the df for the current configuration and domain
                reduced_df = df.query('domain==@domain & planner==@planner & generator==@generator & heuristic==@heuristic').reset_index(drop=True)
                if reduced_df.query('error==0').empty:
                    cells.append('NaN')
                else:
                    # True is non-optimal results found
                    cells.append(str(not reduced_df.query('failure==1').empty))
            f.write(','.join(cells) + '\n')
    f.close()

def save_csv_non_optimal_score(filepath: str, planners: 'list[str]'):
    """
    saves a csv where each :
        - columns headers are domains
        - rows headers are generators and heuristics
        - cells are the number of faulty competitors killed by the configuration (row headers) on the domain (column headers)
    """
    # configurations of the framework considered
    generators = ['generator1', 'generator2', 'generator3', 'generator4']
    heuristics = ['h_cost', 'h_distance_with_g', 'h_distance_with_i', 'h_reversed_cost', 'h_zero']

    f = open(filepath, mode='w')
    # writes header
    f.write(',,' + ','.join([f'{k} ({len(v)})' for (k, v) in domains_dict.items()]) + '\n')
    for generator in generators:
        for heuristic in heuristics:
            # cells of a line
            cells = [generators_dict[generator], heuristics_dict[heuristic]]
            for domain in domains:
                # reduces the df for the current configuration and domain
                reduced_df = df.query('generator==@generator & heuristic==@heuristic & domain==@domain')
                nb_planners_killed = 0
                nb_planners_run = 0
                for planner in planners:
                    planner_df = reduced_df.query('planner==@planner').reset_index(drop=True)
                    # checks if results for the given planner exist
                    if planner_df.query('error==0').empty == False:
                        nb_planners_run += 1
                        if planner_df.query('failure==1').empty == False:
                            nb_planners_killed += 1
                if nb_planners_run == 0:
                    cells.append('NaN')
                else:
                    cells.append(f'{(100 * (nb_planners_killed / nb_planners_run)):.2f}%')
            f.write(','.join(cells) + '\n')
    # adds the performance of the baseline configuration
    generator = 'generator5'
    heuristic = 'h_zero'
    baseline_df = df.loc[df.generator==generator]
    stds = []
    if baseline_df.empty == False:
        p = re.compile(f'{heuristic}\d')
        cells = [generators_dict[generator], heuristics_dict[heuristic]]
        for domain in domains:
            domain_df = baseline_df.query('domain==@domain')
            results = []
            nb_test_cases = len([h for h in domain_df['heuristic'].unique().tolist() if p.match(h)])
            for i in range(1, nb_test_cases + 1):
                reduced_df = domain_df.loc[domain_df.heuristic == f'{heuristic}{i}']
                nb_planners_killed = 0
                nb_planners_run = 0
                for planner in planners:
                    planner_df = reduced_df.query('planner==@planner').reset_index(drop=True)
                    # checks if results for the given planner exist
                    if planner_df.query('error==0').empty == False:
                        nb_planners_run += 1
                        if planner_df.query('failure==1').empty == False:
                            nb_planners_killed += 1
                results.append(100 * (nb_planners_killed / nb_planners_run) if nb_planners_run != 0 else 0)
            if results == []:
                cells.append('NaN')
                stds.append('NaN')
            else:
                cells.append(f'{np.mean(results):.2f}%')
                stds.append(f'{np.std(results):.2f}')
        f.write(','.join(cells) + '\n')
        f.write('std,,' + ','.join(stds) + '\n')
    f.close()

def save_csv_generators_score_compact(filepath: str, metric: str, planners: 'list[str]', heuristic: str):
    """
    saves a csv where each :
        - columns headers are domains
        - rows headers are generators
        - cells are (properly weighted) means of the generators scores.

    Only the configurations matching 'heuristic' (or 'heuristic\d') are considered
    and test cases are only selected among 'planners'. Metric has to be either 'failure' or 'error'
    """
    if metric not in ['failure', 'error']:
        print(f'wrong metric parameter ({metric})')
        return

    f = open(filepath, mode='w')
    # writes header
    f.write(',' + ','.join(domains) + '\n')

    p = re.compile(f'{heuristic}\d')

    for generator in generators:
        # cells of a line
        cells = [generators_dict[generator]]
        generators_values = []
        generators_weights = []
        for domain in domains:
            domain_df = df.query('generator==@generator & domain==@domain')
            problems_of_domains = domain_df['problem'].unique().tolist()
            problems_values = { p: [] for p in problems_of_domains }
            problems_weights = { p: [] for p in problems_of_domains }
            domain_values = []
            domain_weights = []
            # considers each problem...
            for problem in problems_of_domains:
                problem_df = domain_df.loc[domain_df.problem==problem]
                nb_test_cases_for_heuristic = len([h for h in problem_df['heuristic'].unique().tolist() if p.match(h)])
                # computes for each problem the generators scores with 'planners'
                if nb_test_cases_for_heuristic == 0:
                    # compute with heuristic
                    heuristic_df = problem_df.loc[(problem_df.heuristic == heuristic) & (problem_df.planner.isin(planners))]
                    if heuristic_df.empty == False:
                        result = generator_score(heuristic_df, metric)
                        problems_values[problem].append(np.mean(result))
                        problems_weights[problem].append(len(result))
                else:
                    results = []
                    weights = []
                    # iterates with 'heuristic\d'
                    for i in range(1, nb_test_cases_for_heuristic + 1):
                        heuristic_df = problem_df.loc[(problem_df.heuristic == f'{heuristic}{i}') & (problem_df.planner.isin(planners))]
                        if heuristic_df.empty == False:
                            result = generator_score(heuristic_df, metric)
                            results.append(np.mean(result))
                            weights.append(len(result))
                    if results != []:
                        problems_values[problem].append(np.mean(results))
                        problems_weights[problem].append(np.mean(weights))
                if problems_values[problem] != []:
                    domain_values.append(np.average(problems_values[problem], weights=problems_weights[problem]))
                    domain_weights.append(np.mean(problems_weights[problem]))
                    # print(f'{np.average(problems_values[problem], weights=problems_weights[problem])}% {np.mean(problems_weights[problem])}')
            if domain_values != []:
                generators_values.append(np.average(domain_values, weights=domain_weights))
                generators_weights.append(len(generators_values))
                cells.append(f'{np.average(domain_values, weights=domain_weights):.2f}% +/- {np.std(domain_values):.2f}')
            else:
                cells.append('NaN')
        f.write(','.join(cells) + '\n')
    f.close()

def save_chart_heuristics_score(filepath: str, generators: 'list[str]', planners: 'list[str]'):
    """
    saves a single chart which we have for each domain the mean scores (with their std over the problems) for all the heuristics.
    The x values are thus the domains and every heuristic has its own bars.
    """
    p = re.compile(f'h_random\d')

    # finds the best generator
    best_generators = {}
    for domain in domains:
        results = []
        for generator in generators:
            reduced_df = df.loc[(df.domain == domain) & (df.generator == generator) & (df.heuristic == 'h_zero') & (df.planner.isin(planners))]
            if reduced_df.empty == False:
                results.append(100 * (len(reduced_df[reduced_df['failure'] == 1]) / len(reduced_df)))
        if results == []:
            print(f'no result for {domain} {generator}')
        best_generators[domain] = generators[results.index(max(results))]
        print(f'best generator on {domain} is {best_generators[domain]}')

    # scores for each heuristic for all domains
    means = { h: [] for h in heuristics }
    stds = { h: [] for h in heuristics }

    for domain in domains:
        generator = best_generators[domain]
        domain_df = df.loc[(df.domain==domain) & (df.generator==generator)]
        problems_of_domains = domain_df['problem'].unique().tolist()
        nb_results_h_random = 1 + len([h for h in domain_df['heuristic'].unique().tolist() if p.match(h)])
        heuristics_values = { h: [] for h in heuristics }
        heuristics_weights = { h: [] for h in heuristics }
        for problem in problems_of_domains:
            problem_df = domain_df.loc[domain_df.problem==problem]
            for heuristic in heuristics:
                if heuristic != 'h_random':
                    heuristic_df = problem_df.loc[(problem_df.heuristic == heuristic) & (problem_df.planner.isin(planners))]
                    if heuristic_df.empty == False:
                        result = heuristic_score(heuristic_df)
                        if result != []:
                            heuristics_values[heuristic].append(np.mean(result))
                            heuristics_weights[heuristic].append(len(result))
                else:
                    results = []
                    weights = []
                    # iterates with 'h_random\d' to compute the mean score for this single test case
                    for i in range(1, nb_results_h_random):
                        h = f'{heuristic}{i}'
                        heuristic_df = problem_df.loc[(problem_df.heuristic == h) & (problem_df.planner.isin(planners))]
                        if heuristic_df.empty == False:
                            result = heuristic_score(heuristic_df)
                            if result != []:
                                results.append(np.mean(result))
                                weights.append(len(result))
                    if results != []:
                        heuristics_values[heuristic].append(np.mean(result))
                        heuristics_weights[heuristic].append(len(result))
        for k in heuristics_values.keys():
            if heuristics_values[k] != [] and heuristics_weights[k] != []:
                means[k].append(np.average(heuristics_values[k], weights=heuristics_weights[k]))
                stds[k].append(np.std(heuristics_values[k]))
            else:
                print(f'no result for ranking heuristic {k}')
                means[k].append(0)
                stds[k].append(0)
    
    fig, ax = plt.subplots()
    bar_width = 0.12
    x = np.arange(len(domains))
    
    for i in range(len(heuristics)):
        heuristic = heuristics[i]
        x_offset = (i - len(heuristics) / 2) * bar_width + bar_width / 2
        ax.bar(x + x_offset, means[heuristic], yerr=stds[heuristic], width=bar_width, label=heuristics_dict_latex[heuristic]) #, edgecolor = 'black')

    tmp = []
    for d in domains:
        # shortens some domain names
        if d == 'movie-strips':
            tmp.append('movie')
        elif d == 'newspapers':
            tmp.append('news')
        else:
            tmp.append(d)

    ax.set_xticks(x, tmp)
    ax.set_ylabel('Number of executed test cases')
    plt.legend(loc='upper left')
    plt.grid(axis = 'y')
    plt.tight_layout()
    plt.savefig(filepath)

def update_domains_dict(df: pd.DataFrame): 
    global domains
    global domains_dict
    domains_dict = {}
    domains = df['domain'].unique()
    for domain in domains:
        domains_dict[domain] = []
        problems = df['problem'].unique()
        for problem in problems:
            if df.query('domain==@domain & problem==@problem').empty == False:
                domains_dict[domain].append(problem)

#######################################################
## SCRIPT SECTION / MAIN
#######################################################

planners_dict = {
    # faulty competitors
    'dfs_first_solution': 'dfs',
    'a_star_mutant1_add': 'a*1_add',
    'a_star_mutant1_diff': 'a*1_diff',
    'a_star_mutant1_plus': 'a*1_plus',
    'a_star_mutant1_length': 'a*1_length',
    'a_star_mutant2_add': 'a*2_add',
    'a_star_mutant2_diff': 'a*2_diff',
    'a_star_mutant2_plus': 'a*2_plus',
    'a_star_mutant2_length': 'a*2_length',
    'a_star_mutant3_add': 'wa*_add',
    'a_star_mutant3_diff': 'wa*_diff',
    'a_star_mutant3_plus': 'wa*_plus',
    'a_star_mutant3_length': 'wa*_length',
    # real world planners
    'bjolp': 'BJOLP',
    'fdss1': 'FDSS1',
    'fdss2': 'FDSS2',
    'lmcut': 'LM-Cut',
    'merge_shrink': 'Merge&Shrink',
    'planningdomains': 'PD solver',
    'webplanner': 'WebPlanner',
    'astar_add': 'A*_add',
    'astar_ff': 'A*_ff',
    'wastar_add': 'WA*_add',
    'wastar_ff': 'WA*_ff'
}

heuristics_dict = {
    'h_zero': 'h_zero',
    'h_cost': 'h_cost',
    'h_reversed_cost': 'h_rev_cost',
    'h_distance_with_i': 'h_dist_i',
    'h_distance_with_g': 'h_dist_g',
    'h_random': 'h_random'
}

generators_dict = {
    'generator1': 'forw_local',
    'generator2': 'forw_path',
    'generator3' : 'backw_local',
    'generator4' : 'backw_path',
    'generator5' : 'forw_random'
}

heuristics_dict_latex = {
    'h_zero': '$H_0$',
    'h_cost': '$H_{cost}$',
    'h_reversed_cost': '$H_{\overline{cost}}$',
    'h_distance_with_i': '$H_i$',
    'h_distance_with_g': '$H_g$',
    'h_random': '$H_{random}$'
}

generators = ['generator1', 'generator2', 'generator3', 'generator4', 'generator5']
heuristics = ['h_cost', 'h_distance_with_g', 'h_distance_with_i', 'h_reversed_cost', 'h_zero', 'h_random']
real_world_planners = ['bjolp', 'fdss1', 'fdss2', 'lmcut', 'merge_shrink', 'planningdomains', 'webplanner', 'astar_add', 'astar_ff', 'wastar_add', 'wastar_ff']
faulty_competitors = ['a_star_mutant1_add', 'a_star_mutant1_diff', 'a_star_mutant1_length', 'a_star_mutant1_plus', 'a_star_mutant2_add', 'a_star_mutant2_diff', 'a_star_mutant2_length', 'a_star_mutant2_plus', 'a_star_mutant3_add', 'a_star_mutant3_diff', 'a_star_mutant3_length', 'a_star_mutant3_plus', 'dfs_first_solution']

df = pd.read_csv('results.csv', header=0)
domains_dict = {}
domains = ['blocks', 'gripper', 'hanoi', 'miconic', 'movie-strips', 'newspapers', 'travel'] # does not consider depot nor rover domains to begin with
problems = df['problem'].unique()
planners = df['planner'].unique()
optimal_planners = []
faulty_planners = []

for domain in domains:
    domains_dict[domain] = []
    problems = df['problem'].unique()
    for problem in problems:
        if df.query('domain==@domain & problem==@problem').empty == False:
            domains_dict[domain].append(problem)

for planner in planners:
    nb_failures = len(df.query('planner==@planner & failure==1 & error==0'))
    if nb_failures != 0:
        faulty_planners.append(planner)
    else:
        optimal_planners.append(planner)

print(f'optimal planners ({len(optimal_planners)}) :', optimal_planners)
print(f'faulty_planners ({len(faulty_planners)}) :', faulty_planners)

print('computing results...')
save_csv_non_optimal_score('mutation_scores.csv', faulty_competitors)
print('mutation score done')
# csv_to_latex('mutation_scores.csv', [0, 1])
save_chart_heuristics_score('heuristics_chart.png', ['generator1', 'generator2', 'generator3', 'generator4'], faulty_competitors)
print('heuristics chart done')
save_csv_generators_score_compact('generators_scores.csv', 'failure', faulty_competitors, 'h_zero')
print('generator score (failure) done')
# csv_to_latex('generators_scores.csv', [0])

# updates the domains considered (to include the depot and rover domains)
update_domains_dict(df) 

save_csv_non_optimal_detection('non_optimal_detection.csv', df, real_world_planners, 'generator1', 'h_zero')
print('detection on real-world planners done')
# csv_to_latex('non_optimal_detection.csv', [0], True)