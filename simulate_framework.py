import subprocess
import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import multiprocessing
import random

######################################################################################################################################################
############################################################## SETTING ###############################################################################


JSON_SOURCE_RESULTS_COSTS_FP = 'source_results/source_results.json'
PROBLEMS = ['airport06', 'airport07', 'blocks01', 'blocks02', 'blocks03', 'gripper01', 'miconic02', 'miconic03', 'openstacks01', 'pegsol04', 'pegsol05', 'pegsol06', 'psr-small03', 'psr-small06', 'psr-small08', 'psr-small09', 'tpp03', 'transport01', 'travel02']
HEURISTICS = {
    'hmax': 'h_max',
    'hdiff': 'h_diff',
    'hlength': 'h_state_length',
    'hi': 'h_distance_with_i',
    'hg': 'h_distance_with_g',
    'hnba': 'h_nb_actions'
}
source_results_costs = {}
CONFIGURATIONS = [('f5', 'hi'), ('f5', 'hlength'), ('f5', 'hnba'), ('f2', 'hdiff'), ('f2', 'hlength'), ('f2', 'hnba'), ('f4', 'hg'), ('f4', 'hdiff'), ('f4', 'hi'), ('f4', 'hlength'), ('f4', 'hmax'), ('f4', 'hnba'), ('f3', 'hg'), ('f3', 'hdiff'), ('f3', 'hi'), ('f3', 'hlength'), ('f3', 'hmax'), ('f3', 'hnba'), ('f6', 'hg'), ('f6', 'hi'), ('f6', 'hlength'), ('f6', 'hnba')]
PLANNERS = [f'{s}_{h}' for (s, h) in CONFIGURATIONS]
PROBLEM_REGEX = re.compile('(.+)(\d\d)')
GENERATORS_LATEX = {
    # static result keys
    'mutant': '$MorphinPlan$',
    'bfs': '$NoSelect$',
    'random': '$RandomSelect$',
    'random_std': '$S^{std}_{random}$',
    # dynamic result keys
    'walks': '$RandomWalks$', # main_fd_results
    'walks_std': '$R^{std}_{walks}$', # main_fd_results
    'a': 'a!!...'
}
DETERMINISTIC_STATIC_GENERATORS = ['bfs'] # ['min_dist_i', 'min_dist_g', 'max_dist_i', 'max_dist_g', 'bfs', 'mutant', 'select_mutants_killers']
NB_RANDOM_WALKS_REPETITIONS = 3
NB_RANDOM_REPETITIONS = 10
NB_TESTS = 10
NB_THREADS = 3
NO_DIGIT_REGEX = re.compile('(\D+)')

f = open(JSON_SOURCE_RESULTS_COSTS_FP, 'r')
source_results_costs = json.loads(f.read())
f.close()


######################################################################################################################################################
############################################################## LOW ###################################################################################


def get_arguments(configurations: list[tuple[str, str]], problems: list[str], n: int, generators: list[str]) -> list[tuple[str, str, tuple[str, str], int, str, str, list[str]]]:
    args = []
    for problem in problems:
        m = PROBLEM_REGEX.match(problem)
        domain_name = m.group(1)
        i = m.group(2)
        domain = 'domain' if 'domain.pddl' in os.listdir(f'benchmarks/{domain_name}') else f'domain{i}'
        for (s, h) in configurations:
            result_filename = f'results/{s}_{h}_{problem}.csv'
            output_filename = f'tmp/{s}_{h}_{problem}.txt'
            arg = (f'benchmarks/{domain_name}/{domain}.pddl', f'benchmarks/{domain_name}/task{i}.pddl', (s, h), n, result_filename, output_filename, generators)
            args.append(arg)
    return args


def cache_problem(domain_filename: str, problem_filename: str, output: str, configurations: list[tuple[str, str]]) -> None:
    """
    Runs the framework to create the .csv cache file of the given problem.
    Basically, the framework applies metamophic testing on all given configurations with all the nodes cached as follow-up test cases.
    The results are then saved and exported in a .csv file. It is time and resources consuming but it avoids any other mutant planner execution as
    the future selected states and their related results will be based with such a .csv file.
    """
    planners_commands = []
    for (s, h) in configurations:
        planners_commands.append(f'"planners/prolog_planner.exe mutated_astar-{s} {HEURISTICS[h]}"')
    command = f'main.exe --test {domain_filename} {problem_filename} {output} {" ".join(planners_commands)}'
    os.system(command)
    print('done.')


def run_framework_prolog_planner(domain_filename: str, problem_filename: str, configuration: tuple[str, str], nb_tests: int, result_filename: str, output_filename: str, generators: list[str]) -> None:
    """
    Runs a prolog_planner configuration, defined by a search and a heuristic, and returns the execution time.
    """
    planner_command = f'"planners/prolog_planner.exe mutated_astar-{configuration[0]} {HEURISTICS[configuration[1]]}"'
    command = f'main.exe {domain_filename} {problem_filename} {planner_command} {nb_tests} {result_filename} {output_filename} {" ".join(generators)}'
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL)
    except:
        print(f'{configuration[0]}_{configuration[1]} error with args: {domain_filename} {problem_filename} {nb_tests} {generators}.')


def run_framework_fd_planner(domain_filename: str, problem_filename: str, configuration: tuple[str, str], nb_tests: int, result_filename: str, output_filename: str, generators: list[str]) -> None:
    """
    Runs a fast-downward configuration, defined by a search and a heuristic, and returns the execution time.
    """
    planner_command = f'"python planners/fd_planner.py {configuration[0]} {configuration[1]}"'
    command = f'main.exe {domain_filename} {problem_filename} {planner_command} {nb_tests} {result_filename} {output_filename} {" ".join(generators)}'
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL)
    except:
        print(f'{configuration[0]}_{configuration[1]}  error with args: {domain_filename} {problem_filename} {nb_tests} {generators}.')


def simulate_framework(domain_filename: str, problem_filename: str, nb_tests: int, generator: str, result_arg: str) -> list[int]:
    """
    Runs statically the framework, which returns a list of nodes indexes (array of integers)
    """
    command = f'main.exe --{generator} {domain_filename} {problem_filename} {nb_tests} {result_arg}'
    # print(command)
    indexes = []
    try:
        p = subprocess.run(command, capture_output=True)
        indexes = [int(i) for i in p.stdout.decode().splitlines()]
    except:
        print(f'simulate_framework error with args: --{generator} {domain_filename} {problem_filename} {nb_tests} {result_arg}.')
    return indexes


def source_result_filepath(configuration: tuple[str, str], problem: str) -> str:
    return f'source_results/{problem}_{configuration[0]}_{configuration[1]}.txt'


def problem_result_filepath(problem: str) -> str:
    m = PROBLEM_REGEX.match(problem)
    domain = m.group(1)
    task = m.group(2)
    return f'cache/{domain}_task{task}.csv'


def make_mt_cache_file_from_csv(problem: str, configurations: list[tuple[str, str]]) -> None:
    """
    Creates a mutant selection cache file (for future dynamic executions of the framework) from a set of configurations.
    Their results are sourced from current .csv cache file.
    """
    csv_cache_filepath = problem_result_filepath(problem)
    txt_cache_filepath = f'{csv_cache_filepath.split(".")[0]}.txt'
    mut_txt_cache_filepath = f'{csv_cache_filepath.split(".")[0]}_mt.txt'
    df = pd.read_csv(csv_cache_filepath)
    select_df = df.loc[df.failure==1]
    select_df = pd.concat([select_df.loc[(select_df.search==s) & (select_df.heuristic==h)] for (s, h) in configurations], ignore_index=True)[['node_index', 'failure']].groupby(['node_index'], as_index=False).count()
    select_df.sort_values(by='failure', ascending=False, inplace=True)
    indexes = select_df['node_index'].tolist()
    f = open(txt_cache_filepath, 'r')
    nodes = f.readlines()
    f.close()
    f = open(mut_txt_cache_filepath, 'w')
    [f.write(nodes[i]) for i in indexes]
    f.close()


######################################################################################################################################################
############################################################## SIMULATING FRAMEWORK ##################################################################


def result_problem_configs_selections(problem: str, configurations: list[tuple[str, str]], n: int) -> None:
    """
    Exports a .csv file of all deterministic select-based generators results for the given configurations on the problem.
    """
    result_fp = f'data/selection_generators_{problem}_{n}.csv'
    if os.path.exists(result_fp) and not pd.read_csv(result_fp).empty:
        print(f'{result_fp} already exists. Skipped.')
        return

    df = pd.read_csv(problem_result_filepath(problem))
    m = PROBLEM_REGEX.match(problem)
    domain = m.group(1)
    task = m.group(2)
    domain_fp = f'benchmarks/{domain}/domain.pddl' if 'domain.pddl' in os.listdir(f'benchmarks/{domain}') else f'benchmarks/{domain}/domain{task}.pddl'
    problem_fp = f'benchmarks/{domain}/task{task}.pddl'
    sub_config_dfs = []
    for configuration in configurations:
        planner = f'{configuration[0]}_{configuration[1]}'
        source_result_cost = source_results_costs[planner][problem]
        source_result_fp = source_result_filepath(configuration, problem)
        config_df = df.loc[(df.search==configuration[0]) & (df.heuristic==configuration[1])]
        # sub_gen_dfs contains the results of all generators for the configuration
        sub_gen_dfs = []
        for generator in DETERMINISTIC_STATIC_GENERATORS:
            nodes_indexes = simulate_framework(domain_fp, problem_fp, n, generator, source_result_cost if generator.endswith('i') or generator == 'bfs' else source_result_fp)
            retries = 10
            while len(nodes_indexes) == 0 and retries > 0:
                retries -= 1
                print(f'empty indexes for {generator} on {configuration} {problem}', f'({retries} left)')
                nodes_indexes = simulate_framework(domain_fp, problem_fp, n, generator, source_result_cost if generator.endswith('i') or generator == 'bfs' else source_result_fp)
            # pd.concat() is used instead of .loc[] + node_index.isin() to keep indexes ordering, thus allowing n scaling analysis later on
            gen_tmp = pd.concat([config_df.loc[config_df.node_index==i] for i in nodes_indexes], ignore_index=True)
            gen_tmp.insert(1, 'generator', generator, allow_duplicates=True)
            sub_gen_dfs.append(gen_tmp)
        # regroups each generator result and drops the useless columns
        config_tmp = pd.concat(sub_gen_dfs, ignore_index=True)
        config_tmp.drop(['node_index', 'search', 'heuristic'], axis=1, inplace=True)
        # adds the name of the planner / configuration / mutant
        config_tmp.insert(0, 'planner', planner, allow_duplicates=True)
        sub_config_dfs.append(config_tmp)
        print(f'{problem}\t{planner} done.')
    # regroups the results of all configurations
    result = pd.concat(sub_config_dfs, ignore_index=True)
    # adds the problem column
    result.insert(1, 'problem', problem, allow_duplicates=True)
    result.to_csv(result_fp, index=0)
    print(f'{problem} done.')


def result_problem_configs_random(problem:str, configurations: list[tuple[str, str]], n: int) -> None:
    """
    Exports a .csv file of the random state selection generator results for the given configurations on the problem.
    """
    result_fp = f'data/selection_random_{problem}_{n}.csv'
    if os.path.exists(result_fp) and not pd.read_csv(result_fp).empty:
        print(f'{result_fp} already exists. Skipped.')
        return

    df = pd.read_csv(problem_result_filepath(problem))
    m = PROBLEM_REGEX.match(problem)
    domain = m.group(1)
    task = m.group(2)
    domain_fp = f'benchmarks/{domain}/domain.pddl' if 'domain.pddl' in os.listdir(f'benchmarks/{domain}') else f'benchmarks/{domain}/domain{task}.pddl'
    problem_fp = f'benchmarks/{domain}/task{task}.pddl'
    sub_config_dfs = []
    for configuration in configurations:
        planner = f'{configuration[0]}_{configuration[1]}'
        source_result_cost = source_results_costs[planner][problem]
        config_df = df.loc[(df.search==configuration[0]) & (df.heuristic==configuration[1])]
        # sub_gen_dfs contains the results of all generators for the configuration
        sub_gen_dfs = []
        for i in range(NB_RANDOM_REPETITIONS):
            # the framework sets its random seed on time; so it waits a bit between each repetition
            time.sleep(1.0)
            nodes_indexes = simulate_framework(domain_fp, problem_fp, n, 'random', source_result_cost)
            retries = 10
            while len(nodes_indexes) == 0 and retries > 0:
                retries -= 1
                print(f'empty indexes for random selection simulation on {configuration} {problem}', f'({retries} left)')
                nodes_indexes = simulate_framework(domain_fp, problem_fp, n, 'random', source_result_cost)
            # print(f'indexes for {problem} {planner}', nodes_indexes)
            gen_tmp = pd.concat([config_df.loc[config_df.node_index==i] for i in nodes_indexes], ignore_index=True)
            gen_tmp.insert(1, 'generator', f'random{i}', allow_duplicates=True)
            sub_gen_dfs.append(gen_tmp)
        config_tmp = pd.concat(sub_gen_dfs, ignore_index=True)
        config_tmp.drop(['node_index', 'search', 'heuristic'], axis=1, inplace=True)
        config_tmp.insert(0, 'planner', planner, allow_duplicates=True)
        sub_config_dfs.append(config_tmp)
        print(f'{problem}\t{planner} done.')
    result = pd.concat(sub_config_dfs, ignore_index=True)
    result.insert(1, 'problem', problem, allow_duplicates=True)
    result.to_csv(result_fp, index=0)
    print(f'{problem} done.')


def result_problems_configs_mutants(problems: list[str], configurations: list[tuple[str, str]], mutants: list[tuple[str, str]], n: int) -> pd.DataFrame:
    sub_prob_dfs = []
    for problem in problems:
        df = pd.read_csv(problem_result_filepath(problem))
        select_df = df.loc[df.failure==1]
        select_df = pd.concat([select_df.loc[(select_df.search==s) & (select_df.heuristic==h)] for (s, h) in mutants], ignore_index=True)[['node_index', 'failure']].groupby(['node_index'], as_index=False).count()
        select_df.sort_values(by='failure', ascending=False, inplace=True)
        indexes = select_df['node_index'].head(n).tolist()
        # print(problem, f'{0 if select_df.empty else (100 * select_df["failure"].tolist()[0] / len(mutants)):.1f}%', len(indexes), indexes)
        if indexes == []:
            continue
        sub_config_dfs = []
        for configuration in configurations:
            planner = f'{configuration[0]}_{configuration[1]}'
            config_df = df.loc[(df.search==configuration[0]) & (df.heuristic==configuration[1])]
            config_tmp = pd.concat([config_df.loc[config_df.node_index==i] for i in indexes], ignore_index=True)
            config_tmp.insert(1, 'generator', 'mutant', allow_duplicates=True)
            config_tmp.drop(['node_index', 'search', 'heuristic'], axis=1, inplace=True)
            config_tmp.insert(0, 'planner', planner, allow_duplicates=True)
            sub_config_dfs.append(config_tmp)
        prob_df = pd.concat(sub_config_dfs, ignore_index=True)
        prob_df.insert(1, 'problem', problem, allow_duplicates=True)
        sub_prob_dfs.append(prob_df)
    result_df = pd.concat(sub_prob_dfs, ignore_index=True)
    return result_df


######################################################################################################################################################
############################################################## REGROUPING ############################################################################


def regroup_select_results(n: int) -> None:
    fp = f'data/selection_generators_{n}.csv'
    if os.path.exists(fp):
        print(f'{fp} already exists. Aborted.')
    else:
        df = pd.concat([pd.read_csv(f'data/{f}') for f in os.listdir('data') if f.startswith('selection_generators') and f.endswith(f'{n}.csv')], ignore_index=True)
        df.to_csv(fp, index=0)


def regroup_random_select_results(n: int) -> None:
    fp = f'data/selection_random_{n}.csv'
    if os.path.exists(fp):
        print(f'{fp} already exists. Aborted.')
    else:
        df = pd.concat([pd.read_csv(f'data/{f}') for f in os.listdir('data') if f.startswith('selection_random') and f.endswith(f'{n}.csv')], ignore_index=True)
        df.to_csv(fp, index=0)


def regroup_results(n: int) -> None:
    fp = f'data/selection_{n}.csv'
    if os.path.exists(fp):
        print(f'{fp} already exists. Aborted.')
    else:
        select_result_fp = f'data/selection_generators_{n}.csv'
        random_result_fp = f'data/selection_random_{n}.csv'
        if os.path.exists(select_result_fp) and os.path.exists(random_result_fp):
            df = pd.concat([pd.read_csv(select_result_fp), pd.read_csv(random_result_fp)], ignore_index=True)
            df.to_csv(fp, index=0)
        else:
            print('One and more result files are missing. Please build the results first.')


def regroup_mutant_results(n: int, clean_tmp: bool=True) -> pd.DataFrame:
    fp = f'mutant_generator_{n}.csv'
    filepaths = [f'tmp/{f}' for f in os.listdir('tmp') if f.startswith('selection_mutant') and f.endswith(f'{n}.csv')]
    df = pd.concat([pd.read_csv(fp) for fp in filepaths], ignore_index=True)
    df.to_csv(fp, index=0)
    if clean_tmp:
        for fp in filepaths:
            os.remove(fp)
    return df


def df_for_problems_configs(problems: list[str], valid_configs: list[tuple[str, str]], n: int, select_configs: list[tuple[str, str]]=None, include_random: bool=False) -> pd.DataFrame:
    """
    Returns a dataframe composed of the results of the mutant-based methodology and those of the other deterministic selection methods for the given sets of problems and planners.
    """
    planners = [f'{s}_{h}' for (s, h) in valid_configs]
    if select_configs == None:
        mutant_df = regroup_mutant_results(n)
    else:
        mutant_df = result_problems_configs_mutants(problems, valid_configs, select_configs, n)
    problems = [p for p in problems if p in mutant_df['problem'].unique().tolist()]

    gen_df = pd.read_csv(f'data/selection_{n}.csv') if include_random else pd.read_csv(f'data/selection_generators_{n}.csv')
    gen_df = gen_df.loc[(gen_df.problem.isin(problems)) & (gen_df.planner.isin(planners))]

    res_df = pd.concat([mutant_df, gen_df], ignore_index=True)
    return res_df


def merge_result_dataframe_latex(filepaths: list[str], filename: str) -> None:
    """
    Merges the result dataframes from filepaths parameter and exports their average values in .tex file.
    The dataframes are expected to be outputs from either dataframe_mutation_coverage function.
    It proceeds as follows:
        - The mean values as well as the standard deviations are computed seperately.
        - Each cell is then rendered as mean+/-std with latex styling.
    """
    df = pd.concat([pd.read_csv(filepath) for filepath in filepaths], ignore_index=True)
    df_mean = df.groupby(['problem'], sort=False).mean()
    df_std = df.groupby(['problem'], sort=False).std()
    problems = [p for p in df['problem'].unique().tolist() if p != 'mean']
    gens1 = [c for c in df_mean.columns if c in ['mutant', 'bfs']]
    gens2 = [c for c in df_mean.columns if c in ['random', 'walks']]
    gens2.sort()
    columns = gens1 + gens2
    data = {}
    for problem in problems:
        max_value = max(df_mean.loc[problem])
        # each cell is the average of the means with the standard deviation
        data[problem] = []
        for c in gens1:
            cell = f'{df_mean.at[problem, c]:.1f}$\pm${df_std.at[problem, c]:.1f}'
            data[problem].append(cell if df_mean.at[problem, c] != max_value else f'\textbf{{{cell}}}')
        for c in gens2:
            cell = f'({df_mean.at[problem, c]:.1f}$\pm${df_mean.at[problem, c + "_std"]:.1f})$\pm${df_std.at[problem, c]:.1f}'
            data[problem].append(cell if df_mean.at[problem, c] != max_value else f'\textbf{{{cell}}}')

    # for the mean line, each cell is the average of the previous means with the related standard deviation
    problem = 'mean'
    max_value = max(df_mean.loc[problem])
    data['mean'] = [f'{df_mean.at[problem, c]:.1f}$\pm${np.std(df_mean[c].head(len(problems))):.1f}' if df_mean.at[problem, c] != max_value else f'\textbf{{{df_mean.at[problem, c]:.1f}$\pm${np.std(df_mean[c].head(len(problems))):.1f}}}' for c in columns]

    df = pd.DataFrame.from_dict(data, orient='index', columns=[GENERATORS_LATEX[c] for c in columns])
    df.to_latex(filename, escape=False)


def merge_n_scaling_result_dataframe_latex(filepaths: list[str], filename: str) -> None:
    """
    Merges the result dataframes from filepaths parameter and exports their average values in .tex file.
    The dataframes are expected to be outputs from the n_scaling_mutation_coverage function.
    It proceeds as follows:
        - The mean values as well as the standard deviations are computed seperately.
        - Each cell is then rendered as mean+/-std with latex styling.
    """
    df = pd.concat([pd.read_csv(filepath) for filepath in filepaths], ignore_index=True)
    n_values = df['N'].unique().tolist()
    df_mean = df.groupby(['N'], sort=False).mean()
    df_std = df.groupby(['N'], sort=False).std()
    generators = [c for c in ['mutant', 'bfs', 'random', 'walks'] if c in df_mean.columns]
    if filename.endswith('.tex'):
        data = {f'$N_{{{n}}}$': [f'{df_mean.at[n, g]:.1f}$\pm${df_std.at[n, g]:.1f}' for g in generators] for n in n_values}
        df = pd.DataFrame.from_dict(data, orient='index', columns=[GENERATORS_LATEX[g] for g in generators])
        df.to_latex(filename, escape=False)
    elif filename.endswith('.png'):
        # plotting
        _, ax = plt.subplots()
        x = np.arange(1, len(n_values) + 1)
        for g in generators:
            y = [df_mean.at[n, g] for n in n_values]
            err = [df_std.at[n, g] for n in n_values]
            ax.plot(x, [df_mean.at[n, g] for n in n_values], label=GENERATORS_LATEX[g])
            ax.fill_between(x, (np.array(y) - np.array(err)).tolist(), (np.array(y) + np.array(err)).tolist(), alpha=0.2)
        ax.set_xticks(x)
        ax.set_xlim(1, len(n_values))
        ax.grid(True)
        ax.set_xlabel('$N_{max}$')
        ax.set_ylabel('Average mutation score [%]')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(generators), fancybox=True, shadow=True)
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
    else:
        print('unsupported extension file.')


def get_framework_result_dataframe(filepath: str) -> pd.DataFrame:
    """
    Returns a well formed dataframe from a .csv result file.
    """
    p = re.compile('.+/(.+)_(.+)_(.+).csv')
    m = p.match(filepath)
    planner_name = f'{m.group(1)}_{m.group(2)}'
    problem_name = m.group(3)

    f = open(filepath, 'r')
    header = f.readline().split(',')
    source_result_cost = header[3]
    f.close()

    df = pd.read_csv(filepath, header=1)
    df.drop(['execution_time(sec)'], axis=1, inplace=True)
    df.insert(0, 'planner', planner_name, allow_duplicates=True)
    df.insert(1, 'problem', problem_name, allow_duplicates=True)
    df.insert(len(df.columns), 'source_result', source_result_cost, allow_duplicates=True)
    return df


def regroup_framework_result_dataframes(filepaths: list[str], filename: str=None) -> pd.DataFrame:
    """
    Retrieves all the results and regroup them into a single dataframe.
    """
    df = pd.concat(map(get_framework_result_dataframe, filepaths))
    if filename != None:
        df.to_csv(filename, index=0)
    return df


######################################################################################################################################################
############################################################## MUTATION COVERAGE #####################################################################


def plot_mutation_coverage(df: pd.DataFrame, filename: str) -> None:
    """
    Saves a chart that reports either the percentage of killed mutants or their (raw) number for each solution on each problem.
    It works well with results to average.
    """
    df = df.loc[df.error==0]
    # results to average detection
    results_to_average = False
    for g in df['generator'].unique().tolist():
        if g[-1].isdigit():
            results_to_average = True
            break
    generators = [g for g in df['generator'].unique().tolist() if not g[-1].isdigit()]
    planners = df['planner'].unique().tolist()
    problems = df['problem'].unique().tolist()
    scores = {g: [] for g in generators}
    if results_to_average:
        generators_to_average = list(set([NO_DIGIT_REGEX.match(g).group(1) for g in df['generator'].unique().tolist() if g[-1].isdigit()]))
        nb_repetitions = {g: len([x for x in df['generator'].unique().tolist() if x.startswith(g) and x[-1].isdigit()]) for g in generators_to_average}
        for generator_to_average in generators_to_average:
            scores[generator_to_average] = []
    for problem in problems:
        problem_df = df.loc[df.problem==problem]
        for generator in generators:
            generator_df = problem_df.loc[problem_df.generator==generator]
            if generator_df.empty:
                # print(f'no data for {problem} {generator}')
                scores[generator].append(0)
            else:
                score = 0
                nb_planners = 0
                for planner in planners:
                    planner_df = generator_df.loc[generator_df.planner==planner]
                    if not planner_df.empty:
                        # the planner is supposed to be detectable
                        nb_planners +=1
                        if not planner_df.loc[planner_df.failure==1].empty:
                            score += 1
                    # else:
                    #     print(f'no data for {problem} {generator} {planner}')
                scores[generator].append(100 * score / nb_planners)
        if results_to_average:
            for g in generators_to_average:
                nb_repetition = nb_repetitions[g]
                g_scores = []
                for i in range(nb_repetition):
                    generator = f'{g}{i}'
                    generator_df = problem_df.loc[problem_df.generator==generator]
                    if not generator_df.empty:
                    #     print(f'no data for {problem} {generator}')
                    # else:
                        score = 0
                        nb_planners = 0
                        for planner in planners:
                            planner_df = generator_df.loc[generator_df.planner==planner]
                            if not planner_df.empty:
                                # the planner is supposed to be detectable
                                nb_planners +=1
                                if not planner_df.loc[planner_df.failure==1].empty:
                                    score += 1
                            # else:
                            #     print(f'no data for {problem} {generator} {planner}')
                        g_scores.append(100 * score / nb_planners)
                scores[g].append(np.mean(g_scores) if g_scores != [] else 0) # [] means no data
    # removing unrelevant data
    relevant_problems = []
    indexes_to_remove = []
    for i in range(len(problems)):
        data = np.array([scores[g][i] for g in scores.keys()])
        if data.sum(dtype=int) != 0:
            relevant_problems.append(problems[i])
        else:
            # print(f'{problems[i]} data is unrelevant: index {i} has to be removed.')
            indexes_to_remove.append(i)
    if indexes_to_remove != []:
        for g in scores.keys():
            scores[g] = np.delete(scores[g], indexes_to_remove).tolist()
    # plotting
    _, ax = plt.subplots()
    bar_width = 0.1
    x = np.arange(len(relevant_problems))
    gens = list(scores.keys()) # gens should include 'random' if necessary
    for i in range(len(gens)):
        generator = gens[i]
        x_offset = (i - len(gens) / 2) * bar_width + bar_width / 2
        ax.bar(x + x_offset, scores[generator], width=bar_width, label=GENERATORS_LATEX[generator])
    ax.set_xticks(x)
    ax.set_xticklabels(relevant_problems, rotation=30)
    ax.set_ylabel('Detection coverage [%]')
    ax.legend(loc='upper left')

    # puts a legend to the right of the current axis
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(filename, dpi=200)


def dataframe_mutation_coverage(df: pd.DataFrame, filename: str=None) -> pd.DataFrame:
    """
    Computes the mutation coverage on the input dataframe and returns its result as a new one. Columns are the generators and each row is a problem.
    The mean is added at the end of the table. It works well with results to average.
    Depending on the extension of the filename parameter, it exports either the dataframe (.csv file) or its table representation (.tex file).
    """
    df = df.loc[df.error==0]
    # results to average detection
    results_to_average = False
    for g in df['generator'].unique().tolist():
        if g[-1].isdigit():
            results_to_average = True
            break
    generators = [g for g in df['generator'].unique().tolist() if not g[-1].isdigit()]
    planners = df['planner'].unique().tolist()
    problems = df['problem'].unique().tolist()
    scores = {g: [] for g in generators}
    if results_to_average:
        generators_to_average = list(set([NO_DIGIT_REGEX.match(g).group(1) for g in df['generator'].unique().tolist() if g[-1].isdigit()]))
        nb_repetitions = {g: len([x for x in df['generator'].unique().tolist() if x.startswith(g) and x[-1].isdigit()]) for g in generators_to_average}
        for generator_to_average in generators_to_average:
            scores[generator_to_average] = []
            scores[f'{generator_to_average}_std'] = []
    for problem in problems:
        problem_df = df.loc[df.problem==problem]
        for generator in generators:
            generator_df = problem_df.loc[problem_df.generator==generator]
            if generator_df.empty:
                # print(f'no data for {problem} {generator}')
                scores[generator].append(0)
            else:
                score = 0
                nb_planners = 0
                for planner in planners:
                    planner_df = generator_df.loc[generator_df.planner==planner]
                    if not planner_df.empty:
                        # the planner is supposed to be detectable
                        nb_planners +=1
                        if not planner_df.loc[planner_df.failure==1].empty:
                            score += 1
                    # else:
                        # print(f'no data for {problem} {generator} {planner}')
                scores[generator].append(100 * score / nb_planners)
        if results_to_average:
            for g in generators_to_average:
                nb_repetition = nb_repetitions[g]
                g_scores = []
                for i in range(nb_repetition):
                    generator = f'{g}{i}'
                    generator_df = problem_df.loc[problem_df.generator==generator]
                    if not generator_df.empty:
                        score = 0
                        nb_planners = 0
                        for planner in planners:
                            planner_df = generator_df.loc[generator_df.planner==planner]
                            if not planner_df.empty:
                                # the planner is supposed to be detectable
                                nb_planners +=1
                                if not planner_df.loc[planner_df.failure==1].empty:
                                    score += 1
                            else:
                                print(f'no data for {problem} {generator} {planner}')
                        g_scores.append(100 * score / nb_planners)
                scores[g].append(np.mean(g_scores) if g_scores != [] else 0) # [] means no data
                scores[f'{g}_std'].append(np.std(g_scores) if g_scores != [] else 0) # [] means no data
    df = pd.DataFrame(data=scores, index=problems)
    df.loc['mean'] = [df[c].mean() for c in df.columns]
    if filename != None:
        if filename.endswith('.tex'):
            df.to_latex(filename, float_format='{:0.1f}\%'.format, header=[GENERATORS_LATEX[c] for c in df.columns], escape=False)
        elif filename.endswith('.csv'):
            df.to_csv(filename, index_label='problem')
        else:
            print('filename extension not supported.')
    return df


######################################################################################################################################################
############################################################## STATE SELECTION EFFICIENCY ############################################################


def plot_overall_performance(df: pd.DataFrame, filename: str) -> None:
    """
    Saves a chart that describes the efficiency of each solution by reporting the percentage of successful follow-up test cases.
    It works well with results to average.
    """
    # results to average detection
    results_to_average = False
    for g in df['generator'].unique().tolist():
        if g[-1].isdigit():
            results_to_average = True
            break
    generators = [g for g in df['generator'].unique().tolist() if not g[-1].isdigit()]
    if results_to_average:
        generators_to_average = list(set([NO_DIGIT_REGEX.match(g).group(1) for g in df['generator'].unique().tolist() if g[-1].isdigit()]))
        nb_repetitions = {g: len([x for x in df['generator'].unique().tolist() if x.startswith(g) and x[-1].isdigit()]) for g in generators_to_average}

    scores = []
    yerr = []
    for g in generators:
        g_df = df.loc[df.generator==g]
        g_scores = [100 * len(g_df.loc[(g_df.failure==1) & (g_df.problem==p)]) / len(g_df.loc[g_df.problem==p]) if not g_df.loc[g_df.problem==p].empty else 0 for p in PROBLEMS]
        scores.append(np.mean(g_scores))
        yerr.append(np.std(g_scores))
    if results_to_average:
        for g in generators_to_average:
            nb_repetition = nb_repetitions[g]
            g_scores = []
            for p in PROBLEMS:
                p_df = df.loc[df.problem==p]
                i_scores = [100 * len(p_df.loc[(p_df.generator==f'{g}{i}') & (p_df.failure==1)]) / len(p_df.loc[p_df.generator==f'{g}{i}']) if not p_df.loc[p_df.generator==f'{g}{i}'].empty else 0 for i in range(nb_repetition)]
                g_scores.append(np.mean(i_scores))
            scores.append(np.mean(g_scores))
            yerr.append(np.std(g_scores))
            generators.append(g)
    _, ax = plt.subplots()
    bar = ax.bar(generators, scores, yerr=yerr)
    ax.bar_label(bar)
    ax.set_xticks(ticks=np.arange(len(generators)), labels=[GENERATORS_LATEX[g] for g in generators])
    ax.set_ylabel('Rate of fault-revealing test cases [%]')
    plt.savefig(filename, dpi=200)


######################################################################################################################################################
############################################################## N SCALING #############################################################################


def n_scaling_mutation_coverage(df: pd.DataFrame, n_max: int, filename: str=None) -> pd.DataFrame:
    """
    Computes the evolution of the mutation coverage with respect to the number of follow-up test cases on the input dataframe and returns its result as a new one.
    The scaling goes from 1 to n_max parameter. The results are shown in ascending order as index and columns are thus the generators.
    It works well with results to average.
    Depending on the extension of the filename parameter, it exports either the dataframe (.csv file) or its table representation (.tex file).
    """
    # results to average detection
    results_to_average = False
    for g in df['generator'].unique().tolist():
        if g[-1].isdigit():
            results_to_average = True
            break
    generators = [g for g in df['generator'].unique().tolist() if not g[-1].isdigit()]
    planners = df['planner'].unique().tolist()
    problems = df['problem'].unique().tolist()
    if results_to_average:
        generators_to_average = list(set([NO_DIGIT_REGEX.match(g).group(1) for g in df['generator'].unique().tolist() if g[-1].isdigit()]))
        nb_repetitions = {g: len([x for x in df['generator'].unique().tolist() if x.startswith(g) and x[-1].isdigit()]) for g in generators_to_average}
    d = {}
    nb_mutants = len(planners)
    for generator in generators:
        g_df = df.loc[df.generator==generator]
        d[generator] = []
        for n in range(1, n_max + 1):
            scores = []
            for problem in problems:
                p_df = g_df.loc[g_df.problem==problem]
                nb_mutants_killed = 0
                for planner in planners:
                    n_df = p_df.loc[p_df.planner==planner].head(n)
                    if not n_df.empty:
                        nb_mutants_killed += 1 if not n_df.loc[n_df.failure==1].empty else 0
                scores.append(100 * nb_mutants_killed / nb_mutants)
            d[generator].append(np.mean(scores))
    if results_to_average:
        for g in generators_to_average:
            nb_repetition = nb_repetitions[g]
            d[g] = []
            for n in range(1, n_max + 1):
                scores = []
                for problem in problems:
                    problem_df = df.loc[df.problem==problem]
                    score = []
                    for i in range(nb_repetition):
                        gen_df = problem_df.loc[problem_df.generator==f'{g}{i}']
                        nb_mutants_killed = 0
                        for planner in planners:
                            n_df = gen_df.loc[gen_df.planner==planner].head(n)
                            if not n_df.empty:
                                nb_mutants_killed += 1 if not n_df.loc[n_df.failure==1].empty else 0
                        score.append(100 * nb_mutants_killed / nb_mutants)
                    scores.append(np.mean(score))
                d[g].append(np.mean(scores))
    if results_to_average:
        for g in generators_to_average:
            generators.append(g)
    res_df = pd.DataFrame(data=d, index=[i for i in range(1, n_max + 1)])
    if filename != None:
        if filename.endswith('.tex'):
            pd.DataFrame(data=d, index=[f'$N_{{{i}}}$' for i in range(1, n_max + 1)]).to_latex(filename, float_format='{:0.1f}\%'.format, header=[GENERATORS_LATEX[g] for g in generators], escape=False)
        elif filename.endswith('.csv'):
            res_df.to_csv(filename, index_label='N')
        elif filename.endswith('.png'):
            # plotting
            _, ax = plt.subplots()
            x = [i for i in range(1, n_max + 1)]
            gens = list(d.keys())
            for i in range(len(gens)):
                generator = gens[i]
                ax.plot(x, d[generator], label=GENERATORS_LATEX[generator])
            ax.set_xlabel('$N_{max}$')
            ax.set_ylabel('Average detection coverage [%]')
            ax.set_xticks(x)
            ax.set_xlim(1, n_max)
            ax.grid(True)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(gens), fancybox=True, shadow=True)
            plt.tight_layout()
            plt.savefig(filename, dpi=200)
        else:
            print('filename extension not supported.')
    return res_df


######################################################################################################################################################
############################################################## MAIN ##################################################################################


def main_test_mutants_selection_impact():
    """
    main() for studying the impact of various subsets of mutants on the methodology performance.
    """
    n = NB_TESTS
    problems = PROBLEMS
    configurations = CONFIGURATIONS
    # print(f'nb configurations: {len(configurations)}', f'nb problems: {len(problems)}')
    """
    Here, 22 mutants are available. We state the following propositions as 'intuitive':
        - In order to do proper mutation testing, the validation mutant set has to be at least of size 10.
        - For the mutant based state selection methodology to make sense, we want to reserve it at least 5 mutants.
    Thus, the experiments that are about to be executed here will involve validation mutant sets with sizes from 10 to 17, and selection mutants set sizes from 5 to 12.
    """
    nb_configurations = len(configurations)
    nb_experiments = 20
    validation_min_size = 10
    selection_min_size = 5

    validation_max_size = nb_configurations - selection_min_size

    pool = multiprocessing.Pool(processes=NB_THREADS)
    for i in range(nb_experiments):
        random.shuffle(configurations)
        validation_size = random.randint(validation_min_size, validation_max_size)
        selection_size = random.randint(selection_min_size, nb_configurations - validation_size) # all the mutants can be not used
        to_valid = configurations[:validation_size]
        to_select = configurations[validation_size:validation_size + selection_size]
        simulation_result_df = df_for_problems_configs(problems, to_valid, n, to_select, True) # results of the simulation of the framework
        # for the random_walks generator baseline, the framework needs to be dynamically executed
        random_walks_result_dfs = []
        for j in range(NB_RANDOM_WALKS_REPETITIONS):
            my_args = get_arguments(to_valid, problems, n, ['walks_generator'])
            print(f'iteration {j}: {len(my_args)} executions to be launched.')
            pool.starmap(run_framework_prolog_planner, my_args, chunksize=2)
            result_fps = list(map(lambda x: x[4], my_args))
            random_walks_result_df = regroup_framework_result_dataframes(result_fps)
            random_walks_result_df.replace(to_replace={'walks_generator' : f'walks{j}'}, inplace=True)
            random_walks_result_dfs.append(random_walks_result_df.copy())
            for result_fp in result_fps:
                os.remove(result_fp)
        random_walks_result_df = pd.concat(random_walks_result_dfs, ignore_index=True)
        random_walks_result_df.to_csv(f'results/random_walks_{i}_{validation_size}_{selection_size}.csv')

        result_df = pd.concat([simulation_result_df, random_walks_result_df], ignore_index=True)
        dataframe_mutation_coverage(result_df, f'results/coverage_{i}_{validation_size}_{selection_size}.csv')
        n_scaling_mutation_coverage(result_df, 10,  f'results/n_scaling_mutation_coverage_{i}_{validation_size}_{selection_size}.csv')
        result_df.to_csv( f'results/result_{i}_{validation_size}_{selection_size}.csv', index=0)

    coverage_fps = [f'results/{f}' for f in os.listdir('results') if f.startswith('coverage') and f.endswith('.csv')]
    n_scaling_mutation_coverage_fps = [f'results/{f}' for f in os.listdir('results') if f.startswith('n_scaling_mutation_coverage') and f.endswith('.csv')]
    result_fps = [f'results/{f}' for f in os.listdir('results') if f.startswith('result') and f.endswith('.csv')]
    merge_result_dataframe_latex(coverage_fps, f'results/coverage_{nb_experiments}_{n}.tex')
    merge_n_scaling_result_dataframe_latex(n_scaling_mutation_coverage_fps, f'results/n_scaling_mutation_coverage_{nb_experiments}_{n}.png')
    final_result_df = pd.concat([pd.read_csv(result_fp) for result_fp in result_fps], ignore_index=True)
    final_result_df.to_csv(f'results/final_results_{nb_experiments}_{n}.csv', index=0)
    plot_overall_performance(final_result_df, f'results/overall_efficiency_{nb_experiments}_{n}.png')


def main_build_deterministic_results():
    """
    main() for building all the deterministic results. They do not depend on anything so they just need to be run once.
    """
    my_args = [(problem, CONFIGURATIONS, NB_TESTS) for problem in PROBLEMS]
    print(f'{len(my_args)} executions are about to be launched.')
    pool = multiprocessing.Pool(processes=NB_THREADS)
    pool.starmap(result_problem_configs_selections, my_args, chunksize=3)
    regroup_select_results(NB_TESTS)


def main_build_random_results():
    """
    main() for building the random selection method results.
    It consists in running multiples times the framework on the set of problems and regrouping them in a dedicated file.
    """
    my_args = [(problem, CONFIGURATIONS, NB_TESTS) for problem in PROBLEMS]
    print(f'{len(my_args)} executions are about to be launched.')
    pool = multiprocessing.Pool(processes=NB_THREADS)
    pool.starmap(result_problem_configs_random, my_args, chunksize=3)
    regroup_random_select_results(NB_TESTS)


def main_fd_results():
    deterministic_generators = {
        'select_bfs': 'bfs',
        'select_mutants_killers': 'mutant'
    }
    # creates the cache files for all the problems considering all the mutants (as we test fast-downward planners here)
    for problem in PROBLEMS:
        make_mt_cache_file_from_csv(problem, CONFIGURATIONS)

    # defines the fast-downward settings for testing
    searches = ['astar', 'wastar']
    evals = ['ff', 'add', 'cea', 'cg', 'goalcount']
    fd_configurations = [(s, e) for s in searches for e in evals]
    result_dfs = []
    pool = multiprocessing.Pool(processes=NB_THREADS)

    # runs the deterministic results (once)
    my_args = get_arguments(fd_configurations, PROBLEMS, NB_TESTS, list(deterministic_generators.keys()))
    print(f'{len(my_args)} executions are about to be launched.')
    pool.starmap(run_framework_fd_planner, my_args, chunksize=3)

    # regroups the results and cleans the result subfiles
    result_fps = list(map(lambda x: x[4], my_args))
    deterministic_result_df = regroup_framework_result_dataframes(result_fps)
    deterministic_result_df.replace(to_replace=deterministic_generators, inplace=True)
    # saves the dataframe for safety
    deterministic_result_df.to_csv(f'results/deterministic_fd_results_{NB_TESTS}.csv', index=0)
    for result_fp in result_fps:
        os.remove(result_fp)
    result_dfs.append(deterministic_result_df)

    # runs the non-deterministic results NB_RANDOM_REPETITIONS times
    for i in range(NB_RANDOM_REPETITIONS):
        my_args = get_arguments(fd_configurations, PROBLEMS, NB_TESTS, ['select_random', 'walks_generator'])
        print(f'{len(my_args)} executions are about to be launched.')
        pool.starmap(run_framework_fd_planner, my_args, chunksize=3)
        result_fps = list(map(lambda x: x[4], my_args))
        regroup_framework_result_dataframes(result_fps, f'results/random_{i}_fd_results_{NB_TESTS}.csv')
        for result_fp in result_fps:
            os.remove(result_fp)
    # regroups the non-deterministic results
    for i in range(NB_RANDOM_REPETITIONS):
        random_result_fp = f'results/random_{i}_fd_results_{NB_TESTS}.csv'
        random_result_df = pd.read_csv(random_result_fp)
        # post-processes the results by renaming the names of the generator
        random_result_df.replace(to_replace={'select_random' : f'random{i}', 'walks_generator' : f'walks{i}'}, inplace=True)
        # ... thus overwrites every original result subfile
        random_result_df.to_csv(random_result_fp, index=0)
        result_dfs.append(random_result_df.copy())

    # concatenates all the results
    result_df = pd.concat(result_dfs, ignore_index=True)
    # saves the final result dataframe for safety, as it can then be used with plot_overall_performance() for example
    result_df.to_csv(f'results/fd_results_{NB_TESTS}.csv', index=0)
    dataframe_mutation_coverage(result_df, f'results/fd_results_{NB_TESTS}_coverage.csv')
    n_scaling_mutation_coverage(result_df, NB_TESTS,  f'results/fd_results_{NB_TESTS}_n_scaling_coverage.csv')
    # exports n scaling detection coverage results as a latex considering all the problems
    n_scaling_mutation_coverage(result_df, NB_TESTS,  f'results/fd_results_{NB_TESTS}_n_scaling_coverage.tex')
    plot_mutation_coverage(result_df, f'results/fd_results_{NB_TESTS}_coverage.png')

    # the results shown in the paper only consider relevant problems, i.e problems for which at least a detection happened
    paper_df = pd.concat([result_df.loc[result_df.problem==p] for p in result_df['problem'].unique().tolist() if not result_df.loc[(result_df.problem==p) & (result_df.failure==1)].empty], ignore_index=True)
    plot_mutation_coverage(paper_df, f'results/fd_results_{NB_TESTS}_coverage.png')
    n_scaling_mutation_coverage(paper_df, NB_TESTS,  f'results/fd_results_{NB_TESTS}_n_scaling_coverage.png')


def main_build_cache():
    args = []
    for problem in PROBLEMS:
        m = PROBLEM_REGEX.match(problem)
        domain_name = m.group(1)
        i = m.group(2)
        domain = 'domain' if 'domain.pddl' in os.listdir(f'benchmarks/{domain_name}') else f'domain{i}'
        arg = (f'benchmarks/{domain_name}/{domain}.pddl', f'benchmarks/{domain_name}/task{i}.pddl', f'tmp/{problem}.txt', CONFIGURATIONS)
        args.append(arg)

    print(f'{len(args)} executions are about to be launched.')
    pool = multiprocessing.Pool(processes=NB_THREADS)
    pool.starmap(cache_problem, args, chunksize=2)

    tmp_files = list(map(lambda x: x[2], args))
    for tmp_file in tmp_files:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


def update_cache(problem: str, indexes: list[str]):
    csv_cache_filepath = problem_result_filepath(problem)
    txt_cache_filepath = f'{csv_cache_filepath.split(".")[0]}.txt'
    mut_txt_cache_filepath = f'{csv_cache_filepath.split(".")[0]}_mt.txt'
    f = open(txt_cache_filepath, 'r')
    nodes = f.readlines()
    f.close()
    f = open(mut_txt_cache_filepath, 'w')
    [f.write(nodes[i]) for i in indexes]
    f.close()


def remove_cache(problem: str):
    filepath = f'{problem_result_filepath(problem).split(".")[0]}.txt'
    if os.path.exists(filepath):
        os.remove(filepath)
    else:
        print(f'file\t{filepath}\t not found.')


def generate_framework(problem):
    m = PROBLEM_REGEX.match(problem)
    domain_name = m.group(1)
    i = m.group(2)
    domain = 'domain' if 'domain.pddl' in os.listdir(f'benchmarks/{domain_name}') else f'domain{i}'
    domain_filename = f'benchmarks/{domain_name}/{domain}.pddl'
    problem_filename = f'benchmarks/{domain_name}/task{i}.pddl'
    command = f'main.exe --bfs {domain_filename} {problem_filename} 2 5000'
    print(command)
    try:
        subprocess.run(command, capture_output=True)
    except:
        print(f'generate_framework error with args: --bfs {domain_filename} {problem_filename} 2 5000.')


def barsurface(df, feature, filename):
    nx, ny = len(PROBLEMS), len(fd_configurations)
    x = range(nx)
    y = range(ny)
    data = np.zeros((nx, ny))
    for i in x:
        data[i] = [len(df.loc[(df.problem==PROBLEMS[i]) & (df.planner==f'{s}_{h}') & (df[feature]==1)]) for (s, h) in fd_configurations]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xx = np.array([[i] * ny for i in x]).ravel() # x coordinates of each bar
    yy = np.array([i for i in y] * nx) # y coordinates of each bar
    z = np.zeros(nx * ny) # z coordinates of each bar
    dx = np.ones(nx * ny) # length along x-axis of each bar
    dy = np.ones(nx * ny) # length along y-axis of each bar
    dz = data.ravel() # length along z-axis of each bar (height)
    ax.bar3d(xx, yy, z, dx, dy, dz)
    ax.set_xticks(x, PROBLEMS, rotation=30)
    ax.set_yticks(y, [f'{s}_{h}' for (s, h) in fd_configurations])
    plt.savefig(filename, dpi=200)


def faulty_oracle(df, name):
    """
    First intuition thought Friday 21/10.
    """
    df['faulty_oracle'] = df['result'] + df['node_cost'] - df['source_result']
    df.loc[df.faulty_oracle==0, name] = 0
    df.loc[~(df.faulty_oracle==0), name] = 1
    df = df.astype({name: int})
    return df


# exec(open('simulate_framework.py').read())
if __name__ == '__main__':
    print(f'number of problems: {len(PROBLEMS)}')
    print(f'number of configurations: {len(CONFIGURATIONS)}')


    # defines the fast-downward settings for testing
    searches = ['astar', 'wastar']
    evals = ['ff', 'add', 'cea', 'cg', 'goalcount']
    fd_configurations = [(s, e) for s in searches for e in evals]
    result_dfs = []
    pool = multiprocessing.Pool(processes=NB_THREADS)

    relevant_problems = ['airport07', 'blocks01', 'gripper01', 'miconic02', 'miconic03', 'openstacks01', 'pegsol04', 'pegsol05', 'pegsol06', 'tpp03', 'transport01']


    error_problems = ['airport06', 'airport07', 'pegsol04', 'pegsol05', 'pegsol06', 'blocks01', 'gripper01', 'miconic02', 'openstacks01', 'tpp03', 'travel02', 'miconic03', 'transport01']

    for p in error_problems:
        # remove_cache(p)
        generate_framework(p)
        exit(10)
    exit(10)
    # for problem in PROBLEMS:
    #     make_mt_cache_file_from_csv(problem, CONFIGURATIONS)

    # selects the state with the oracle (but it looks like too many states kill all the mutants)
    # for problem in relevant_problems:
    #     df = pd.read_csv(problem_result_filepath(problem))
    #     df = df.loc[df.error==0]
    #     df['optimal'] = df['result'] + df['node_cost'] - df['source_result']
    #     df.loc[df.optimal==0, 'a'] = 0
    #     df.loc[~(df.optimal==0), 'a'] = 1
    #     df = df.astype({'a': int})
    #     indexes = df.groupby(['node_index'], as_index=False).sum().sort_values(by='a', ascending=False, inplace=False)['node_index'].head(NB_TESTS).tolist()
    #     # counts = np.unique(np.array(df.groupby(['node_index'], as_index=False).sum().sort_values(by='a', ascending=False, inplace=False)['a'].tolist()), return_counts=True)
    #     update_cache(problem, indexes)

    my_args = get_arguments([('wastar', 'ff')], error_problems, 500, ['select_bfs'])
    print(f'{len(my_args)} executions are about to be launched.')
    pool.starmap(run_framework_fd_planner, my_args, chunksize=3)
    result_fps = list(map(lambda x: x[4], my_args))
    indexes_df = regroup_framework_result_dataframes(result_fps)
    indexes_df.replace(to_replace={'select_bfs': 'bfs'}, inplace=True)
    indexes_df.to_csv('fd_fixed_bfs.csv', index=0)
    print(indexes_df.loc[indexes_df.error==1]['problem'].unique().tolist())
    for result_fp in result_fps:
        os.remove(result_fp)


    # for fp in ['i1.csv', 'i2.csv']:
    #     df1 = pd.read_csv(fp)
    #     df = df1.copy()
    #     df['optimal'] = df['result'] + df['node_cost'] - df['source_result']
    #     df.loc[df.optimal==0, 'a'] = 0
    #     df.loc[~(df.optimal==0), 'a'] = 1
    #     df = df.astype({'a': int})
    #     df['failure'] = df['a']
    #     df['generator'] = 'a'
    # plot_mutation_coverage(pd.concat([df1, df], ignore_index=True), fp.split('.')[0] + '.png')


    def transform_row(r):
        if not r.generator.startswith('walks'):
            costs = [r.source_result, r.result, r.node_cost]
            costs.sort()
            r.failure = 1 if costs[2] != costs[1] + costs[0] else 0
        return r

    df = pd.read_csv('results/fd_results_10.csv')

