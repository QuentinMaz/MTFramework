import multiprocessing
import subprocess
import os
import re
import pandas as pd
import numpy as np

PROBLEMS = ['airport06', 'airport07', 'blocks01', 'blocks02', 'blocks03', 'gripper01', 'miconic02', 'miconic03', 'openstacks01', 'pegsol04', 'pegsol05', 'pegsol06', 'psr-small03', 'psr-small06', 'psr-small08', 'psr-small09', 'tpp03', 'transport01', 'travel02']
NB_THREADS = 19
NB_TESTS = 10
NB_REPETITIONS = 10
RESULT_COLUMN_NAME = 'execution_time(sec)'
PROBLEM_REGEX = re.compile('(.+)(\d\d)')
PROBLEM_FILEPATH = re.compile('.*/(.+)/task(\d\d).pddl')
HEURISTICS = {
    'hmax': 'h_max',
    'hdiff': 'h_diff',
    'hlength': 'h_state_length',
    'hi': 'h_distance_with_i',
    'hg': 'h_distance_with_g',
    'hnba': 'h_nb_actions'
}

CONFIGURATIONS = [('f4', 'hi'), ('f4', 'hlength'), ('f4', 'hnba'), ('f1', 'hdiff'), ('f1', 'hlength'), ('f1', 'hnba'), ('f3', 'hg'), ('f3', 'hdiff'), ('f3', 'hi'), ('f3', 'hlength'), ('f3', 'hmax'), ('f3', 'hnba'), ('f2', 'hg'), ('f2', 'hdiff'), ('f2', 'hi'), ('f2', 'hlength'), ('f2', 'hmax'), ('f2', 'hnba'), ('f5', 'hg'), ('f5', 'hi'), ('f5', 'hlength'), ('f5', 'hnba')]

# https://stackoverflow.com/a/20929881
def is_float(element: any) -> bool:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

######################################################################################################################################################
############################################################## ARGUMENTS HELPERS #####################################################################


def get_arguments_helper_1(problems: list[str]):
    args = []
    for problem in problems:
        m = PROBLEM_REGEX.match(problem)
        domain_name = m.group(1)
        i = m.group(2)
        domain = 'domain' if 'domain.pddl' in os.listdir(f'benchmarks/{domain_name}') else f'domain{i}'
        state_generation_filename = f'{domain_name}_{problem}.txt'
        arg = (f'benchmarks/{domain_name}/{domain}.pddl', f'benchmarks/{domain_name}/task{i}.pddl', state_generation_filename)
        args.append(arg)
    return args


def get_arguments_helper_2(problems: list[str], generator: str):
    args = []
    for problem in problems:
        m = PROBLEM_REGEX.match(problem)
        domain_name = m.group(1)
        i = m.group(2)
        domain = 'domain' if 'domain.pddl' in os.listdir(f'benchmarks/{domain_name}') else f'domain{i}'
        arg = (f'benchmarks/{domain_name}/{domain}.pddl', f'benchmarks/{domain_name}/task{i}.pddl', generator)
        args.append(arg)
    return args


######################################################################################################################################################
############################################################## FRAMEWORK LAUNCHERS ################################################################


def generate_states(domain_filename: str, problem_filename: str, state_generation_filename: str) -> tuple[str, float]:
    """Generates the states and returns the total execution time as a 2-tuple containing @problem_filename first."""
    command = f'main.exe --state_generation {domain_filename} {problem_filename} {state_generation_filename}'
    execution_time = 0.0
    try:
        p = subprocess.run(command, capture_output=True)
        # the framework is expected to output its execution time in the standard output
        execution_time = float(p.stdout.decode().splitlines()[-1])
    except:
        print(f'Framework error with args: --state_generation {domain_filename} {problem_filename}.')
    return (problem_filename, execution_time)


def select_states(domain_filename: str, problem_filename: str, selection: str) -> tuple[str, float]:
    """
    Selects NB_TESTS states and returns the total execution time as a 2-tuple containing @problem_filename first.
    It expects the states to have already been generated.
    """
    command = f'main.exe --{selection} {domain_filename} {problem_filename} {NB_TESTS} 10000'
    execution_time = 0.0
    try:
        p = subprocess.run(command, capture_output=True)
        execution_time = float(p.stdout.decode().splitlines()[-1])
    except:
        print(f'Framework error with args: --{selection} {domain_filename} {problem_filename} {NB_TESTS} 10000.')
    return (problem_filename, execution_time)


def compute_mutants_cache_file(domain_filename: str, problem_filename: str, output_filename: str, configurations: list[tuple[str, str]]) -> tuple[str, list[float]]:
    planners_commands = []
    for (s, h) in configurations:
        planners_commands.append(f'"planners/prolog_planner.exe mutated_astar-{s} {HEURISTICS[h]}"')
    command = f'main.exe --test {domain_filename} {problem_filename} {output_filename} {" ".join(planners_commands)}'
    execution_times = [0.0, 0.0, 0.0]
    try:
        p = subprocess.run(command, shell=True, capture_output=True)
        exec_times = [float(l) for l in p.stdout.decode().splitlines() if is_float(l)]
        execution_times = [exec_times[0], np.sum(exec_times[1:]), exec_times[0] - np.sum(exec_times[1:])]
    except:
        print(f'Error when building mutant cache for problem {problem_filename}.')
    return (problem_filename, execution_times)


######################################################################################################################################################
############################################################## EXECUTION TIME RESULTS ################################################################


def state_generation_results(problems: list[str]):
    my_args = get_arguments_helper_1(problems)
    state_generation_results = {p: [] for p in list(map(lambda x: x[1], my_args))}
    print('State generation execution time computation starts.')
    for i in range(NB_REPETITIONS):
        state_generation_times = pool.starmap(generate_states, my_args, chunksize=2)
        for problem_filename, exec_time in state_generation_times:
            state_generation_results[problem_filename].append(exec_time)
        print(f'Iteration {i} done ({[r[1] for r in state_generation_times]}).')
    data = {}
    for k, v in state_generation_results.items():
        m = PROBLEM_FILEPATH.match(k)
        problem_name = f'{m.group(1)}{m.group(2)}'
        data[problem_name] = np.mean(v)
    df = pd.DataFrame.from_dict(data, orient='index', columns=[RESULT_COLUMN_NAME])
    results_filename = f'results/state_generation_results_{NB_REPETITIONS}.csv'
    df.to_csv(results_filename, index_label='problem')
    print(f'Results exported to {results_filename}.')
    for filename in list(map(lambda x: x[2], my_args)):
        if os.path.exists(filename):
            os.remove(filename)
    print('Temporary files cleaned.')
    print('State generation execution time computation done.')
    return df


def state_selection_results(problems: list[str], generators: list[str]):
    dfs = []
    for generator in generators:
        print(f'{generator} state selection execution time computation starts.')
        my_args = get_arguments_helper_2(problems, generator)
        state_selection_results = {p: [] for p in list(map(lambda x: x[1], my_args))}
        for i in range(NB_REPETITIONS):
            state_selection_times = pool.starmap(select_states, my_args, chunksize=2)
            for problem_filename, exec_time in state_selection_times:
                state_selection_results[problem_filename].append(exec_time)
            print(f'Iteration {i} done ({[r[1] for r in state_selection_times]}).')
        data = {}
        for k, v in state_selection_results.items():
            m = PROBLEM_FILEPATH.match(k)
            problem_name = f'{m.group(1)}{m.group(2)}'
            data[problem_name] = np.mean(v)
        df = pd.DataFrame.from_dict(data, orient='index', columns=[RESULT_COLUMN_NAME])
        results_filename = f'results/{generator}_state_selection_results_{NB_REPETITIONS}.csv'
        df.to_csv(results_filename, index_label='problem')
        dfs.append(df)
        print(f'{generator} state selection execution time computation done.')
    return dfs


def mutant_state_selection_results(problems: list[str], configurations: list[tuple[str, str]]):
    print('Mutants state selection execution time starts.')
    results = {p: [] for p in problems}
    for j in range(NB_REPETITIONS):
        args = []
        cache_files = []
        # creates arguments for multithreading (one per problem)
        for problem in problems:
            m = PROBLEM_REGEX.match(problem)
            domain_name = m.group(1)
            i = m.group(2)
            domain = 'domain' if 'domain.pddl' in os.listdir(f'benchmarks/{domain_name}') else f'domain{i}'
            arg = (f'benchmarks/{domain_name}/{domain}.pddl', f'benchmarks/{domain_name}/task{i}.pddl', f'tmp/{problem}.txt', configurations)
            args.append(arg)
            cache_files.append(f'cache/{domain_name}_task{i}')
        execution_times_results = pool.starmap(compute_mutants_cache_file, args, chunksize=1)
        # saves the results
        for problem_filename, execution_times in execution_times_results:
            m = PROBLEM_FILEPATH.match(problem_filename)
            problem_name = f'{m.group(1)}{m.group(2)}'
            results[problem_name].append(execution_times[2])
        # removes temporary files
        tmp_fps = list(map(lambda x: x[2], args))
        for tmp_fp in tmp_fps:
            if os.path.exists(tmp_fp):
                os.remove(tmp_fp)
        # also removes the files cached to include the state generation execution time everytime
        for cache_file in cache_files:
            if os.path.exists(f'{cache_file}.txt'):
                os.remove(f'{cache_file}.txt')
            if os.path.exists(f'{cache_file}.csv'):
                os.remove(f'{cache_file}.csv')
        # exports raw results in case of failure
        f = open(f'raw_mutant_execution_time_results_{j}.txt', 'w')
        f.write(f'problem,total_{RESULT_COLUMN_NAME},mutants_{RESULT_COLUMN_NAME},{RESULT_COLUMN_NAME}\n')
        for problem_filename, execution_times in execution_times_results:
            m = PROBLEM_FILEPATH.match(problem_filename)
            problem_name = f'{m.group(1)}{m.group(2)}'
            f.write(f'{problem_name},{",".join([str(exec_time) for exec_time in execution_times])}\n')
        f.close()
        print(f'Iteration {j} done ({[r[1][2] for r in execution_times_results]}).')
    data = {k: np.mean(v) for k, v in results.items()}
    df = pd.DataFrame.from_dict(data, orient='index', columns=[RESULT_COLUMN_NAME])
    results_filename = f'results/mutants_state_selection_results_{NB_REPETITIONS}.csv'
    df.to_csv(results_filename, index_label='problem')
    print('Mutants state selection execution time done.')
    return df



######################################################################################################################################################
############################################################## MAIN ##################################################################################


if __name__ == '__main__':
    """
    This script aims at comparing execution times of the framework with the baselines.
    The execution times are defined as follows:
    - bfs_det := state generation + bfs state selection
    - bfs_ran := state generation + random selection
    - wal_ran := random walk based state generation
    - framework := state generation + mutant cache file creation considering all mutants available 
    Note that by using all the mutants, the results will represent the worst cost scenarios in terms of computation costs.
    All the results are averages of NB_REPETITIONS.
    As a disclaimer, those values can vary greatly from a machine to another.
    The number of threads can be changed with the constant NB_THREADS (top of this file).
    """

    # makes sure the cache folder is empty (so that accurate execution times will be computed)
    cache_files = os.listdir('cache/')
    for cache_file in cache_files:
        if cache_file.endswith('.csv') or cache_file.endswith('.txt'):
            os.remove(f'cache/{cache_file}')

    pool = multiprocessing.Pool(processes=NB_THREADS)

    # state_gen = pd.read_csv(f'results/state_generation_results_{NB_REPETITIONS}.csv')
    # bfs_det_select = pd.read_csv(f'results/bfs_state_selection_results_{NB_REPETITIONS}.csv')
    # bfs_rand_select = pd.read_csv(f'results/random_state_selection_results_{NB_REPETITIONS}.csv')
    # wal_rand_gen = pd.read_csv(f'results/random_walk_generation_state_selection_results_{NB_REPETITIONS}.csv')
    # mutants_select = pd.read_csv(f'results/mutants_state_selection_results_{NB_REPETITIONS}.csv')

    # state generation execution times
    state_gen = state_generation_results(PROBLEMS)
    # state selection execution times
    bfs_det_select, bfs_rand_select = state_selection_results(PROBLEMS, ['bfs', 'random'])
    # random walk state generation execution times
    wal_rand_gen = state_selection_results(PROBLEMS, ['random_walk_generation'])[0]
    mutants_select = mutant_state_selection_results(PROBLEMS, CONFIGURATIONS)

    # regroups the results
    result_df = pd.DataFrame()
    result_df['problem'] = state_gen.index if not 'problem' in state_gen.columns else state_gen['problem'] # if the dataframes are read from .csv files, problems are in a column
    result_df.set_index('problem', inplace=True)
    result_df['mutant'] = mutants_select[RESULT_COLUMN_NAME]
    result_df['bfs_det'] = state_gen[RESULT_COLUMN_NAME] + bfs_det_select[RESULT_COLUMN_NAME]
    result_df['bfs_rand'] = state_gen[RESULT_COLUMN_NAME] + bfs_rand_select[RESULT_COLUMN_NAME]
    result_df['wal_rand'] = wal_rand_gen[RESULT_COLUMN_NAME]
    result_df.loc['mean'] = [np.mean(result_df[c]) for c in result_df.columns]
    result_df.to_csv('results/execution_time_results.csv', index=True)