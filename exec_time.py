import multiprocessing
import subprocess
import os
import re
import pandas as pd
import numpy as np

GENERATORS = ['bfs', 'random', 'random_walk_generation']
PROBLEMS = ['airport06', 'airport07', 'blocks01', 'blocks02', 'blocks03', 'gripper01', 'miconic02', 'miconic03', 'openstacks01', 'pegsol04', 'pegsol05', 'pegsol06', 'psr-small03', 'psr-small06', 'psr-small08', 'psr-small09', 'tpp03', 'transport01', 'travel02']
NB_THREADS = 19
NB_TESTS = 10
NB_REPETITIONS = 2
PROBLEM_REGEX = re.compile('(.+)(\d\d)')
# RESULT_REGEX = re.compile('.*/(.+)_(.+)_(.+).txt')
HEURISTICS = {
    'hmax': 'h_max',
    'hdiff': 'h_diff',
    'hlength': 'h_state_length',
    'hi': 'h_distance_with_i',
    'hg': 'h_distance_with_g',
    'hnba': 'h_nb_actions'
}

CONFIGURATIONS = [('f5', 'hi'), ('f5', 'hlength'), ('f5', 'hnba'), ('f2', 'hdiff'), ('f2', 'hlength'), ('f2', 'hnba'), ('f4', 'hg'), ('f4', 'hdiff'), ('f4', 'hi'), ('f4', 'hlength'), ('f4', 'hmax'), ('f4', 'hnba'), ('f3', 'hg'), ('f3', 'hdiff'), ('f3', 'hi'), ('f3', 'hlength'), ('f3', 'hmax'), ('f3', 'hnba'), ('f6', 'hg'), ('f6', 'hi'), ('f6', 'hlength'), ('f6', 'hnba')]


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


def get_arguments1(problems: list[str]):
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


def get_arguments2(problems: list[str], generator: str):
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
############################################################## EXECUTION TIME RESULTS ################################################################


def state_generation_exec_time_framework_analysis(domain_filename: str, problem_filename: str, state_generation_filename: str):
    command = f'main.exe --state_generation {domain_filename} {problem_filename} {state_generation_filename}'
    exec_time = 0
    try:
        p = subprocess.run(command, capture_output=True)
        exec_time = float(p.stdout.decode().splitlines()[-1])
    except:
        print(f'Framework error with args: --state_generation {domain_filename} {problem_filename}.')
    return problem_filename, exec_time


def state_selection_exec_time_framework_analysis(domain_filename: str, problem_filename: str, selection: str):
    command = f'main.exe --{selection} {domain_filename} {problem_filename} {NB_TESTS} 10000'
    exec_time = 0
    try:
        p = subprocess.run(command, capture_output=True)
        exec_time = float(p.stdout.decode().splitlines()[-1])
    except:
        print(f'Framework error with args: --{selection} {domain_filename} {problem_filename} {NB_TESTS}.')
    return problem_filename, exec_time


def mutation_state_selection_exec_time_framework_analysis(domain_filename: str, problem_filename: str, output: str, configurations: list[tuple[str, str]]) -> list[int]:
    planners_commands = []
    for (s, h) in configurations:
        planners_commands.append(f'"planners/prolog_planner.exe mutated_astar-{s[0]+str(int(s[1])-1)} {HEURISTICS[h]}"')
    command = f'main.exe --test {domain_filename} {problem_filename} {output} {" ".join(planners_commands)}'
    results = [0, 0, 0]
    try:
        p = subprocess.run(command, shell=True, capture_output=True)
        exec_times = [float(l) for l in p.stdout.decode().splitlines() if is_float(l)]
        results = [exec_times[0], np.sum(exec_times[1:]), exec_times[0] - np.sum(exec_times[1:])]
    except:
        print(f'Error when building mutant cache for problem {problem_filename}.')
    return results


######################################################################################################################################################
############################################################## MAIN ##################################################################################


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=NB_THREADS)

    # state generation execution time
    my_args = get_arguments1(PROBLEMS)
    state_generation_results = {p: [] for p in list(map(lambda x: x[1], my_args))}
    print('State generation execution time.')
    for i in range(NB_REPETITIONS):
        print(f'{len(my_args)} executions are about to be launched.')
        execution_time_result = pool.starmap(state_generation_exec_time_framework_analysis, my_args, chunksize=2)
        for problem_filename, exec_time in execution_time_result:
            state_generation_results[problem_filename].append(exec_time)
    f = open('state_generation_results.txt', 'w')
    for p_filename, exec_times in state_generation_results.items():
        f.write(f'{p_filename},{np.mean(exec_times)},{np.std(exec_times)}\n')
    f.close()

    # state selection and random walk generation execution time
    for generator in GENERATORS:
        print(f'Execution time for {generator}.')
        my_args = get_arguments2(PROBLEMS, generator)
        state_selection_results = {p: [] for p in list(map(lambda x: x[1], my_args))}
        for i in range(NB_REPETITIONS):
            print(f'{len(my_args)} executions are about to be launched.')
            execution_time_result = pool.starmap(state_selection_exec_time_framework_analysis, my_args, chunksize=2)
            for problem_filename, exec_time in execution_time_result:
                state_selection_results[problem_filename].append(exec_time)
        f = open(f'{generator}_state_selection_results.txt' if generator != 'random_walk_generation' else 'random_walk_state_generation_results.txt', 'w')
        for p_filename, exec_times in state_selection_results.items():
            f.write(f'{p_filename},{np.mean(exec_times)},{np.std(exec_times)}\n')
        f.close()

    for index_repetition in range(NB_REPETITIONS):
        args = []
        cache_files = []
        # creates arguments for multithreaded called (one per problem)
        for problem in PROBLEMS:
            m = PROBLEM_REGEX.match(problem)
            domain_name = m.group(1)
            i = m.group(2)
            domain = 'domain' if 'domain.pddl' in os.listdir(f'benchmarks/{domain_name}') else f'domain{i}'
            arg = (f'benchmarks/{domain_name}/{domain}.pddl', f'benchmarks/{domain_name}/task{i}.pddl', f'tmp/{problem}.txt', CONFIGURATIONS)
            args.append(arg)
            cache_files.append(f'cache/{domain_name}_task{i}')
        execution_times_results = pool.starmap(mutation_state_selection_exec_time_framework_analysis, args, chunksize=1)
        tmp_fps = list(map(lambda x: x[2], args))
        for tmp_fp in tmp_fps:
            if os.path.exists(tmp_fp):
                os.remove(tmp_fp)
        # also removes the files cached to include the state generation execution time
        for cache_file in cache_files:
            if os.path.exists(f'{cache_file}.txt'):
                os.remove(f'{cache_file}.txt')
            if os.path.exists(f'{cache_file}.csv'):
                os.remove(f'{cache_file}.csv')
        # exports raw results
        f = open(f'{index_repetition}.txt', 'w')
        for execution_times in execution_times_results:
            # execution times reported: total execution time, execution time spent waiting for the mutated AI planners to solve problems (i.e, follow-up test cases) and their difference (i.e, total MorphinPlan's execution time excluding the ones of the mutants)
            f.write(f'{",".join([str(et) for et in execution_times])}\n')
        f.close()
