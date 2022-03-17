import os
import re
import subprocess
import pandas as pd

NB_TESTS = 15
NB_RANDOM = 10
DATA_SET_FOLDER = 'experiments'
MR0_GENERATORS = ['generator1', 'generator2']
MR1_GENERATORS = ['generator3', 'generator4']
HEURISTICS = ['h_cost', 'h_reversed_cost', 'h_distance_with_i', 'h_distance_with_g', 'h_zero']

BEST_GENERATORS = {
    'blocks': 'generator1',
    'miconic': 'generator1',
    'gripper': 'generator3',
    'hanoi': 'generator1',
    'news': 'generator3',
    'travel': 'generator1',
    'movie': 'generator3'
}

FD_PLANNERS = {
    'wastar_add': 'python planners/fd_wastar.py 10 add 10',
    'wastar_ff': 'python planners/fd_wastar.py 10 ff 10',
    'wastar_gc': 'python planners/fd_wastar.py 10 goalcount 10',
    'mutant_add': 'python planners/fd_mutant.py 10 add', 
    'mutant_ff': 'python planners/fd_mutant.py 10 ff',
    'mutant_gc': 'python planners/fd_mutant.py 10 goalcount'
}

PLANNERS = {
    'dfs_first_solution': './planners/mutated_planner.exe forward dfs_first_solution h_0',
    'a_star_mutant1_diff': './planners/mutated_planner.exe forward a_star_mutant1 h_diff',
    'a_star_mutant1_add': './planners/mutated_planner.exe forward a_star_mutant1 h_add',
    'a_star_mutant1_plus': './planners/mutated_planner.exe forward a_star_mutant1 h_plus', 
    'a_star_mutant1_length': './planners/mutated_planner.exe forward a_star_mutant1 h_length',
    'a_star_mutant2_diff': './planners/mutated_planner.exe forward a_star_mutant2 h_diff',
    'a_star_mutant2_add': './planners/mutated_planner.exe forward a_star_mutant2 h_add',
    'a_star_mutant2_plus': './planners/mutated_planner.exe forward a_star_mutant2 h_plus', 
    'a_star_mutant2_length': './planners/mutated_planner.exe forward a_star_mutant2 h_length',
    'a_star_mutant3_diff': './planners/mutated_planner.exe forward a_star_mutant3 h_diff',
    'a_star_mutant3_add': './planners/mutated_planner.exe forward a_star_mutant3 h_add',
    'a_star_mutant3_plus': './planners/mutated_planner.exe forward a_star_mutant3 h_plus', 
    'a_star_mutant3_length': './planners/mutated_planner.exe forward a_star_mutant3 h_length'
}

REAL_WORLD_PLANNERS = {
    'planningdomains': 'python planners/planningdomains.py',
    'webplanner': 'python planners/webplanner.py',
    'fdss1': 'python planners/fd_alias.py 10 seq-opt-fdss-1',
    'fdss2': 'python planners/fd_alias.py 10 seq-opt-fdss-2', 
    'lmcut': 'python planners/fd_alias.py 10 seq-opt-lmcut',
    'merge_shrink': 'python planners/fd_alias.py 10 seq-opt-merge-and-shrink',
    'bjolp': 'python planners/fd_alias.py 10 seq-opt-bjolp',
    'wastar_add': 'python planners/fd_wastar.py 10 add 10',
    'astar_add': 'python planners/fd_astar.py 10 add',
    'wastar_ff': 'python planners/fd_wastar.py 10 ff 10',
    'astar_ff': 'python planners/fd_astar.py 10 ff'
}

problem_name = re.compile('.+/(.+).pddl')

#######################################################
## RUNNING FUNCTIONS
#######################################################

def run_framework(domain: str, problem: str, planner_command: str, mr: str, nb_tests: int, output: str, generator: str, heuristic: str):
    """
    runs the framework a single time with the given configuration.
    """
    command = f'./main.exe {domain} {problem} "{planner_command}" {mr} {nb_tests} true {output} {generator} {heuristic}'
    # process = subprocess.run(command, stdout=subprocess.DEVNULL)
    process = subprocess.run(command, capture_output=True)
    if process.returncode != 0:
        print(f'something went wrong (error {process.returncode}) with command :')
        print(command)

def run_configurations(domain_filepath: str, problem_filepath: str, planners_dict: 'dict[str, str]', mr: str, generators: 'list[str]', heuristics: 'list[str]'):
    """
    runs the framework on all planners for each configuration possible (different combinations of generator / heuristic).
    It iterates on the list of planners at the end so it always run all planners and then change the configuration of the framework. 
    """
    for generator in generators:
        for heuristic in heuristics:
            for (k, v) in planners_dict.items():
                problem = problem_name.match(problem_filepath).group(1).lower()
                output = f'data/{k}_{problem}__{generator}__{heuristic}.csv'
                run_framework(domain_filepath, problem_filepath, v, mr, NB_TESTS, output, generator, heuristic)

def run_configurations_multiple(domain_filepath: str, problem_filepath: str, planners_dict: 'dict[str, str]', mr: str, generators: 'list[str]', heuristics: 'list[str]'):
    """
    runs NB_RANDOM times the framework on all planners for each configuration possible (different combinations of generator / heuristic).
    It iterates on the list of planners at the end so it always run all planners (NB_RANDOM times) and then change the configuration of the framework. 
    """
    for generator in generators:
        for heuristic in heuristics:
            for (k, v) in planners_dict.items():
                for i in range(1, NB_RANDOM + 1):
                    problem = problem_name.match(problem_filepath).group(1).lower()
                    output = f'data/{k}_{problem}__{generator}__{heuristic}{i}.csv'
                    run_framework(domain_filepath, problem_filepath, v, mr, NB_TESTS, output, generator, heuristic)

def run_experiments_generator5():
    """
    runs NB_RANDOM times the configuration generator1 / h_zero on the input data set (all planners and all problems).
    """
    # domain_folders = os.listdir(DATA_SET_FOLDER)
    domain_folders = ['blocks', 'gripper', 'hanoi', 'miconic', 'movie', 'news', 'travel'] # does not use depot nor rover domains
    for domain_folder in domain_folders:
        domain_filepath = f'{DATA_SET_FOLDER}/{domain_folder}/domain.pddl'
        problem_filepaths = [f'{DATA_SET_FOLDER}/{domain_folder}/' + f for f in os.listdir(f'{DATA_SET_FOLDER}/{domain_folder}') if f.endswith('.pddl') and 'domain' not in f]
        for problem_filepath in problem_filepaths:
            run_configurations_multiple(domain_filepath, problem_filepath, PLANNERS, 'mr0', ['generator5'], ['h_zero'])

def run_experiments_real_world_planners():
    """
    runs the configuration generator1 / h_zero only REAL_WORLD_PLANNERS on all problems.
    """
    domain_folders = os.listdir(DATA_SET_FOLDER)
    for domain_folder in domain_folders:
        domain_filepath = f'{DATA_SET_FOLDER}/{domain_folder}/domain.pddl'
        problem_filepaths = [f'{DATA_SET_FOLDER}/{domain_folder}/' + f for f in os.listdir(f'{DATA_SET_FOLDER}/{domain_folder}') if f.endswith('.pddl') and 'domain' not in f]
        for problem_filepath in problem_filepaths:
            run_configurations(domain_filepath, problem_filepath, REAL_WORLD_PLANNERS, 'mr0', ['generator1'], ['h_zero'])

def run_experiments_h_random():
    """
    runs NB_RANDOM times the framework on the input data set (PLANNERS and all problems) with different configurations defined as follows :
    - h_random is always used
    - only the best generator is used for a given domain (to define BEST_GENERATORS)
    """
    # domain_folders = os.listdir(DATA_SET_FOLDER)
    domain_folders = ['blocks', 'gripper', 'hanoi', 'miconic', 'movie', 'news', 'travel'] # does not use depot nor rover domains
    for domain_folder in domain_folders:
        domain_filepath = f'{DATA_SET_FOLDER}/{domain_folder}/domain.pddl'
        problem_filepaths = [f'{DATA_SET_FOLDER}/{domain_folder}/' + f for f in os.listdir(f'{DATA_SET_FOLDER}/{domain_folder}') if f.endswith('.pddl') and 'domain' not in f]
        for problem_filepath in problem_filepaths:
            generator = BEST_GENERATORS[domain_folder]
            mr = 'mr0' if generator in MR0_GENERATORS else 'mr1'
            run_configurations_multiple(domain_filepath, problem_filepath, PLANNERS, mr, [generator], ['h_random'])

def run_experiments_planners():
    """
    runs all the configurations on PLANNERS on all problems.
    """
    # domain_folders = os.listdir(DATA_SET_FOLDER)
    domain_folders = ['blocks', 'gripper', 'hanoi', 'miconic', 'movie', 'news', 'travel'] # does not use depot nor rover domains
    for domain_folder in domain_folders:
        domain_filepath = f'{DATA_SET_FOLDER}/{domain_folder}/domain.pddl'
        problem_filepaths = [f'{DATA_SET_FOLDER}/{domain_folder}/' + f for f in os.listdir(f'{DATA_SET_FOLDER}/{domain_folder}') if f.endswith('.pddl') and 'domain' not in f]
        for problem_filepath in problem_filepaths:
            run_configurations(domain_filepath, problem_filepath, PLANNERS, 'mr0', MR0_GENERATORS, HEURISTICS)
            run_configurations(domain_filepath, problem_filepath, PLANNERS, 'mr1', MR1_GENERATORS, HEURISTICS)

def run_experiments_all():
    # runs the core benchmark
    run_experiments_planners()
    # runs the baseline generation strategy benchmark
    run_experiments_generator5()
    # runs the baseline random strategy benchmark
    run_experiments_h_random()
    # runs the real world planners benchmark
    run_experiments_real_world_planners()

#######################################################
## IMPORT FUNCTIONS
#######################################################

def get_dataframe(filepath: str):
    """
    returns a dataframe from a .csv file (whose filepath matches the re)
    """
    p = re.compile('.+/([a-z_0-9]+)_([a-z_0-9-]+\d)__[a-z_0-9]+__([a-z_0-9]+).csv')
    m = p.match(filepath)
    planner_name = m.group(1)
    problem_name = m.group(2)
    heuristic_name = m.group(3)

    f = open(filepath, 'r')
    header = f.readline().split(',')
    domain_name = header[0]
    source_result_cost = header[2]
    f.close()

    df = pd.read_csv(filepath, header=1)
    df.insert(0, 'planner', planner_name, True)
    df.insert(1, 'domain', domain_name, True)
    df.insert(2, 'problem', problem_name, True)
    df.insert(3, 'heuristic', heuristic_name, True)
    df.insert(len(df.columns), 'source_result_cost', source_result_cost, True)
    return df

def regroup_dataframes(filepaths: 'list[str]', result_filename: str):
    """
    retrieves all the results and regroup them into a single dataframe
    """
    df = pd.concat(map(get_dataframe, filepaths))
    df.to_csv(result_filename, index=False)
    return df

#######################################################
## SCRIPT SECTION / MAIN
#######################################################

# runs all the experiments
run_experiments_all()
# concatenates the results
filepaths = ['data/' + f for f in os.listdir('data') if f.endswith('.csv')]
regroup_dataframes(filepaths, 'results.csv')
