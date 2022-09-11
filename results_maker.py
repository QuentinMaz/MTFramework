import os
import re
import subprocess
import pandas as pd

NB_TESTS = 10
DATA_SET_FOLDER = 'experiments'
MR0_GENERATORS = ['generator1']
HEURISTICS = ['h_distance_with_i', 'h_distance_with_g', 'h_random']

FD_PLANNERS = {
    # real non-optimal planners
    'wastar_add': 'python3 fd_wastar.py 30 add 10',
    'wastar_ff': 'python3 fd_wastar.py 30 ff 10',
    'astar_add': 'python3 fd_astar.py 30 add',
    'astar_ff': 'python3 fd_astar.py 30 ff',
    # mutated planners
    'mutant1_add': 'python3 fd_mutant1.py 30 add', 
    'mutant1_ff': 'python3 fd_mutant1.py 30 ff', 
    'mutant2_add': 'python3 fd_mutant2.py 30 add', 
    'mutant2_ff': 'python3 fd_mutant2.py 30 ff', 
    'mutant3_add': 'python3 fd_mutant3.py 30 add', 
    'mutant3_ff': 'python3 fd_mutant3.py 30 ff'
}

problem_name = re.compile('.+/(.+).pddl')

#######################################################
## RUNNING FUNCTIONS
#######################################################

def run_framework(domain: str, problem: str, planner_command: str, mr: str, nb_tests: int, output: str, generator: str, heuristics: 'list[str]'):
    """
    runs the framework a single time with the given configuration.
    """
    command = f'sicstus -l framework.pl --goal "start, halt." -- {domain} {problem} "{planner_command}" {mr} {nb_tests} true {output} {generator} {" ".join(heuristics)}'
    # process = subprocess.run(command, stdout=subprocess.DEVNULL)
    print(f'\t\t\t{problem} started.')
    process = subprocess.run(command, shell=True, capture_output=True)
    if process.returncode != 0:
        print(f'something went wrong (error {process.returncode}) with command :')
        print(command)
    else:
        print(process.stdout.decode())

def run_configurations(domain_filepath: str, domain:str, problem_filepath: str, planners_dict: 'dict[str, str]', mr: str, generators: 'list[str]', heuristics: 'list[str]'):
    """
    runs the framework on all planners for each configuration possible (different combinations of generator / heuristic).
    It iterates on the list of planners at the end so it always run all planners and then change the configuration of the framework. 
    """
    for generator in generators:
        for (k, v) in planners_dict.items():
            problem = problem_name.match(problem_filepath).group(1).lower()
            output = f'data/{k}_{domain}_{problem}__{generator}.csv'
            if f'{k}_{domain}_{problem}__{generator}.csv' in os.listdir('data'):
                print(f'results for planner {k} on {domain}-{problem} with {generator} already exists.')
                continue
            else:
                run_framework(domain_filepath, problem_filepath, v, mr, NB_TESTS, output, generator, heuristics)

def run_experiments_planners():
    """
    runs all the configurations on PLANNERS on all problems.
    """
    domain_folders = os.listdir(DATA_SET_FOLDER)
    domain_folders.sort()
    for domain_folder in domain_folders:
        domain_filepath = f'{DATA_SET_FOLDER}/{domain_folder}/domain.pddl'
        problem_filepaths = [f'{DATA_SET_FOLDER}/{domain_folder}/' + f for f in os.listdir(f'{DATA_SET_FOLDER}/{domain_folder}') if f.endswith('.pddl') and 'domain' not in f]
        problem_filepaths.sort()
        for problem_filepath in problem_filepaths:
            run_configurations(domain_filepath, domain_folder, problem_filepath, FD_PLANNERS, 'mr0', MR0_GENERATORS, HEURISTICS)

#######################################################
## IMPORT FUNCTIONS
#######################################################

def get_dataframe(filepath: str):
    """
    returns a dataframe from a .csv file (whose filepath matches the re)
    """
    p = re.compile('.+/([a-z0-9_]+)_([a-z0-9]+)_([a-z0-9-]+\d)__([a-z0-9]+).csv')
    m = p.match(filepath)
    planner_name = m.group(1)
    domain_name = m.group(2)
    problem_name = m.group(3)
    generator_name = m.group(4)

    f = open(filepath, 'r')
    header = f.readline().split(',')
    # domain_name = header[0]
    source_result_cost = header[2]
    f.close()

    df = pd.read_csv(filepath, header=1)
    df.insert(0, 'planner', planner_name, True)
    df.insert(1, 'domain', domain_name, True)
    df.insert(2, 'problem', problem_name, True)
    df.insert(3, 'generator', generator_name, True)
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
run_experiments_planners()
# concatenates the results
filepaths = ['data/' + f for f in os.listdir('data') if f.endswith('.csv')]
regroup_dataframes(filepaths, 'fd_results_experiments.csv')
