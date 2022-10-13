import subprocess
import multiprocessing
import os
import re


FOLDER = 'benchmarks'
HEURISTICS = {
    'hmax': 'h_max',
    'hdiff': 'h_diff',
    'hlength': 'h_state_length',
    'hi': 'h_distance_with_i',
    'hg': 'h_distance_with_g',
    'hnba': 'h_nb_actions'
}


def build_main() -> None:
    """
    Builds the framework (main.exe) and removes all the artifacts that are created during the process.
    """
    compile_command = 'sicstus --goal "compile(framework), save_program(\'main.sav\'), halt."'
    build_command = 'cd "C:\Program Files (x86)\Microsoft Visual Studio\\2019\Community\VC\Auxiliary\Build" && vcvars64.bat && cd "C:\\Users\Quentin\Documents\\5INFO\Simula\MTFramework" && spld --output=main.exe --static main.sav'
    try:
        subprocess.run(compile_command, shell=True, stdout=subprocess.DEVNULL)
        subprocess.run(build_command, shell=True, stdout=subprocess.DEVNULL)
        for artifact in ['main.sav', 'main.pdb', 'main.ilk', 'main.exp', 'main.lib']:
            os.remove(artifact)
    except:
        print('something went wrong')


def cache_problem(domain_filename: str, problem_filename: str, output: str, configurations: list[tuple[str, str]]) -> None:
    """
    Runs a configuration, defined by a search and a heuristic, and returns the execution time.
    """
    planners_commands = []
    for (s, h) in configurations:
        planners_commands.append(f'"planners/prolog_planner.exe mutated_astar-{s} {HEURISTICS[h]}"')
    command = f'main.exe --cache {domain_filename} {problem_filename} {output} {" ".join(planners_commands)}'
    print(command)
    os.system(command)
    print('done.')


def get_arguments(configurations: list[tuple[str, str]], problems: list[str]) -> list[tuple[str, str, str, list[tuple[str, str]]]]:
    args = []

    regex = re.compile('(.+)(\d\d)')
    for problem in problems:
        m = regex.match(problem)
        domain_name = m.group(1)
        i = m.group(2)
        domain = 'domain' if 'domain.pddl' in os.listdir(f'{FOLDER}/{domain_name}') else f'domain{i}'
        arg = (f'{FOLDER}/{domain_name}/{domain}.pddl', f'{FOLDER}/{domain_name}/task{i}.pddl', f'tmp/{problem}.txt', configurations)
        args.append(arg)

    return args


def main():
    ########################## 1st benchmark setup (28-29/09) #######################################
    # validation configurations = [('f5', 'hdiff'), ('f5', 'hg'), ('f2', 'hdiff'), ('f5', 'hlength'), ('f5', 'hi')]
    # configurations = [('f2', 'hg'), ('f3', 'hi'), ('f5', 'hnba'), ('f3', 'hlength'), ('f2', 'hlength'), ('f2', 'hnba'), ('f3', 'hnba'), ('f3', 'hg'), ('f4', 'hg'), ('f3', 'hdiff'), ('f4', 'hdiff'), ('f4', 'hmax'), ('f3', 'hmax'), ('f4', 'hlength'), ('f4', 'hnba')]
    # problems = ['psr-small05', 'psr-small09', 'sokoban02', 'blocks09', 'miconic03', 'openstacks01', 'blocks06', 'pegsol09', 'satellite01', 'miconic04', 'transport01', 'depot01', 'newspapers02', 'psr-small02', 'psr-small06', 'psr-small07', 'pegsol06']
    # NB: openstack01 failed
    ########################## 2nd benchmark setup (29-30/09) #######################################
    # problems where 24 (over 30) configs are indeed mutants:
    # problems = ['miconic03', 'openstacks01', 'pegsol09', 'satellite01', 'miconic04', 'transport01', 'depot01', 'newspapers02', 'pegsol06']
    # configs for validation: [('f1', 'hmax'), ('f1', 'hdiff'), ('f1', 'hlength'), ('f1', 'hi'), ('f1', 'hg'), ('f1', 'hnba')]
    # configs for methodology:
    # configurations = [('f2', 'hmax'), ('f2', 'hdiff'), ('f2', 'hlength'), ('f2', 'hi'), ('f2', 'hg'), ('f2', 'hnba'), ('f3', 'hmax'), ('f3', 'hdiff'), ('f3', 'hlength'), ('f3', 'hi'), ('f3', 'hg'), ('f3', 'hnba'), ('f4', 'hmax'), ('f4', 'hdiff'), ('f4', 'hlength'), ('f4', 'hi'), ('f4', 'hg'), ('f4', 'hnba'), ('f5', 'hmax'), ('f5', 'hdiff'), ('f5', 'hlength'), ('f5', 'hi'), ('f5', 'hg'), ('f5', 'hnba')]
    # validation canceled because no nodes generated (precisely, only one for pegsol06)
    ########################## 2nd benchmark setup (30/09) #######################################
    # problems where 19 (over 30) configs are indeed mutants:
    problems = ['psr-small05', 'psr-small09', 'sokoban02', 'blocks09', 'miconic03', 'openstacks01', 'blocks06', 'pegsol09', 'satellite01', 'miconic04', 'transport01', 'depot01', 'newspapers02', 'psr-small02', 'pegsol06', 'psr-small06', 'psr-small07', 'blocks08']
    # configs which are not mutants on all the selected problems: [('f1', 'hmax'), ('f1', 'hdiff'), ('f1', 'hlength'), ('f1', 'hi'), ('f1', 'hg'), ('f1', 'hnba'), ('f2', 'hmax'), ('f2', 'hi'), ('f4', 'hi'), ('f5', 'hmax'), ('f5', 'hi')]
    # configs for validation: [('f5', 'hdiff'), ('f5', 'hg'), ('f2', 'hdiff'), ('f2', 'hg'), ('f5', 'hlength'), ('f3', 'hi'), ('f3', 'hlength'), ('f5', 'hnba'), ('f2', 'hlength'), ('f3', 'hnba')]
    # configs for methodology:
    configurations = [('f2', 'hnba'), ('f3', 'hg'), ('f4', 'hg'), ('f3', 'hdiff'), ('f4', 'hdiff'), ('f4', 'hmax'), ('f3', 'hmax'), ('f4', 'hlength'), ('f4', 'hnba')]
    # remaining usable configurations: [('f1', 'hmax'), ('f1', 'hnba'), ('f1', 'hlength'), ('f1', 'hi'), ('f1', 'hdiff'), ('f1', 'hg'), ('f5', 'hmax'), ('f2', 'hi'), ('f2', 'hmax'), ('f5', 'hi'), ('f4', 'hi')]

    my_args = get_arguments(configurations, problems)
    print(f'{len(my_args)} executions are about to be launched.')
    pool = multiprocessing.Pool(processes=3)
    pool.starmap(cache_problem, my_args, chunksize=3)


if __name__ == '__main__':
    if 'main.exe' not in os.listdir():
        build_main()
    main()
