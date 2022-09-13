import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_chart_heuristics_score(filename: str):
    """
    Saves a chart that reports the number of non-optimal detections for each problem and heuristic (as a stacked bar chart).
    """
    scores = { h: [] for h in heuristics }
    nb_problems = 0
    labels = []
    for domain in domains:
        domain_df = df.loc[df.domain==domain]
        problems = domain_df['problem'].unique().tolist()
        problems.sort()
        for problem in problems:
            labels.append(f'{domain}-{problem}')
            nb_problems += 1
            problem_df = domain_df.loc[domain_df.problem==problem]
            # for now no filtering with generators and planners/mutants
            for heuristic in heuristics:
                heuristic_df = problem_df.loc[problem_df.heuristic_name == heuristic]
                scores[heuristic].append(len(heuristic_df.loc[heuristic_df.failure==1]))        
    # plotting
    _, ax = plt.subplots()
    for i in range(len(heuristics) - 1, 0, -1):
        ax.bar(np.arange(nb_problems), list(scores.values())[i], 0.35, label=list(scores.keys())[i], bottom=[sum(value) for value in zip(*[list(scores.values())[v] for v in range(0, i)])])
    ax.bar(np.arange(nb_problems), list(scores.values())[0], 0.35, label=list(scores.keys())[0])
    ax.set_ylabel('Faulty test cases')
    ax.set_title('Number of fault-revealing follow-up test cases for each heuristic per problem', size=11, x=0.45)
    plt.xticks(np.arange(nb_problems), labels, rotation=45)
    # plt.subplots_adjust(bottom=0.15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)


def save_chart_heuristics_score_per_domain(filename: str):
    """
    Saves a chart that reports the number of non-optimal detections for each heuristic per domain (as a stacked bar chart).
    """
    scores = { h: [] for h in heuristics }
    for heuristic in heuristics:
        heuristic_df = df.loc[df.heuristic_name == heuristic]
        for domain in domains:
            domain_df = heuristic_df.loc[heuristic_df.domain==domain]
            # for now no filtering with generators and planners/mutants
            scores[heuristic].append(len(domain_df.loc[domain_df.failure==1]))        
    # plotting
    _, ax = plt.subplots()
    bar_width = 0.12
    x = np.arange(len(domains))
    for i in range(len(heuristics)):
        heuristic = heuristics[i]
        x_offset = (i - len(heuristics) / 2) * bar_width + bar_width / 2
        ax.bar(x + x_offset, scores[heuristic], width=bar_width, label=heuristics_dict[heuristic])
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel('Faulty test cases')
    ax.set_title('Number of fault-revealing follow-up test cases for each heuristic per domain', size=11, x=0.45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)


def save_chart_heuristics_coverage(filename: str):
    """
    Saves a chart that reports for each heuristic the percentage of problems where non-optimal behavior was detected.  
    """
    coverages = { h: 0 for h in heuristics }
    nb_problems = 0
    # must loop over domains because some problems may have identical names (across multiple domains) 
    for domain in domains:
        domain_df = df.loc[df.domain==domain]
        problems = domain_df['problem'].unique().tolist()
        for problem in problems:
            nb_problems += 1
            problem_df = domain_df.loc[domain_df.problem==problem]
            # for now no filtering with generators and planners/mutants
            for heuristic in heuristics:
                heuristic_df = problem_df.loc[problem_df.heuristic_name == heuristic]
                if heuristic_df.empty == False and len(heuristic_df.loc[heuristic_df.failure==1]) != 0:
                    coverages[heuristic] += 1
    # plotting
    _, ax = plt.subplots()
    ax.bar(list(coverages.keys()), [100 * (p / nb_problems) for p in list(coverages.values())])
    ax.set_ylabel('Coverage of problems (%)')
    ax.set_title('Percentage of problems leading to fault detection for each heuristic')
    plt.xticks(np.arange(len(heuristics)), [heuristics_dict[h] for h in list(coverages.keys())], rotation=45)
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig(filename)


def save_chart_heuristics_coverage_per_domain(filename: str):
    """
    Saves a chart that reports for each heuristic the percentage of problems where non-optimal behavior was detected per domain.  
    """
    coverages = { h: [] for h in heuristics }
    for heuristic in heuristics:
        heuristic_df = df.loc[df.heuristic_name == heuristic]
        for domain in domains:
            domain_df = heuristic_df.loc[heuristic_df.domain==domain]
            # for now no filtering with generators and planners/mutants
            problems = domain_df['problem'].unique().tolist()
            count = 0
            for problem in problems:
                problem_df = domain_df.loc[domain_df.problem==problem]    
                if problem_df.empty == False and len(problem_df.loc[problem_df.failure==1]) != 0:
                    count += 1
            coverages[heuristic].append(100 * (count / len(problems)) if len(problems) != 0 else 0)
    # plotting
    _, ax = plt.subplots()
    bar_width = 0.12
    x = np.arange(len(domains))
    for i in range(len(heuristics)):
        heuristic = heuristics[i]
        x_offset = (i - len(heuristics) / 2) * bar_width + bar_width / 2
        ax.bar(x + x_offset, coverages[heuristic], width=bar_width, label=heuristics_dict[heuristic])
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel('Coverage of problems (%)')
    ax.set_title('Percentage of problems leading to fault detection for each heuristic per domain', size=11, x=0.45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)


def save_chart_heuristics_efficiency_per_domain(filename: str):
    """
    Saves a chart that describes the efficiency of each heuristic by reporting the percentage of successful follow-up test cases. 
    """
    efficiencies = { h: [] for h in heuristics }
    for heuristic in heuristics:
        heuristic_df = df.loc[df.heuristic_name==heuristic]
        # for now no filtering with generators and planners/mutants
        for domain in domains:
            domain_df = heuristic_df.loc[heuristic_df.domain==domain]
            efficiencies[heuristic].append(100 * len(domain_df.loc[domain_df.failure==1]) / len(domain_df))
    # plotting
    _, ax = plt.subplots()
    bar_width = 0.12
    x = np.arange(len(domains))
    for i in range(len(heuristics)):
        heuristic = heuristics[i]
        x_offset = (i - len(heuristics) / 2) * bar_width + bar_width / 2
        ax.bar(x + x_offset, efficiencies[heuristic], width=bar_width, label=heuristics_dict[heuristic])
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel('Fault-revealing follow-up test cases (%)')
    ax.set_title('Efficiency of each heuristic per domain')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)


def save_chart_mutation_score(filename: str, metric: str) -> None:
    """
    Saves a chart that reports the mutation score for each heuristic. The metric can either be:
        - the number of fault-revealing follow-up test cases
        - the number of fault-revealing problems
    """
    scores = { h: [] for h in heuristics }
    for planner in planners:
        planner_df = df.loc[df.planner==planner]
        for heuristic in heuristics:
            score = 0
            heuristic_df = planner_df.loc[planner_df.heuristic_name==heuristic]
            for domain in domains:
                domain_df = heuristic_df.loc[heuristic_df.domain==domain]
                problems = domain_df['problem'].unique().tolist()
                for problem in problems:
                    problem_df = domain_df.loc[domain_df.problem==problem]
                    # scores 1 per problem if the metric is 'problems'
                    if metric == 'problems':
                        if problem_df.loc[problem_df.failure==1].empty == False:
                            score += 1
                    # scores the number follow-up test cases performed if the metric is 'cases'
                    if metric == 'cases':
                        score += len(problem_df.loc[problem_df.failure==1])
            scores[heuristic].append(score)
    # plotting
    _, ax = plt.subplots()
    bar_width = 0.12
    x = np.arange(len(planners))
    for i in range(len(heuristics)):
        heuristic = heuristics[i]
        x_offset = (i - len(heuristics) / 2) * bar_width + bar_width / 2
        ax.bar(x + x_offset, scores[heuristic], width=bar_width, label=heuristics_dict[heuristic])
    ax.set_xticks(x)
    ax.set_xticklabels(planners, rotation=45)
    ax.set_ylabel('Number of fault-revealing problems' if metric == 'problems' else 'Number of fault-revealing test cases')
    ax.set_title('Mutation scores per planner/mutant for each heuristic')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)

heuristics_dict = {
    'min_h_distance_with_i': 'min_h_dist_i',
    'max_h_distance_with_i': 'max_h_dist_i',
    'min_h_distance_with_g': 'min_h_dist_g',
    'max_h_distance_with_g': 'max_h_dist_g',
    'h_random': 'h_random',
}


#######################################################
## SCRIPT SECTION / MAIN
#######################################################


df = pd.read_csv('prolog_results_experiments.csv', header=0)
domains = df['domain'].unique().tolist()
domains.sort()
problems = df['problem'].unique().tolist()
planners = df['planner'].unique().tolist()
generators = df['generator'].unique().tolist()
heuristics = df['heuristic_name'].unique().tolist()

prefix = 'prolog_'
# general charts
save_chart_heuristics_score(prefix + 'score.png')
save_chart_heuristics_coverage(prefix + 'coverage.png')
# charts where results are regrouped per domain
save_chart_heuristics_score_per_domain(prefix + 'score_per_domain.png')
save_chart_heuristics_coverage_per_domain(prefix + 'coverage_per_domain.png')
save_chart_heuristics_efficiency_per_domain(prefix + 'efficiency_per_domain.png')
# charts that detail results wrt the planners
save_chart_mutation_score(prefix + 'mutation_score_problems.png', 'problems')
save_chart_mutation_score(prefix + 'mutation_score_cases.png', 'cases')