:- use_module(library(system), [now/1]).
:- use_module(library(random), [setrand/1]).
:- use_module(library(plunit)).

:- [framework].

:- ensure_loaded(filesystem_handler).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INPUTS PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% planner_command(+PlannerCommand, +PlannerName).
   
% optimal planners
% planner_command('python planners/planningdomains.py', planningdomains).                               
% planner_command('python planners/webplanner.py', webplanner).             
% planner_command('python planners/fd_alias.py 10 seq-opt-fdss-1', fdss1). 
% planner_command('python planners/fd_alias.py 10 seq-opt-fdss-2', fdss2). 
% planner_command('python planners/fd_alias.py 10 seq-opt-lmcut', lmcut). 
% planner_command('python planners/fd_alias.py 10 seq-opt-merge-and-shrink', merge_shrink). 
% planner_command('python planners/fd_alias.py 10 seq-opt-bjolp', bjolp). 

% non-optimal planners (Fast Downward version)
% planner_command('python planners/fd_wastar.py 10 add 10', wastar_add).  
% planner_command('python planners/fd_wastar.py 10 ff 10', wastar_ff).  
% planner_command('python planners/fd_wastar.py 10 gc 10', wastar_gc). 

% mutant planners (Fast Downward version)
% planner_command('python planners/fd_mutant1.py 10 add', mutant1_add).  
% planner_command('python planners/fd_mutant1.py 10 ff', mutant1_ff).  
% planner_command('python planners/fd_mutant1.py 10 gc', mutant1_gc). 
% planner_command('python planners/fd_mutant2.py 10 add', mutant2_add).  
% planner_command('python planners/fd_mutant2.py 10 ff', mutant2_ff). 
% planner_command('python planners/fd_mutant2.py 10 gc', mutant2_gc). 
% planner_command('python planners/fd_mutant3.py 10 add', mutant3_add).  
% planner_command('python planners/fd_mutant3.py 10 ff', mutant3_ff). 
% planner_command('python planners/fd_mutant3.py 10 gc', mutant3_gc). 

% mutant planners
planner_command('./planners/mutated_planner.exe forward dfs_first_solution h_0', dfs_first_solution).
planner_command('./planners/mutated_planner.exe forward a_star_mutant1 h_diff', a_star_mutant1_diff).
planner_command('./planners/mutated_planner.exe forward a_star_mutant1 h_add', a_star_mutant1_add).
planner_command('./planners/mutated_planner.exe forward a_star_mutant1 h_plus', a_star_mutant1_plus).
planner_command('./planners/mutated_planner.exe forward a_star_mutant1 h_length', a_star_mutant1_length).
planner_command('./planners/mutated_planner.exe forward a_star_mutant2 h_diff', a_star_mutant2_diff).
planner_command('./planners/mutated_planner.exe forward a_star_mutant2 h_add', a_star_mutant2_add).
planner_command('./planners/mutated_planner.exe forward a_star_mutant2 h_plus', a_star_mutant2_plus).  
planner_command('./planners/mutated_planner.exe forward a_star_mutant2 h_length', a_star_mutant2_length). 
planner_command('./planners/mutated_planner.exe forward a_star_mutant3 h_diff', a_star_mutant3_diff).  
planner_command('./planners/mutated_planner.exe forward a_star_mutant3 h_add', a_star_mutant3_add).     
planner_command('./planners/mutated_planner.exe forward a_star_mutant3 h_plus', a_star_mutant3_plus).    
planner_command('./planners/mutated_planner.exe forward a_star_mutant3 h_length', a_star_mutant3_length).

%% assert_source_input_predicates(+Directory, +MaximumNumberOfProblemsPerDomain).
assert_source_input_predicates(Directory, MaximumNumberOfProblemsPerDomain) :-
    make_inputs_from_directory(Directory, MaximumNumberOfProblemsPerDomain, Inputs),
    assert_predicates(Inputs).

assert_predicates([]).
assert_predicates([Predicate|T]) :-
    assert(Predicate),
    format('~p added in database.\n', [Predicate]),
    assert_predicates(T).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FRAMEWORK CONFIGURATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% generator_predicate_mr0(+GeneratorPredicate).
generator_predicate_mr0(generator1).
generator_predicate_mr0(generator2).

%% generator_predicate_mr1(+GeneratorPredicate).
generator_predicate_mr1(generator3).
generator_predicate_mr1(generator4).

%% heuristic_predicate(+HeuristicPredicate).
heuristic_predicate(h_zero).
heuristic_predicate(h_cost).
heuristic_predicate(h_reversed_cost).
heuristic_predicate(h_distance_with_i).
heuristic_predicate(h_distance_with_g).
% heuristic_predicate(h_difference_size_with_i).
% heuristic_predicate(h_plus_with_i).
% heuristic_predicate(h_difference_size_with_g).
% heuristic_predicate(h_plus_with_g).
% heuristic_predicate(h_difference_size_with_path).
% heuristic_predicate(h_distance_with_path).

%% generator_for_domain(+DomainFilename, -Generator, -MetamorphicRelation).
generator_for_domain(_, generator1, mr0).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXPERIMENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% make_result_filename(+PlannerName, +Problem, +GeneratorPredicate, +HeuristicPredicate, -ResultFilename).
make_result_filename(PlannerName, Problem, GeneratorPredicate, HeuristicPredicate, ResultFilename) :-
    atom_concat('data/', PlannerName, ProblemNameWithFolder),
    atom_concat(ProblemNameWithFolder, '_', Tmp1),
    problem_name(Problem, ProblemName),
    atom_concat(Tmp1, ProblemName, Tmp2),
    atom_concat(Tmp2, '__', Tmp3),
    GeneratorPredicate =.. [GeneratorName|_GeneratorParameters],
    atom_concat(Tmp3, GeneratorName, Tmp4),
    atom_concat(Tmp4, '__', Tmp5),
    atom_concat(Tmp5, HeuristicPredicate, Tmp6),
    atom_concat(Tmp6, '.csv', ResultFilename).

% gathers pddl inputs for experiments
:- assert_source_input_predicates('experiments', 10).

:- begin_tests(validation_testing).

test(mr0, [nondet, forall((source_input(DomainFilename, ProblemFilename), generator_predicate_mr0(GeneratorPredicate), heuristic_predicate(HeuristicPredicate), planner_command(PlannerCommand, PlannerName)))]) :-
    %%%%%%%%%%%%%%% makes the domain and problem models %%%%%%%%%%%%%%%%
    make_input(DomainFilename, ProblemFilename, Domain-Problem),
    %%%%%%%%%%%%%%% makes the configuration %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    make_result_filename(PlannerName, Problem, GeneratorPredicate, HeuristicPredicate, ResultFilename),
    Configuration = configuration(PlannerCommand, DomainFilename, ProblemFilename, ResultFilename, mr0, 15, true, [GeneratorPredicate], HeuristicPredicate),
    %%%%%%%%%%%%%%% initilises the blackboard %%%%%%%%%%%%%%%%%%%%%%%%%%
    set_blackboard(Configuration, Domain, Problem),
    %%%%%%%%%%%%%%% launches metamorphic testing %%%%%%%%%%%%%%%%%%%%%%%
    test_planner_all(mr0, PlannerCommand, DomainFilename, ProblemFilename, 'tmp/output.txt', [SourceResultCost|TestResults]),
    write('metamorphic testing done\n'),
    %%%%%%%%%%%%%%% exports the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    set_csv_result(Domain, Problem, Configuration, SourceResultCost),
    write_csv_results(ResultFilename, TestResults),
    format('results exported (~a)\n', [ResultFilename]).

test(mr1, [nondet, forall((source_input(DomainFilename, ProblemFilename), generator_predicate_mr1(GeneratorPredicate), heuristic_predicate(HeuristicPredicate), planner_command(PlannerCommand, PlannerName)))]) :-
    %%%%%%%%%%%%%%% makes the domain and problem models %%%%%%%%%%%%%%%%
    make_input(DomainFilename, ProblemFilename, Domain-Problem),
    %%%%%%%%%%%%%%% makes the configuration %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    make_result_filename(PlannerName, Problem, GeneratorPredicate, HeuristicPredicate, ResultFilename),
    Configuration = configuration(PlannerCommand, DomainFilename, ProblemFilename, ResultFilename, mr1, 15, true, [GeneratorPredicate], HeuristicPredicate),
    %%%%%%%%%%%%%%% initilises the blackboard %%%%%%%%%%%%%%%%%%%%%%%%%%
    set_blackboard(Configuration, Domain, Problem),
    %%%%%%%%%%%%%%% launches metamorphic testing %%%%%%%%%%%%%%%%%%%%%%%
    test_planner_all(mr1, PlannerCommand, DomainFilename, ProblemFilename, 'tmp/output.txt', [SourceResultCost|TestResults]),
    write('metamorphic testing done\n'),
    %%%%%%%%%%%%%%% exports the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    set_csv_result(Domain, Problem, Configuration, SourceResultCost),
    write_csv_results(ResultFilename, TestResults),
    format('results exported (~a)\n', [ResultFilename]).

test(generator5, [nondet, forall((source_input(DomainFilename, ProblemFilename), planner_command(PlannerCommand, PlannerName)))]) :-
    HeuristicPredicate = h_zero,
    GeneratorPredicate = generator5,
    (
        for(I, 1, 10),
        param(DomainFilename, ProblemFilename, GeneratorPredicate, HeuristicPredicate, PlannerCommand, PlannerName)
    do
        now(Time),
        setrand(Time),
        format('setrand with ~d\n', [Time]),
        %%%%%%%%%%%%%%% makes the domain and problem models %%%%%%%%%%%%%%%%
        make_input(DomainFilename, ProblemFilename, Domain-Problem),
        %%%%%%%%%%%%%%% makes the configuration %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        number_codes(I, Codes),
        atom_codes(IAtom, Codes),
        atom_concat(HeuristicPredicate, IAtom, IndexedHeuristicPredicate),
        make_result_filename(PlannerName, Problem, GeneratorPredicate, IndexedHeuristicPredicate, ResultFilename),
        Configuration = configuration(PlannerCommand, DomainFilename, ProblemFilename, ResultFilename, mr0, 15, true, [GeneratorPredicate], HeuristicPredicate),
        %%%%%%%%%%%%%%% initilises the blackboard %%%%%%%%%%%%%%%%%%%%%%%%%%
        set_blackboard(Configuration, Domain, Problem),
        %%%%%%%%%%%%%%% launches metamorphic testing %%%%%%%%%%%%%%%%%%%%%%%
        test_planner_all(mr0, PlannerCommand, DomainFilename, ProblemFilename, 'tmp/output.txt', [SourceResultCost|TestResults]),
        %%%%%%%%%%%%%%% exports the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        set_csv_result(Domain, Problem, Configuration, SourceResultCost),
        write_csv_results(ResultFilename, TestResults),
        format('results exported (~a)\n', [ResultFilename])
    ),
    write('metamorphic testing done\n').

test(h_random, [nondet, forall((source_input(DomainFilename, ProblemFilename), planner_command(PlannerCommand, PlannerName)))]) :-
    generator_for_domain(DomainFilename, GeneratorPredicate, MR),
    HeuristicPredicate = h_random,
    (
        for(I, 1, 10),
        param(DomainFilename, ProblemFilename, GeneratorPredicate, HeuristicPredicate, PlannerCommand, PlannerName, MR)
    do
        now(Time),
        setrand(Time),
        format('setrand with ~d\n', [Time]),
        %%%%%%%%%%%%%%% makes the domain and problem models %%%%%%%%%%%%%%%%
        make_input(DomainFilename, ProblemFilename, Domain-Problem),
        %%%%%%%%%%%%%%% makes the configuration %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        number_codes(I, Codes),
        atom_codes(IAtom, Codes),
        atom_concat(HeuristicPredicate, IAtom, IndexedHeuristicPredicate),
        make_result_filename(PlannerName, Problem, GeneratorPredicate, IndexedHeuristicPredicate, ResultFilename),
        Configuration = configuration(PlannerCommand, DomainFilename, ProblemFilename, ResultFilename, MR, 15, true, [GeneratorPredicate], HeuristicPredicate),
        %%%%%%%%%%%%%%% initilises the blackboard %%%%%%%%%%%%%%%%%%%%%%%%%%
        set_blackboard(Configuration, Domain, Problem),
        %%%%%%%%%%%%%%% launches metamorphic testing %%%%%%%%%%%%%%%%%%%%%%%
        test_planner_all(MR, PlannerCommand, DomainFilename, ProblemFilename, 'tmp/output.txt', [SourceResultCost|TestResults]),
        %%%%%%%%%%%%%%% exports the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        set_csv_result(Domain, Problem, Configuration, SourceResultCost),
        write_csv_results(ResultFilename, TestResults),
        format('results exported (~a)\n', [ResultFilename])
    ),
    write('metamorphic testing done\n').

:- end_tests(validation_testing).