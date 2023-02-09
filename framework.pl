:- use_module(library(process), [process_create/3]).
:- use_module(library(lists), [maplist/3, keys_and_values/3, reverse/2]).
:- use_module(library(system), [now/1]).
:- use_module(library(random), [setrand/1]).
:- use_module(library(file_systems), [file_exists/1]).

:- ensure_loaded(pddl_parser).
:- ensure_loaded(pddl_serialiser).
:- ensure_loaded(blackboard_data).
:- ensure_loaded(configuration).
:- ensure_loaded(problem).
:- ensure_loaded(results).
:- ensure_loaded(generators).
:- ensure_loaded(cache).
:- ensure_loaded(domain).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SYSTEM COMMANDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% /!\ NO SAFE PREDICATE
remove_files([]).
remove_files([Filename|T]) :-
    process_create(path(powershell), ['-Command', 'rm', Filename], [wait(_ExitStatus)]),
    remove_files(T).

remove_files_from_txt_file(Filepath) :-
    atom_concat(Filename, '.txt', Filepath),
    atom_concat(Filename, '*', Pattern),
    process_create(path(powershell), ['-Command', 'rm', Pattern], [wait(_ExitStatus)]).

remove_tmp_files :-
    process_create(path(powershell), ['-Command', 'rm', 'tmp/*.txt'], [wait(_ExitStatus)]),
    process_create(path(powershell), ['-Command', 'rm', 'tmp/*.pddl'], [wait(_ExitStatus)]).

%% run_planner_command(+Command, +DomainFilepath, +ProblemFilepath, +ResultFilename, -ExitStatus, -ExecutionTime).
run_planner_command(Command, DomainFilepath, ProblemFilepath, ResultFilename, ExitStatus, ExecutionTime) :-
    statistics(walltime, [StartTime, _]),
    process_create(path(powershell), ['-Command', Command, DomainFilepath, ProblemFilepath, ResultFilename], [wait(ExitStatus)]),
    statistics(walltime, [CurrentTime, _]),
    ExecutionTime is (CurrentTime - StartTime) / 1000.

%% run_planner_commands(+PlannerCommand, +DomainFilepath, +ProblemFilepaths, -ResultsFilenames, -ExecutionTimes).
run_planner_commands(_PlannerCommand, _DomainFilepath, [], [], []).
run_planner_commands(PlannerCommand, DomainFilepath, [ProblemFilepath|T1], [ResultFilename|T2], [ExecutionTime|T3]) :-
    atom_concat(ProblemName, '.pddl', ProblemFilepath),
    atom_concat(ProblemName, '.txt', ResultFilename),
    run_planner_command(PlannerCommand, DomainFilepath, ProblemFilepath, ResultFilename, _ExitStatus, ExecutionTime),
    run_planner_commands(PlannerCommand, DomainFilepath, T1, T2, T3).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NODE PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% node(+State, +Cost).
node(_State, _Cost).

generate_node_generator_pairs(Generators, NumberOfTests, Results) :-
    % retrieves in a list every nodes set (for each generator)
    (
        foreach(Generator, Generators),
        fromto([], In, Out, List),
        param(NumberOfTests)
    do
        Generator =.. [GeneratorName|GeneratorArguments],
        % appends at the end of the arguments list the number of nodes and a variable to unify the result
        append(GeneratorArguments, [NumberOfTests, GeneratedNodes], Arguments),
        GeneratorPredicate =.. [GeneratorName|Arguments],
        GeneratorPredicate,
        Out = [GeneratedNodes-Generator|In],
        write(Generator), write(' : '), length(GeneratedNodes, GNL), write(GNL), nl
    ),
    % flattens the list and adds to the nodes their generator
    (
        foreach(Nodes-Generator, List),
        fromto([], In, Out, Results)
    do
        maplist(add_key(Generator), Nodes, Pairs),
        append(In, Pairs, Out)
    ).

%% add_key(+Key, +Element, -Result).
add_key(Key, Element, Element-Key).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INPUTS TESTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% make_follow_up_inputs(-FollowUpInputs).
make_follow_up_inputs(FollowUpInputs) :-
    get_configuration(Configuration),
    configuration_generators(Configuration, Generators),
    configuration_nb_tests(Configuration, NumberOfTests),
    generate_node_generator_pairs(Generators, NumberOfTests, Pairs),
    get_problem(Problem),
    (
        foreach(Node-Generator, Pairs),
        foreach(FollowUpInput, FollowUpInputs),
        param(Problem)
    do
        Problem = problem(Name, D, R, OD, _, G, C, MS, LS),
        Node = node(State, _Cost),
        FollowUpProblem = problem(Name, D, R, OD, State, G, C, MS, LS),
        FollowUpInput = input(FollowUpProblem, Node, Generator)
    ).

%% serialise_follow_up_inputs(+Inputs, +SourceResultFilepath, -InputsFilenames, +Index).
serialise_follow_up_inputs([], _SourceResultFilepath, [], _Index).
serialise_follow_up_inputs([Input|T1], SourceResultFilepath, [Filename|T2], Index) :-
    serialise_follow_up_input(Input, SourceResultFilepath, Filename, Index),
    NewIndex is Index + 1,
    serialise_follow_up_inputs(T1, SourceResultFilepath, T2, NewIndex).

serialise_follow_up_input(Input, SourceResultFilepath, Filename, Index) :-
    Input = input(Problem, _Node, _Generator),
    number_codes(Index, Codes),
    atom_codes(IndexAtom, Codes),
    atom_concat(SourceFilename, '.txt', SourceResultFilepath),
    atom_concat(SourceFilename, IndexAtom, Name),
    atom_concat(Name, '.pddl', Filename),
    serialise_problem(Problem, Filename).

%% make_input(+DomainFilepath, +ProblemFilepath, -Domain, -Problem, CacheName).
make_input(DomainFilepath, ProblemFilepath, Domain, Problem, CacheName) :-
    atom_concat('benchmarks/', TmpD, DomainFilepath),
    atom_chars(TmpD, TmpDChars),
    read_until_slash(TmpDChars, DomainChars),
    atom_chars(DomainName, DomainChars),
    filename_from_path(ProblemFilepath, ProblemFilename),
    atom_concat(DomainName, '_', Tmp),
    atom_concat(Tmp, ProblemFilename, Tmp2),
    atom_concat(Tmp2, '.txt', Tmp3),
    atom_concat('cache/', Tmp3, CacheName),
    parse_domain(DomainFilepath, Domain),
    parse_problem(ProblemFilepath, TmpProblem),
    sort_problem(TmpProblem, Problem).

filename_from_path(FilePath, Filename) :-
    atom_concat(Path, '.pddl', FilePath),
    atom_chars(Path, Chars),
    reverse(Chars, ReversedChars),
    append(ReversedFilenameChars, ['/'|_], ReversedChars),
    !,
    reverse(ReversedFilenameChars, FilenameChars),
    atom_chars(Filename, FilenameChars).

read_until_slash([], []).
read_until_slash(['/'|_], []) :-
    !.
read_until_slash([Char|T1], [Char|T2]) :-
    read_until_slash(T1, T2).

atom_to_int(Atom, Integer) :-
    atom_codes(Atom, Codes),
    number_codes(Integer, Codes).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RELATION CHECKING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% check_metamorphic_relation(+SourceResult, +FollowUpInputs, +FollowUpResults,-Results).
check_metamorphic_relation(SourceResult, FollowUpInputs, FollowUpResults, Results) :-
    length(SourceResult, SourceResultLength),
    (
        foreach(FollowUpResult, FollowUpResults),
        foreach(FollowUpInput, FollowUpInputs),
        foreach(Result, Results),
        param(SourceResultLength)
    do
        FollowUpInput = input(_Problem, Node, Generator),
        Node = node(_State, Cost),
        length(FollowUpResult, FollowUpResultLength),
        (   FollowUpResultLength = 0 -> (Failure = 0, Error = 1)
        ;   SourceResultLength > Cost + FollowUpResultLength -> (Failure = 1, Error = 0)
        ;
            (Failure = 0, Error = 0)
        ),
        Result = result(Node, Generator, Failure, Error, _ExecutionTime, FollowUpResultLength)
    ).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FRAMEWORK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

user:runtime_entry(start) :-
    start.

start :-
    prolog_flag(argv, [ConfigurationFilename]),
    !,
    now(Time),
    setrand(Time),
    format('setrand with ~d\n', [Time]),
    main_read_configuration(ConfigurationFilename),
    halt.
start :-
    prolog_flag(argv, ['--cache', DomainFilepath, ProblemFilepath, OutputFilename|PlannersCommands]),
    !,
    make_mutants_killers_nodes_cache(DomainFilepath, ProblemFilepath, OutputFilename, PlannersCommands),
    halt.
start :-
    prolog_flag(argv, ['--test', DomainFilepath, ProblemFilepath, OutputFilename|PlannersCommands]),
    !,
    make_deep_mutants_cache(DomainFilepath, ProblemFilepath, OutputFilename, PlannersCommands, ExecutionTimes),
    write_indexes(ExecutionTimes),
    halt.
start :-
    prolog_flag(argv, ['--state_generation', DomainFilepath, ProblemFilepath, OutputFilepath]),
    !,
    make_input(DomainFilepath, ProblemFilepath, Domain, Problem, _CacheName),
    set_blackboard(Domain, Problem, _CacheName),
    statistics(walltime, [StartTime, _]),
    bfs_generator(500, Nodes),
    save_nodes(OutputFilepath, Nodes),
    statistics(walltime, [CurrentTime, _]),
    ExecutionTime is (CurrentTime - StartTime) / 1000,
    write(ExecutionTime),
    nl,
    halt.
start :-
    prolog_flag(argv, ['--random_walk_generation', DomainFilepath, ProblemFilepath, NAtom, SRCAtom]),
    !,
    now(Time),
    setrand(Time),
    statistics(walltime, [StartTime, _]),
    atom_to_int(NAtom, N),
    atom_to_int(SRCAtom, MaximumWalkLength),
    make_input(DomainFilepath, ProblemFilepath, Domain, Problem, CacheName),
    set_blackboard(Domain, Problem, CacheName),
    load_nodes_or_generate(CacheName, Nodes),
    compute_reachable_nodes(N, MaximumWalkLength, Nodes, NewN),
    random_walks_generator(NewN, MaximumWalkLength, _Results),
    statistics(walltime, [CurrentTime, _]),
    ExecutionTime is (CurrentTime - StartTime) / 1000,
    write(ExecutionTime),
    nl,
    halt.
start :-
    prolog_flag(argv, ['--bfs', DomainFilepath, ProblemFilepath, NAtom, SRCAtom]),
    !,
    statistics(walltime, [StartTime, _]),
    bfs_nodes_indexes(DomainFilepath, ProblemFilepath, NAtom, SRCAtom, Indexes),
    statistics(walltime, [CurrentTime, _]),
    ExecutionTime is (CurrentTime - StartTime) / 1000,
    append(Indexes, [ExecutionTime], Results),
    write_indexes(Results),
    halt.
start :-
    prolog_flag(argv, ['--min_dist_i', DomainFilepath, ProblemFilepath, NAtom, SRCAtom]),
    !,
    statistics(walltime, [StartTime, _]),
    dist_i_nodes_indexes(DomainFilepath, ProblemFilepath, min, NAtom, SRCAtom, Indexes),
    statistics(walltime, [CurrentTime, _]),
    ExecutionTime is (CurrentTime - StartTime) / 1000,
    append(Indexes, [ExecutionTime], Results),
    write_indexes(Results),
    halt.
start :-
    prolog_flag(argv, ['--max_dist_i', DomainFilepath, ProblemFilepath, NAtom, SRCAtom]),
    !,
    statistics(walltime, [StartTime, _]),
    dist_i_nodes_indexes(DomainFilepath, ProblemFilepath, max, NAtom, SRCAtom, Indexes),
    statistics(walltime, [CurrentTime, _]),
    ExecutionTime is (CurrentTime - StartTime) / 1000,
    append(Indexes, [ExecutionTime], Results),
    write_indexes(Results),
    halt.
start :-
    prolog_flag(argv, ['--min_dist_g', DomainFilepath, ProblemFilepath, NAtom, SourceResultFilepath]),
    !,
    statistics(walltime, [StartTime, _]),
    dist_g_nodes_indexes(DomainFilepath, ProblemFilepath, min, NAtom, SourceResultFilepath, Indexes),
    statistics(walltime, [CurrentTime, _]),
    ExecutionTime is (CurrentTime - StartTime) / 1000,
    append(Indexes, [ExecutionTime], Results),
    write_indexes(Results),
    halt.
start :-
    prolog_flag(argv, ['--max_dist_g', DomainFilepath, ProblemFilepath, NAtom, SourceResultFilepath]),
    !,
    statistics(walltime, [StartTime, _]),
    dist_g_nodes_indexes(DomainFilepath, ProblemFilepath, max, NAtom, SourceResultFilepath, Indexes),
    statistics(walltime, [CurrentTime, _]),
    ExecutionTime is (CurrentTime - StartTime) / 1000,
    append(Indexes, [ExecutionTime], Results),
    write_indexes(Results),
    halt.
start :-
    prolog_flag(argv, ['--random', DomainFilepath, ProblemFilepath, NAtom, SRCAtom]),
    !,
    now(Time),
    setrand(Time),
    statistics(walltime, [StartTime, _]),
    random_nodes_indexes(DomainFilepath, ProblemFilepath, NAtom, SRCAtom, Indexes),
    statistics(walltime, [CurrentTime, _]),
    ExecutionTime is (CurrentTime - StartTime) / 1000,
    append(Indexes, [ExecutionTime], Results),
    write_indexes(Results),
    halt.
start :-
    prolog_flag(argv, Arguments),
    now(Time),
    setrand(Time),
    format('setrand with ~d\n', [Time]),
    main_with_configuration(Arguments),
    halt.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INDEXES RELATED PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bfs_nodes_indexes(DomainFilepath, ProblemFilepath, NAtom, SRCAtom, Results) :-
    atom_to_int(NAtom, N),
    atom_to_int(SRCAtom, SRC),
    make_input(DomainFilepath, ProblemFilepath, Domain, Problem, CacheName),
    set_blackboard(Domain, Problem, CacheName),
    select_bfs_indexes(N, SRC, Results).

random_nodes_indexes(DomainFilepath, ProblemFilepath, NAtom, SRCAtom, Results) :-
    atom_to_int(NAtom, N),
    atom_to_int(SRCAtom, SRC),
    make_input(DomainFilepath, ProblemFilepath, Domain, Problem, CacheName),
    set_blackboard(Domain, Problem, CacheName),
    select_random_indexes(N, SRC, Results).

dist_i_nodes_indexes(DomainFilepath, ProblemFilepath, MinOrMax, NAtom, SRCAtom, Results) :-
    atom_to_int(NAtom, N),
    atom_to_int(SRCAtom, SRC),
    make_input(DomainFilepath, ProblemFilepath, Domain, Problem, CacheName),
    set_blackboard(Domain, Problem, CacheName),
    load_nodes_or_generate(CacheName, Nodes),
    length(Nodes, NN),
    IMax is NN - 1,
    problem_initial_state(Problem, InitialState),
    (
        foreach(node(State, Cost), Nodes),
        for(I, 0, IMax),
        foreach(H-pair(Cost, I), IndexedPairs),
        param(InitialState)
    do
        compute_distance_between_states(InitialState, State, H)
    ),
    keysort(IndexedPairs, SortedIndexedPairs),
    keys_and_values(SortedIndexedPairs, _, Pairs),
    (
        MinOrMax == max -> (reverse(Pairs, ReversedPairs), get_indexes(ReversedPairs, N, SRC, Results))
    ;
        get_indexes(Pairs, N, SRC, Results)
    ).

dist_g_nodes_indexes(DomainFilepath, ProblemFilepath, MinOrMax, NAtom, SourceResultFilepath, Results) :-
    atom_to_int(NAtom, N),
    make_input(DomainFilepath, ProblemFilepath, Domain, Problem, CacheName),
    set_blackboard(Domain, Problem, CacheName),
    deserialise_plan(SourceResultFilepath, SourceResult),
    validate_plan(SourceResult),
    length(SourceResult, SRC),
    load_nodes_or_generate(CacheName, Nodes),
    length(Nodes, NN),
    IMax is NN - 1,
    get_final_state(FinalState),
    (
        foreach(node(State, Cost), Nodes),
        for(I, 0, IMax),
        foreach(H-pair(Cost, I), IndexedPairs),
        param(FinalState)
    do
        compute_distance_between_states(FinalState, State, H)
    ),
    keysort(IndexedPairs, SortedIndexedPairs),
    keys_and_values(SortedIndexedPairs, _, Pairs),
    (
        MinOrMax == max -> (reverse(Pairs, ReversedPairs), get_indexes(ReversedPairs, N, SRC, Results))
    ;
        get_indexes(Pairs, N, SRC, Results)
    ).

get_indexes([], _, _, []) :-
    !.
get_indexes(_, 0, _, []) :-
    !.
get_indexes([pair(Cost, Index)|T1], N, MaximumCost, [Index|T2]) :-
    Cost < MaximumCost,
    % format('~d < ~d\n', [Cost, MaximumCost]),
    NewN is N - 1,
    !,
    get_indexes(T1, NewN, MaximumCost, T2).
get_indexes([_|T], N, MaximumCost, Indexes) :-
    get_indexes(T, N, MaximumCost, Indexes).

write_indexes([]).
write_indexes([H|T]) :-
    write(H),
    nl,
    write_indexes(T).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% METAMORPHIC TESTING PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


main_read_configuration(ConfigurationFilename) :-
    %%%%%%%%%%%%%%% manages the configuration reading %%%%%%%%%%%%%%%%%%
    deserialise_configuration(ConfigurationFilename, Configuration),
    configuration_domain_filename(Configuration, DomainFilepath),
    configuration_problem_filename(Configuration, ProblemFilepath),
    configuration_planner_command(Configuration, PlannerCommand),
    configuration_result_filename(Configuration, CsvFilename),
    configuration_output_filename(Configuration, OutputFilename),
    %%%%%%%%%%%%%%% makes the domain and problem models %%%%%%%%%%%%%%%%
    make_input(DomainFilepath, ProblemFilepath, Domain, Problem, CacheName),
    %%%%%%%%%%%%%%% initilises the blackboard %%%%%%%%%%%%%%%%%%%%%%%%%%
    set_blackboard(Configuration, Domain, Problem, CacheName),
    %%%%%%%%%%%%%%% launches metamorphic testing %%%%%%%%%%%%%%%%%%%%%%%
    test_planner(PlannerCommand, DomainFilepath, ProblemFilepath, OutputFilename, [SourceResultCost|TestResults]),
    write('metamorphic testing done\n'),
    %%%%%%%%%%%%%%% exports the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    set_csv_result(Domain, Problem, Configuration, SourceResultCost),
    write_csv_results(CsvFilename, TestResults),
    format('results exported (~a)\n', [CsvFilename]).

main_with_configuration([DomainFilepath, ProblemFilepath, PlannerCommand, NBTestsAtom, CsvFilename, OutputFilename|GeneratorsPredicates]) :-
    atom_to_int(NBTestsAtom, NBTests),
    Configuration = configuration(PlannerCommand, DomainFilepath, ProblemFilepath, CsvFilename, OutputFilename, NBTests, GeneratorsPredicates),
    %%%%%%%%%%%%%%% makes the domain and problem models %%%%%%%%%%%%%%%%
    make_input(DomainFilepath, ProblemFilepath, Domain, Problem, CacheName),
    %%%%%%%%%%%%%%% initilises the blackboard %%%%%%%%%%%%%%%%%%%%%%%%%%
    set_blackboard(Configuration, Domain, Problem, CacheName),
    %%%%%%%%%%%%%%% launches metamorphic testing %%%%%%%%%%%%%%%%%%%%%%%
    test_planner(PlannerCommand, DomainFilepath, ProblemFilepath, OutputFilename, [SourceResultCost|TestResults]),
    write('metamorphic testing done\n'),
    %%%%%%%%%%%%%%% exports the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    set_csv_result(Domain, Problem, Configuration, SourceResultCost),
    write_csv_results(CsvFilename, TestResults),
    format('results exported (~a)\n', [CsvFilename]).

test_planner(PlannerCommand, DomainFilepath, ProblemFilepath, SourceResultFilepath, [SourceResultCost|TestResults]) :-
    run_planner_command(PlannerCommand, DomainFilepath, ProblemFilepath, SourceResultFilepath, _ExitStatus, _ExecutionTime),
    write('initial problem run\n'),
    deserialise_plan(SourceResultFilepath, SourceResult),
    write('source result deserialised'),
    validate_plan(SourceResult),
    !,
    length(SourceResult, SourceResultCost),
    format(' (of cost ~d)\n', [SourceResultCost]),
    set_source_result(SourceResult),
    write('blackboard updated with the source result\n'),
    make_follow_up_inputs(FollowUpInputs),
    length(FollowUpInputs, L),
    format('follow-up inputs made (~d)\n', [L]),
    % serialises the follow-up inputs
    serialise_follow_up_inputs(FollowUpInputs, SourceResultFilepath, FollowUpInputsFilepaths, 0),
    write('follow-up inputs serialised\n'),
    % runs the planner on every follow-up input
    run_planner_commands(PlannerCommand, DomainFilepath, FollowUpInputsFilepaths, FollowUpResultsFilepaths, ExecutionTimes),
    write('follow-up problems run\n'),
    % deserialises all the follow-up results
    deserialise_plans(FollowUpResultsFilepaths, FollowUpResults),
    write('follow-up results deserialised\n'),
    % checks the metamorphic relation for every follow-up test case result
    check_metamorphic_relation(SourceResult, FollowUpInputs, FollowUpResults, Results),
    % completes the results by adding the previously retrieved execution times (when running planner)
    set_execution_times(Results, ExecutionTimes, TestResults),
    % removes all the temporary generated files (safe but slow)
    % append([SourceResultFilepath|FollowUpInputsFilepaths], FollowUpResultsFilepaths, FilenamesToRemove),
    % remove_files(FilenamesToRemove).
    % HOPES to remove only the temporary generated files (fast but unsafe)
    remove_files_from_txt_file(SourceResultFilepath).
test_planner(_, _, _, _, [0]) :-
    write('\nsomething went wrong during metamorphic testing (empty source result?).\n').

%% set_execution_times(+Results, +ExecutionTimes, -TestResults).
set_execution_times([], [], []).
set_execution_times([result(N, G, F, E, _, RC)|T1], [ExecutionTime|T2], [result(N, G, F, E, ExecutionTime, RC)|T3]) :-
    set_execution_times(T1, T2, T3).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CACHE PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


make_mutants_killers_nodes_cache(DomainFilepath, ProblemFilepath, _, _) :-
    make_input(DomainFilepath, ProblemFilepath, _, _, CacheName),
    atom_concat(Tmp, '.txt', CacheName),
    atom_concat(Tmp, '_mt.txt', MTCacheName),
    file_exists(MTCacheName),
    !,
    format('cache already generated (~a). Procedure aborted.\n', [MTCacheName]).
make_mutants_killers_nodes_cache(DomainFilepath, ProblemFilepath, OutputFilename, PlannersCommands) :-
    make_input(DomainFilepath, ProblemFilepath, Domain, Problem, CacheName),
    set_blackboard(Domain, Problem, CacheName), % needed for plan validation and bfs generator if empty cache
    make_commands_pairs(DomainFilepath, ProblemFilepath, OutputFilename, PlannersCommands, CommandsPairs),
    load_nodes_or_generate(CacheName, Nodes),
    select_nodes(DomainFilepath, ProblemFilepath, Problem, Nodes, 0, CommandsPairs, OutputFilename, SelectedNodes),
    atom_concat(Tmp, '.txt', CacheName),
    atom_concat(Tmp, '_mt.txt', MTCacheName),
    save_nodes(MTCacheName, SelectedNodes),
    length(SelectedNodes, L),
    format('~a cache file successfully created (~d nodes selected).\n', [MTCacheName, L]).

make_commands_pairs(DomainFilepath, ProblemFilepath, OutputFilename, PlannersCommands, CommandsPairs) :-
    (
        foreach(PlannerCommand, PlannersCommands),
        foreach(ResultCost-PlannerCommand, IndexedCommands),
        param(DomainFilepath, ProblemFilepath, OutputFilename)
    do
        run_planner_command(PlannerCommand, DomainFilepath, ProblemFilepath, OutputFilename, _ExitStatus, _ExecutionTime),
        deserialise_plan(OutputFilename, Result),
        validate_plan(Result),
        length(Result, ResultCost)
    ),
    sort(IndexedCommands, CommandsPairs).

select_nodes(_DomainFilepath, _ProblemFilepath, _Problem, [], _Index, _CommandsPairs, _OutputFilename, []).
select_nodes(DF, PF, problem(N, D, R, OD, _, G, C, MS, LS), [node(State, Cost)|T1], Index, CP, OF, [node(State, Cost)|T2]) :-
    NewProblem = problem(N, D, R, OD, State, G, C, MS, LS),
    atom_concat(Tmp, '.txt', OF),
    atom_concat(Tmp, '.pddl', NewProblemFilePath),
    serialise_problem(NewProblem, NewProblemFilePath),
    (
        foreach(SRC-Command, CP),
        param(DF, NewProblemFilePath, OF, Cost)
    do
        run_planner_command(Command, DF, NewProblemFilePath, OF, _, _),
        deserialise_plan(OF, Result),
        length(Result, L),
        L \== 0,
        SRC > Cost + L
        % format('~d > ~d + ~d\n', [SRC, Cost, L])
    ),
    format('node number ~d kills all the mutants.\n', [Index]),
    NewIndex is Index + 1,
    select_nodes(DF, PF, problem(N, D, R, OD, _, G, C, MS, LS), T1, NewIndex, CP, OF, T2).
select_nodes(DF, PF, problem(N, D, R, OD, _, G, C, MS, LS), [_|T1], Index, CP, OF, SelectedNodes) :-
    % format('node number ~d dropped.\n', [Index]),
    NewIndex is Index + 1,
    select_nodes(DF, PF, problem(N, D, R, OD, _, G, C, MS, LS), T1, NewIndex, CP, OF, SelectedNodes).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DEEP RESULTS PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% pedicate that runs all the cached nodes of a problem on the mutants and stores the results in a .csv file
make_deep_mutants_cache(DomainFilepath, ProblemFilepath, Output, PlannersCommands, [TotalExecutionTime|ExecutionTimes]) :-
    statistics(walltime, [StartTime, _]),
    make_input(DomainFilepath, ProblemFilepath, Domain, Problem, CacheName),
    set_blackboard(Domain, Problem, CacheName), % needed for plan validation and bfs generator if empty cache
    make_commands(DomainFilepath, ProblemFilepath, Output, PlannersCommands, Commands), % involves running the mutants on the source problems
    load_nodes_or_generate(CacheName, Nodes),
    atom_concat(Tmp, '.txt', CacheName),
    % Output ends with '.txt' means the .csv filename is based on CacheName, otherwise Output is used to create the .csv filename as well as the tmp .txt and .pddl files
    (atom_concat(_, '.txt', Output) -> (atom_concat(Tmp, '.csv', CsvFilename), OutputFilename = Output) ; (atom_concat(Output, '.csv', CsvFilename), atom_concat(Output, '.txt', OutputFilename))),
    set_nodes_csv_result(CsvFilename),
    test_mutants(DomainFilepath, Problem, 0, Nodes, Commands, OutputFilename, CsvFilename, MutantExecutionTimesList),
    (
        foreach(MutantExecutionTimes, MutantExecutionTimesList),
        fromto([], In, Out, ExecutionTimes)
    do
        append(MutantExecutionTimes, In, Out)
    ),
    statistics(walltime, [CurrentTime, _]),
    TotalExecutionTime is (CurrentTime - StartTime) / 1000.


test_mutants(_DomainFilepath, _Problem, _Index, [], _CommandsPairs, _OutputFilename, _CsvFilename, []).
test_mutants(DF, problem(N, D, R, OD, _, G, C, MS, LS), Index, [node(State, Cost)|T1], CP, OF, CsvFilename, [ExecutionTimes|T2]) :-
    NewProblem = problem(N, D, R, OD, State, G, C, MS, LS),
    atom_concat(Tmp, '.txt', OF),
    atom_concat(Tmp, '.pddl', NewProblemFilePath),
    serialise_problem(NewProblem, NewProblemFilePath),
    (
        foreach(command(PC, S, H, SRC), CP),
        fromto([], In, Out, Results),
        foreach(ET, ExecutionTimes),
        param(DF, NewProblemFilePath, OF, Cost, Index)
    do
        run_planner_command(PC, DF, NewProblemFilePath, OF, _, ET),
        deserialise_plan(OF, Result),
        length(Result, L),
        (   L = 0 -> (Failure = 0, Error = 1)
        ;   SRC > Cost + L -> (Failure = 1, Error = 0)
        ;
            (Failure = 0, Error = 0)
        ),
        Out = [result(Index, Cost, S, H, Failure, Error, L, SRC, ET)|In]
    ),
    write_nodes_csv_results(CsvFilename, Results),
    NewIndex is Index + 1,
    test_mutants(DF, problem(N, D, R, OD, _, G, C, MS, LS), NewIndex, T1, CP, OF, CsvFilename, T2).

make_commands(DomainFilepath, ProblemFilepath, OutputFilename, PlannersCommands, Commands) :-
    (
        foreach(PlannerCommand, PlannersCommands),
        foreach(command(PlannerCommand, Search, Heuristic, ResultCost), Commands),
        param(DomainFilepath, ProblemFilepath, OutputFilename)
    do
        run_planner_command(PlannerCommand, DomainFilepath, ProblemFilepath, OutputFilename, _ExitStatus, _ExecutionTime),
        deserialise_plan(OutputFilename, Result),
        validate_plan(Result),
        length(Result, ResultCost),
        command_parameters(PlannerCommand, Search, Heuristic)
    ).

command_parameters(PlannerCommand, Search, HeuristicName) :-
    atom_chars(PlannerCommand, Chars),
    read_until_dash(Chars, TmpChars),
    append(SearchChars, [' '|HeuristicChars], TmpChars),
    atom_chars(Search, SearchChars),
    atom_chars(Heuristic, HeuristicChars),
    heuristic_name(Heuristic, HeuristicName).

read_until_dash([], []).
read_until_dash(['-'|Tail], Tail) :-
    !.
read_until_dash([_|T1], T2) :-
    read_until_dash(T1, T2).

heuristic_name(h_max, hmax).
heuristic_name(h_diff, hdiff).
heuristic_name(h_state_length, hlength).
heuristic_name(h_distance_with_i, hi).
heuristic_name(h_distance_with_g, hg).
heuristic_name(h_nb_actions, hnba).