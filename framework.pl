:- use_module(library(process), [process_create/3]).
:- use_module(library(lists), [prefix_length/3, maplist/3]).
:- use_module(library(samsort), [samsort/3]).
:- use_module(library(system), [now/1]).
:- use_module(library(random), [setrand/1]).

:- ensure_loaded(generators).
:- ensure_loaded(heuristics).
:- ensure_loaded(pddl_parser).
:- ensure_loaded(pddl_serialiser).
:- ensure_loaded(problem).
:- ensure_loaded(blackboard_data).
:- ensure_loaded(configuration).
:- ensure_loaded(results).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SYSTEM COMMANDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% /!\ NO SAFE PREDICATE
remove_files([]).
remove_files([Filename|T]) :-
    process_create(path(powershell), ['-Command', 'rm', Filename], [wait(_ExitStatus)]),
    remove_files(T).

remove_tmp_files :-
    process_create(path(powershell), ['-Command', 'rm', 'tmp/*.txt'], [wait(_ExitStatus)]),
    process_create(path(powershell), ['-Command', 'rm', 'tmp/*.pddl'], [wait(_ExitStatus)]).

%% run_planner_command(+Command, +DomainFilename, +ProblemFilename, +ResultFilename, -ExitStatus, -ExecutionTime).
run_planner_command(Command, DomainFilename, ProblemFilename, ResultFilename, ExitStatus, ExecutionTime) :-
    statistics(walltime, [StartTime, _]),
    process_create(path(powershell), ['-Command', Command, DomainFilename, ProblemFilename, ResultFilename], [wait(ExitStatus)]),
    statistics(walltime, [CurrentTime, _]),
    ExecutionTime is (CurrentTime - StartTime) / 1000.

%% run_planner_commands(+PlannerCommand, +DomainFilename, +ProblemsFilenames, -ResultsFilenames, -ExecutionTimes).
run_planner_commands(_PlannerCommand, _DomainFilename, [], [], []).
run_planner_commands(PlannerCommand, DomainFilename, [ProblemFilename|T1], [ResultFilename|T2], [ExecutionTime|T3]) :-
    atom_concat(ProblemName, '.pddl', ProblemFilename),
    atom_concat(ProblemName, '.txt', ResultFilename),
    run_planner_command(PlannerCommand, DomainFilename, ProblemFilename, ResultFilename, _ExitStatus, ExecutionTime),
    run_planner_commands(PlannerCommand, DomainFilename, T1, T2, T3).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NODE PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% node(+State, +Cost, +Heuristic).
node(_State, _Cost, _Heuristic).

%% compare_generated_node(+Node1-_Generator, +Node2-_Generator).
compare_generated_node(node(_, _, Heuristic1)-_, node(_, _, Heuristic2)-_) :-
    Heuristic1 >= Heuristic2.

generate_nodes_from_generators(Generators, Results) :-
    % retrieves in a list every nodes set (for each generator)
    (
        foreach(Generator, Generators),
        fromto([], In, Out, List)
    do
        Generator =.. [GeneratorName|GeneratorArguments],
        % appends at the end of the arguments list a variable to unify the result
        append(GeneratorArguments, [GeneratedNodes], Arguments),
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

rank_nodes(Heuristic, GeneratedNodes, RankedNodes) :-
    (
        foreach(Node-Generator, GeneratedNodes),
        foreach(RankedNode-Generator, RankedNodes),
        param(Heuristic)
    do
        Heuristic =.. [HeuristicName|HeuristicArguments],
        % adds at the end of the heuristic predicate both the node and the ranked node
        append(HeuristicArguments, [Node, RankedNode], Arguments),
        HeuristicPredicate =.. [HeuristicName|Arguments],
        HeuristicPredicate
    ).

sort_nodes(RankedNodes, SortedNodes) :-
    % uses node-related ordering predicates to sort the nodes
    samsort(compare_generated_node, RankedNodes, SortedNodes).

generate_nodes(Nodes) :-
    get_configuration(Configuration),
    configuration_generators(Configuration, Generators),
    configuration_heuristic(Configuration, Heuristic),
    configuration_nb_tests(Configuration, NumberOfTests),
    % generates a set of pairs Node-GeneratorPredicate
    generate_nodes_from_generators(Generators, GeneratedNodes),
    % ranks and sorts the set
    rank_nodes(Heuristic, GeneratedNodes, RankedNodes),
    sort_nodes(RankedNodes, SortedNodes),
    % limits the number of tests
    length(SortedNodes, NumberOfNodes),
    (NumberOfTests > NumberOfNodes -> N = NumberOfNodes ; N = NumberOfTests),
    % selects the first ones
    prefix_length(SortedNodes, Nodes, N).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INPUTS TESTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% make_follow_up_inputs(+MetamorphicRelation, -FollowUpInputs).
make_follow_up_inputs(mr0, FollowUpInputs) :-
    generate_nodes(GeneratedNodes),
    get_problem(Problem),
    (
        foreach(Node-Generator, GeneratedNodes),
        foreach(FollowUpInput, FollowUpInputs),
        param(Problem)
    do
        Problem = problem(Name, D, R, OD, _, G, C, MS, LS),
        Node = node(State, _Cost, _Heuristic),
        FollowUpProblem = problem(Name, D, R, OD, State, G, C, MS, LS),
        FollowUpInput = input(FollowUpProblem, Node, Generator)
    ).
make_follow_up_inputs(mr1, FollowUpInputs) :-
    generate_nodes(GeneratedNodes),
    get_problem(Problem),
    (
        foreach(Node-Generator, GeneratedNodes),
        foreach(FollowUpInput, FollowUpInputs),
        param(Problem)
    do
        Problem = problem(Name, D, R, OD, IS, _, C, MS, LS),
        Node = node(State, _Cost, _Heuristic),
        FollowUpProblem = problem(Name, D, R, OD, IS, State, C, MS, LS),
        FollowUpInput = input(FollowUpProblem, Node, Generator)
    ).

%% serialise_follow_up_inputs(+Inputs, -InputsFilenames, +Index).
serialise_follow_up_inputs([], [], _Index).
serialise_follow_up_inputs([Input|T1], [Filename|T2], Index) :-
    serialise_follow_up_input(Input, Filename, Index),
    NewIndex is Index + 1,
    serialise_follow_up_inputs(T1, T2, NewIndex).

serialise_follow_up_input(Input, Filename, Index) :-
    Input = input(Problem, _Node, _Generator),
    problem_name(Problem, Name),
    number_codes(Index, Codes),
    atom_codes(IndexAtom, Codes),
    atom_concat('tmp/', Name, NameInTmpFolder),
    atom_concat(NameInTmpFolder, IndexAtom, NewProblemName),
    atom_concat(NewProblemName, '.pddl', Filename),
    serialise_problem(Problem, Filename).

%% make_input(+DomainFilename, +ProblemFilename, -Input).
make_input(DomainFilename, ProblemFilename, Domain-Problem) :-
    parse_domain(DomainFilename, Domain),
    parse_problem(ProblemFilename, TmpProblem),
    sort_problem(TmpProblem, Problem).

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
        Node = node(_State, Cost, _Heuristic),
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
    now(Time),
    setrand(Time),
    format('setrand with ~d\n', [Time]),
    start.

start :-
    prolog_flag(argv, [ConfigurationFilename]),
    main_read_configuration(ConfigurationFilename),
    halt.

start :-
    prolog_flag(argv, Arguments),
    length(Arguments, 9),
    main_with_configuration(Arguments),
    halt.

main_read_configuration(ConfigurationFilename) :-
    %%%%%%%%%%%%%%% manages the configuration reading %%%%%%%%%%%%%%%%%%
    deserialise_configuration(ConfigurationFilename, Configuration),
    configuration_domain_filename(Configuration, DomainFilename),
    configuration_problem_filename(Configuration, ProblemFilename),
    configuration_planner_command(Configuration, PlannerCommand),
    configuration_metamorphic_relation(Configuration, MetamorphicRelation),
    configuration_result_filename(Configuration, CsvFilename),
    configuration_run_all_tests(Configuration, RAT),
    %%%%%%%%%%%%%%% makes the domain and problem models %%%%%%%%%%%%%%%%
    make_input(DomainFilename, ProblemFilename, Domain-Problem),
    %%%%%%%%%%%%%%% initilises the blackboard %%%%%%%%%%%%%%%%%%%%%%%%%%
    set_blackboard(Configuration, Domain, Problem),
    %%%%%%%%%%%%%%% launches metamorphic testing %%%%%%%%%%%%%%%%%%%%%%%
    (   RAT -> test_planner_all(MetamorphicRelation, PlannerCommand, DomainFilename, ProblemFilename, 'tmp/output.txt', [SourceResultCost|TestResults])
    ;   test_planner_until_failure(MetamorphicRelation, PlannerCommand, DomainFilename, ProblemFilename, 'tmp/output.txt', [SourceResultCost|TestResults])
    ),
    write('metamorphic testing done\n'),
    %%%%%%%%%%%%%%% exports the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    set_csv_result(Domain, Problem, Configuration, SourceResultCost),
    write_csv_results(CsvFilename, TestResults),
    format('results exported (~a)\n', [CsvFilename]).

main_with_configuration([DomainFilename, ProblemFilename, PlannerCommand, MetamorphicRelation, NBTestsAtom, RunAllTests, CsvFilename, GeneratorPredicate, HeuristicPredicate]) :-
    % turns NBTestsAtom into an integer
    atom_codes(NBTestsAtom, Codes),
    number_codes(NBTests, Codes),
    Configuration = configuration(PlannerCommand, DomainFilename, ProblemFilename, CsvFilename, MetamorphicRelation, NBTests, RunAllTests, [GeneratorPredicate], HeuristicPredicate),
    %%%%%%%%%%%%%%% makes the domain and problem models %%%%%%%%%%%%%%%%
    make_input(DomainFilename, ProblemFilename, Domain-Problem),
    %%%%%%%%%%%%%%% initilises the blackboard %%%%%%%%%%%%%%%%%%%%%%%%%%
    set_blackboard(Configuration, Domain, Problem),
    %%%%%%%%%%%%%%% launches metamorphic testing %%%%%%%%%%%%%%%%%%%%%%%
    (   RunAllTests -> test_planner_all(MetamorphicRelation, PlannerCommand, DomainFilename, ProblemFilename, 'tmp/output.txt', [SourceResultCost|TestResults])
    ;   test_planner_until_failure(MetamorphicRelation, PlannerCommand, DomainFilename, ProblemFilename, 'tmp/output.txt', [SourceResultCost|TestResults])
    ),
    write('metamorphic testing done\n'),
    %%%%%%%%%%%%%%% exports the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    set_csv_result(Domain, Problem, Configuration, SourceResultCost),
    write_csv_results(CsvFilename, TestResults),
    format('results exported (~a)\n', [CsvFilename]).

%% test_planner_all(+MetamorphicRelation, +PlannerCommand, +DomainFilename, +ProblemFilename, +SourceResultFilename, -TestResults).
test_planner_all(MetamorphicRelation, PlannerCommand, DomainFilename, ProblemFilename, SourceResultFilename, [SourceResultCost|TestResults]) :-
    run_planner_command(PlannerCommand, DomainFilename, ProblemFilename, SourceResultFilename, _ExitStatus, _ExecutionTime),
    write('initial problem run\n'),
    deserialise_plan(SourceResultFilename, SourceResult),
    write('source result deserialised'),
    SourceResult \== [],
    !,
    length(SourceResult, SourceResultCost),
    format(' (of cost ~d)\n', [SourceResultCost]),
    set_source_result(SourceResult),
    write('blackboard updated with the source result\n'),
    % makes the follow-up inputs wrt the metamorphic relation
    make_follow_up_inputs(MetamorphicRelation, FollowUpInputs),
    length(FollowUpInputs, L),
    format('follow-up inputs made (~d)\n', [L]),
    % serialises the follow-up inputs
    serialise_follow_up_inputs(FollowUpInputs, FollowUpInputsFilenames, 0),
    write('follow-up inputs serialised\n'),
    % runs the planner on every follow-up input
    run_planner_commands(PlannerCommand, DomainFilename, FollowUpInputsFilenames, FollowUpResultsFilenames, ExecutionTimes),
    write('follow-up problems run\n'),
    % deserialises all the follow-up results
    deserialise_plans(FollowUpResultsFilenames, FollowUpResults),
    write('follow-up results deserialised\n'),
    % checks the metamorphic relation for every follow-up test case result
    check_metamorphic_relation(SourceResult, FollowUpInputs, FollowUpResults, Results),
    % completes the results by adding the previously retrieved execution times (when running planner)
    set_execution_times(Results, ExecutionTimes, TestResults),
    % removes all the temporary generated files
    remove_tmp_files.
    % append([SourceResultFilename|FollowUpInputsFilenames], FollowUpResultsFilenames, FilenamesToRemove),
    % remove_files(FilenamesToRemove).
test_planner_all(_, _, _, _, _, [0]) :-
    write('\nempty source result\n').

%% set_execution_times(+Results, +ExecutionTimes, -TestResults).
set_execution_times([], [], []).
set_execution_times([result(N, G, F, E, _, RC)|T1], [ExecutionTime|T2], [result(N, G, F, E, ExecutionTime, RC)|T3]) :-
    set_execution_times(T1, T2, T3).

%% test_planner_until_failure(+MetamorphicRelation, +PlannerCommand, +DomainFilename, +ProblemFilename, +SourceResultFilename, -TestResults).
test_planner_until_failure(MetamorphicRelation, PlannerCommand, DomainFilename, ProblemFilename, SourceResultFilename, [SourceResultCost|TestResults]) :-
    run_planner_command(PlannerCommand, DomainFilename, ProblemFilename, SourceResultFilename, _ExitStatus, _ExecutionTime),
    write('initial problem run\n'),
    deserialise_plan(SourceResultFilename, SourceResult),
    write('source result deserialised'),
    SourceResult \== [],
    !,
    length(SourceResult, SourceResultCost),
    format(' (of cost ~d)\n', [SourceResultCost]),
    set_source_result(SourceResult),
    write('blackboard updated with the source result\n'),
    % makes the follow-up inputs wrt the metamorphic relation
    make_follow_up_inputs(MetamorphicRelation, FollowUpInputs),
    length(FollowUpInputs, L),
    format('follow-up inputs made (~d)\n', [L]),
    write('testing until the metamorphic relation is violated...\n'),
    % tests each input until failure / violation is detected
    test_inputs_one_by_one(FollowUpInputs, 0, SourceResult, PlannerCommand, DomainFilename, TestResults),
    % removes all the temporary generated files
    remove_tmp_files.
test_planner_until_failure(_, _, _, _, _, [0]) :-
    write('\nempty source result\n').

%% test_inputs_one_by_one(+Inputs, +Index, +SourceResult, +PlannerCommand, +DomainFilename, -Results).
test_inputs_one_by_one([], _Index, _SourceResult, _PlannerCommand, _DomainFilename, []).
test_inputs_one_by_one([Input|T1], Index, SourceResult, PlannerCommand, DomainFilename, [Result|T2]) :-
    % serialises the follow-up input
    serialise_follow_up_input(Input, InputFilename, Index),
    atom_concat(ProblemName, '.pddl', InputFilename),
    atom_concat(ProblemName, '.txt', ResultFilename),
    % runs the planner on the serialised follow-up input
    run_planner_command(PlannerCommand, DomainFilename, InputFilename, ResultFilename, _ExitStatus, ExecutionTime),
    % deserialises the follow-up result
    deserialise_plan(ResultFilename, InputResult),
    % checks the metamorphic relation for the follow-up test case result
    length(InputResult, InputResultCost),
    check_metamorphic_relation(SourceResult, [Input], [InputResult], [result(Node, Generator, Failure, Error, _, InputResultCost)]),
    Result = result(Node, Generator, Failure, Error, ExecutionTime, InputResultCost),
    NewIndex is Index + 1,
    % stops testing (if failure) by resuming with an empty inputs list
    (   Failure == 1 -> test_inputs_one_by_one([], NewIndex, SourceResult, PlannerCommand, DomainFilename, T2)
    ;   test_inputs_one_by_one(T1, NewIndex, SourceResult, PlannerCommand, DomainFilename, T2)
    ).