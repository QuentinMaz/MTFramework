:- use_module(library(process), [process_create/3]).
:- use_module(library(lists), [nth1/3, keys_and_values/3, prefix_length/3, suffix_length/3, maplist/3, reverse/2]).
:- use_module(library(system), [now/1]).
:- use_module(library(random), [setrand/1, random_permutation/2]).

:- ensure_loaded(generators).
:- ensure_loaded(heuristics).
:- ensure_loaded(pddl_parser).
:- ensure_loaded(pddl_serialiser).
:- ensure_loaded(problem).
:- ensure_loaded(blackboard_data).
:- ensure_loaded(configuration).
:- ensure_loaded(results).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% UTILS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

write_integer_list([LastInteger]) :-
    format('~d', [LastInteger]).
write_integer_list([Integer|Tail]) :-
    format('~d ', [Integer]),
    write_integer_list(Tail).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SYSTEM COMMANDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% /!\ NO SAFE PREDICATE
remove_files([]).
remove_files([Filename|T]) :-
    atom_concat('rm ', Filename, Command),
    process_create(path(sh), ['-c', Command], [wait(_ExitStatus)]),
    remove_files(T).

remove_tmp_files :-
    process_create(path(sh), ['-c', 'rm tmp/*.txt'], [wait(_ExitStatus)]),
    process_create(path(sh), ['-c', 'rm tmp/*.pddl'], [wait(_ExitStatus)]).

%% run_planner_command(+Command, +DomainFilename, +ProblemFilename, +ResultFilename, -ExitStatus, -ExecutionTime).
run_planner_command(Command, DomainFilename, ProblemFilename, ResultFilename, ExitStatus, ExecutionTime) :-
    statistics(walltime, [StartTime, _]),    
    atom_concat(Command, ' ', Tmp1),
    atom_concat(Tmp1, DomainFilename, Tmp2),
    atom_concat(Tmp2, ' ', Tmp3),
    atom_concat(Tmp3, ProblemFilename, Tmp4),
    atom_concat(Tmp4, ' ', Tmp5),
    atom_concat(Tmp5, ResultFilename, Tmp6),
    process_create(path(sh), ['-c', Tmp6], [wait(ExitStatus)]),
    statistics(walltime, [CurrentTime, _]),
    ExecutionTime is (CurrentTime - StartTime) / 1000.

%% run_problems(+PlannerCommand, +DomainFilename, +ProblemsFilenames, -ResultsFilenames, -ExecutionTimes).
% runs a single planner on multiple problems
run_planner_commands(_PlannerCommand, _DomainFilename, [], [], []).
run_planner_commands(PlannerCommand, DomainFilename, [ProblemFilename|T1], [ResultFilename|T2], [ExecutionTime|T3]) :-
    atom_concat(ProblemName, '.pddl', ProblemFilename),
    atom_concat(ProblemName, '.txt', ResultFilename),
    run_planner_command(PlannerCommand, DomainFilename, ProblemFilename, ResultFilename, _ExitStatus, ExecutionTime),
    run_planner_commands(PlannerCommand, DomainFilename, T1, T2, T3).

run_problem(PlannersCommands, DomainFilename, ProblemFilename, Results) :-
    atom_concat(ProblemPath, '.pddl', ProblemFilename),
    atom_chars(ProblemPath, Chars),
    reverse(Chars, ReversedChars),
    append(ReversedProblemChars, ['/'|_], ReversedChars),
    reverse(ReversedProblemChars, ProblemChars),
    atom_chars(ProblemName, ProblemChars),
    atom_concat('tmp/', ProblemName, ProblemNameInTmpFolder),
    atom_concat(ProblemNameInTmpFolder, '.txt', ResultFilename),
    (
        foreach(PlannerCommand, PlannersCommands),
        foreach(Result, Results),
        param(DomainFilename, ProblemFilename, ResultFilename)
    do
        run_planner_command(PlannerCommand, DomainFilename, ProblemFilename, ResultFilename, _ExitStatus, _ExecutionTime),
        deserialise_plan(ResultFilename, Result),
        (validate_plan(Result) -> true ; Result = [])
    ).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NODE PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% node(+State, +Cost, +Heuristics).
node(_State, _Cost, _Heuristics).
node(_State, _Cost, _HeuristicName, _HeuristicValue).

generate_nodes_from_generator(Generator, GeneratedNodes) :-
    % call the generator predicate
    Generator =.. [GeneratorName|GeneratorArguments],
    % appends at the end of the arguments list a variable to unify the result
    append(GeneratorArguments, [GeneratedNodes], Arguments),
    GeneratorPredicate =.. [GeneratorName|Arguments],
    GeneratorPredicate,
    write(Generator), write(' : '), length(GeneratedNodes, GNL), write(GNL), nl.

compute_nodes_heuristics(Heuristics, GeneratedNodes, RankedNodes) :-
    (
        foreach(Node, GeneratedNodes),
        foreach(RankedNode, RankedNodes),
        param(Heuristics)
    do
        compute_heuristic_values(Node, Heuristics, Values),
        Node = node(State, Cost, _),
        RankedNode = node(State, Cost, Values)
    ).

compute_heuristic_values(_Node, [], []).
compute_heuristic_values(Node, [Heuristic|T1], [HeuristicValue|T2]) :-
    Heuristic =.. [HeuristicName|HeuristicArguments],
    % adds at the end of the heuristic predicate both the node and the ranked node
    append(HeuristicArguments, [Node, HeuristicValue], Arguments),
    HeuristicPredicate =.. [HeuristicName|Arguments],
    HeuristicPredicate, 
    compute_heuristic_values(Node, T1, T2).

generate_nodes(Nodes) :-
    get_configuration(Configuration),
    configuration_generator(Configuration, Generator),
    configuration_heuristics(Configuration, Heuristics),
    % generates the nodes
    generate_nodes_from_generator(Generator, GeneratedNodes),
    % computes all the heuristics available for each node
    compute_nodes_heuristics(Heuristics, GeneratedNodes, Nodes).

select_nodes(_Nodes, [], []).
select_nodes(Nodes, [[]|T1], [[]|T2]) :-
    !,
    write('empty source detected. Nodes selection aborted.\n'),
    select_nodes(Nodes, T1, T2).
select_nodes(Nodes, [SourceResult|T1], [SelectedNodes|T2]) :-
    compute_bound(SourceResult, CostBound),
    filter_nodes(Nodes, CostBound, FilteredNodes),
    length(FilteredNodes, FDL),
    format('~d nodes available after filtration with cost bound of ~d\n', [FDL, CostBound]),
    process_nodes(FilteredNodes, SelectedNodes),
    select_nodes(Nodes, T1, T2).

compute_bound(Plan, CostBound) :-
    length(Plan, PlanLength),
    CostBound is PlanLength // 2 + 1.

filter_nodes([], _, []).
filter_nodes([nodes(_, Cost, _)|T1], CostBound, Result) :-
    Cost > CostBound,
    !,
    filter_nodes(T1, Result).
filter_nodes([Node|T1], CostBound, [Node|T2]) :-
    filter_nodes(T1, CostBound, T2).

%% process_nodes(+Nodes, -ProcessedNodes).
process_nodes(Nodes, Results) :-
    get_configuration(Configuration),
    configuration_heuristics(Configuration, Heuristics),
    configuration_nb_tests(Configuration, Nb_Tests),
    length(Nodes, Nb_Nodes),
    (Nb_Nodes < Nb_Tests -> N is Nb_Nodes ; N is Nb_Tests),
    (
        foreach(Heuristic, Heuristics),
        fromto([], In, Out, Results),
        param(Nodes, N, Heuristics)
    do
        nth1(I, Heuristics, Heuristic),
        (
            foreach(node(S, C, H), Nodes),
            foreach(V-node(S, C, V), Indexed_Nodes),
            param(I)
        do
            nth1(I, H, V)
        ),
        sort(Indexed_Nodes, Sorted_Indexed_Nodes),
        keys_and_values(Sorted_Indexed_Nodes, _, SortedNodes),
        (
            Heuristic == h_random -> 
            (
                random_permutation(SortedNodes, Tmp1), 
                prefix_length(Tmp1, Tmp2, N),
                nodes3_to_nodes4(Tmp2, Heuristic, '', SelectedNodes)
            )
            ;
            (
                prefix_length(SortedNodes, Prefix, N),
                nodes3_to_nodes4(Prefix, Heuristic, min_, MinNodes),
                suffix_length(SortedNodes, Suffix, N),
                nodes3_to_nodes4(Suffix, Heuristic, max_, MaxNodes),
                append(MinNodes, MaxNodes, SelectedNodes)
            )
        ),
        append(In, SelectedNodes, Out)
    ),
    length(Results, Nb_Results),
    format('~d nodes selected\n', [Nb_Results]).

nodes3_to_nodes4([], _HeuristicName, _NamePrefix, []).
nodes3_to_nodes4([node(S, C, HV)|T1], HeuristicName, NamePrefix, [node(S, C, HN, HV)|T2]) :-
    atom_concat(NamePrefix, HeuristicName, HN),
    !,
    nodes3_to_nodes4(T1, HeuristicName, NamePrefix, T2).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INPUTS TESTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% make_follow_up_inputs(+MetamorphicRelation, +Nodes, -FollowUpInputs).
make_follow_up_inputs(mr0, Nodes, FollowUpInputs) :-
    get_problem(Problem),
    (
        foreach(Node, Nodes),
        foreach(FollowUpInput, FollowUpInputs),
        param(Problem)
    do
        Problem = problem(Name, D, R, OD, _, G, C, MS, LS),
        Node = node(State, _Cost, _HeuristicName, _HeuristicValue),
        FollowUpProblem = problem(Name, D, R, OD, State, G, C, MS, LS),
        FollowUpInput = input(FollowUpProblem, Node)
    ).

%% serialise_follow_up_inputs(+Inputs, -InputsFilenames, +Index).
serialise_follow_up_inputs([], [], _Index).
serialise_follow_up_inputs([Input|T1], [Filename|T2], Index) :-
    serialise_follow_up_input(Input, Filename, Index),
    NewIndex is Index + 1,
    serialise_follow_up_inputs(T1, T2, NewIndex).

serialise_follow_up_input(Input, Filename, Index) :-
    Input = input(Problem, _Node),
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
        FollowUpInput = input(_Problem, Node),
        Node = node(_State, Cost, _HeuristicName, _HeuristicValue),
        length(FollowUpResult, FollowUpResultLength),
        (   FollowUpResultLength = 0 -> (Failure = 0, Error = 1)
        ;   SourceResultLength > Cost + FollowUpResultLength -> (Failure = 1, Error = 0)
        ;
            (Failure = 0, Error = 0)
        ),
        Result = result(Node, Failure, Error, _ExecutionTime, FollowUpResultLength)
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
    prolog_flag(argv, Arguments),
    (
        Arguments = [ConfigurationFilename] -> deserialise_configuration(ConfigurationFilename, Configuration)
    ;   (Configuration = configuration(_, _, _, _, _, _, _, _, _), args_to_configuration(Arguments, Configuration))
    ),
    ground(Configuration),
    main(Configuration),
    halt.

start :-
    write('something went wrong while initialising the configuration.\n'),
    halt.

main(Configuration) :-
    %%%%%%%%%%%%%%% accesses the configuration %%%%%%%%%%%%%%%%
    configuration_domain_filename(Configuration, DomainFilename),
    configuration_problem_filename(Configuration, ProblemFilename),
    configuration_planners_commands(Configuration, PlannersCommands),
    configuration_metamorphic_relation(Configuration, MetamorphicRelation),
    configuration_results_filenames(Configuration, CsvFilenames),
    configuration_run_all_tests(Configuration, RAT),
    %%%%%%%%%%%%%%% makes the domain and problem models %%%%%%%%%%%%%%%%
    make_input(DomainFilename, ProblemFilename, Domain-Problem),
    %%%%%%%%%%%%%%% initilises the blackboard %%%%%%%%%%%%%%%%%%%%%%%%%%
    set_blackboard(Configuration, Domain, Problem),
    %%%%%%%%%%%%%%% launches metamorphic testing %%%%%%%%%%%%%%%%%%%%%%%
    (   RAT -> test_planner_all(MetamorphicRelation, PlannersCommands, DomainFilename, ProblemFilename, TestsResults)
    ;   test_planner_until_failure(MetamorphicRelation, PlannersCommands, DomainFilename, ProblemFilename, 'tmp/output.txt', TestsResults)
    ),
    write('metamorphic testing done\n'),
    %%%%%%%%%%%%%%% exports the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    (
        foreach([SourceResultCost|TestResults], TestsResults),
        foreach(CsvFilename, CsvFilenames),
        param(Domain, Problem, Configuration)
    do
        set_csv_result(Domain, Problem, Configuration, CsvFilename, SourceResultCost),
        write_csv_results(CsvFilename, TestResults),
        format('results exported (~a)\n', [CsvFilename])
    ),
    write('all results exported.\n').

args_to_configuration([], _).
args_to_configuration(Arguments, Configuration) :-
    get_argument(Arguments, Configuration, Tail),
    args_to_configuration(Tail, Configuration).

get_argument(['-d', DomainFilename|T], configuration(_, DomainFilename, _, _, _, _, _, _, _), T).
get_argument(['-p', ProblemFilename|T], configuration(_, _, ProblemFilename, _, _, _, _, _, _), T).
get_argument(['-m', MetamorphicRelation|T], configuration(_, _, _, _, MetamorphicRelation, _, _, _, _), T).
get_argument(['-g', GeneratorPredicate|T], configuration(_, _, _, _, _, _, _, GeneratorPredicate, _), T).
get_argument(['-r', RunAllTests|T], configuration(_, _, _, _, _, _, RunAllTests, _, _), T).

get_argument(['-n', NBTestsAtom|T], configuration(_, _, _, _, _, NBTests, _, _, _), T) :-
    atom_codes(NBTestsAtom, Codes),
    number_codes(NBTests, Codes).

get_argument(['-o'|Arguments], configuration(_, _, _, ResultsFilenames, _, _, _, _, _), [NextArgumentFlag|T]) :-
    append(ResultsFilenames, [NextArgumentFlag|T], Arguments),
    atom_codes(NextArgumentFlag, [45, _]),
    !.
get_argument(['-o'|ResultsFilenames], configuration(_, _, _, ResultsFilenames, _, _, _, _, _), []).

get_argument(['-h'|Arguments], configuration(_, _, _, _, _, _, _, _, HeuristicsPredicates), [NextArgumentFlag|T]) :-
    append(HeuristicsPredicates, [NextArgumentFlag|T], Arguments),
    atom_codes(NextArgumentFlag, [45, _]),
    !.
get_argument(['-h'|HeuristicsPredicates], configuration(_, _, _, _, _, _, _, _, HeuristicsPredicates), []).

get_argument(['-c'|Arguments], configuration(PlannersCommands, _, _, _, _, _, _, _, _), [NextArgumentFlag|T]) :-
    append(PlannersCommands, [NextArgumentFlag|T], Arguments),
    atom_codes(NextArgumentFlag, [45, _]),
    !.
get_argument(['-c'|PlannersCommands], configuration(PlannersCommands, _, _, _, _, _, _, _, _), []).

%% test_planner_all(+MetamorphicRelation, +PlannersCommands, +DomainFilename, +ProblemFilename, -TestsResults).
test_planner_all(MetamorphicRelation, PlannersCommands, DomainFilename, ProblemFilename, TestsResults) :-
    % runs the planners on the source problem
    run_problem(PlannersCommands, DomainFilename, ProblemFilename, SourceResults),
    write('initial problem run on all planners'),
    maplist(length, SourceResults, SourceResultsCosts),
    format(' (of cost ~@)\n', [write_integer_list(SourceResultsCosts)]),
    set_source_result(SourceResults),
    write('blackboard updated with the source results\n'),
    % generates all the nodes and ranks them all
    generate_nodes(AllGeneratedNodes),
    % selects for each planner a selection of nodes (wrt their source result)
    select_nodes(AllGeneratedNodes, SourceResults, SelectedNodes),
    % loops over the MT procedure for each planner (command)
    (
        foreach(PlannerCommand, PlannersCommands),
        foreach(SourceResult, SourceResults),
        foreach(Nodes, SelectedNodes),
        foreach(SourceResultCost, SourceResultsCosts),
        foreach([SourceResultCost|TestResults], TestsResults),
        param(MetamorphicRelation, DomainFilename)  
    do
        (SourceResultCost == 0 -> TestResults = [] ; (
        % makes the follow-up inputs wrt the metamorphic relation
        make_follow_up_inputs(MetamorphicRelation, Nodes, FollowUpInputs),
        length(FollowUpInputs, L),
        format('follow-up inputs made (~d)\n', [L]),
        % serialises the follow-up inputs
        serialise_follow_up_inputs(FollowUpInputs, FollowUpInputsFilenames, 0),
        write('follow-up inputs serialised\n'),
        % runs the planners on every follow-up input
        run_planner_commands(PlannerCommand, DomainFilename, FollowUpInputsFilenames, FollowUpResultsFilenames, ExecutionTimes),
        write('follow-up problems run\n'),
        % deserialises all the follow-up results
        deserialise_plans(FollowUpResultsFilenames, FollowUpResults),
        % checks the metamorphic relation for every follow-up test case result
        check_metamorphic_relation(SourceResult, FollowUpInputs, FollowUpResults, Results),
        % completes the results by adding the previously retrieved execution times (when running planner)
        set_execution_times(Results, ExecutionTimes, TestResults),
        remove_tmp_files))
    ).
test_planner_all(_, _, _, _, _) :-
    write('\nsomething went wrong during metamorphic testing.\n').

%% set_execution_times(+Results, +ExecutionTimes, -TestResults).
set_execution_times([], [], []).
set_execution_times([result(N, F, E, _, RC)|T1], [ExecutionTime|T2], [result(N, F, E, ExecutionTime, RC)|T3]) :-
    set_execution_times(T1, T2, T3).

%%%%%%%%%%%%%%%%% TOREDO %%%%%%%%%%%%%%%%%%

%% test_planner_until_failure(+MetamorphicRelation, +PlannerCommand, +DomainFilename, +ProblemFilename, +SourceResultFilename, -TestResults).
test_planner_until_failure(MetamorphicRelation, PlannerCommand, DomainFilename, ProblemFilename, SourceResultFilename, [SourceResultCost|TestResults]) :-
    run_planner_command(PlannerCommand, DomainFilename, ProblemFilename, SourceResultFilename, _ExitStatus, _ExecutionTime),
    write('initial problem run\n'),
    deserialise_plan(SourceResultFilename, SourceResult),
    write('source result deserialised'),
    validate_plan(SourceResult),
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
    write('\nsource result not valid\n').

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