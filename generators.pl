:- module(generators,
    [
        bfs_generator/2, walks_generator/2,

        select_min_dist_i/2, select_max_dist_i/2,
        select_min_dist_g/2, select_max_dist_g/2,
        select_bfs/2,
        select_random/2, select_mutants_killers/2,
        select_bfs_indexes/3, select_random_indexes/3,

        validate_plan/1
    ]).

:- use_module(library(ordsets), [ord_subtract/3, ord_union/3, ord_union/2, ord_subset/2]).
:- use_module(library(queues), [queue_cons/3, list_queue/2, queue_append/3, queue_memberchk/2, empty_queue/1]).
:- use_module(library(sets), [is_set/1, list_to_set/2]).
:- use_module(library(lists), [reverse/2, nth0/3]).
:- use_module(library(random), [random_member/2, random_permutation/2, random/3]).

:- ensure_loaded(domain).
:- ensure_loaded(problem).
:- ensure_loaded(blackboard_data).
:- ensure_loaded(cache).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GENERATORS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% bfs_generator is one of the baseline generators. It explores the state space in a breath first search manner and returns the N first states discovered.
% if no heuristic given, then return the N first states visited
bfs_generator(N, RelevantResults) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    StartNode = node(InitialState, 0),
    list_queue([StartNode], Queue),
    bfs(Queue, N, [], VisitedNodes),
    % removes the start node (as it is the initial state...)
    reverse(VisitedNodes, [_|Results]),
    problem_goal_state(Problem, GoalState),
    filter_nodes(Results, GoalState, RelevantResults).

%% bfs(+Queue, +N, +VisitedNodes, -Results).
bfs(Queue, _N, VisitedNodes, VisitedNodes) :-
    empty_queue(Queue),
    write('empty queue (the entire state space has been visited).\n'),
    !.
bfs(_Queue, N, VisitedNodes, VisitedNodes) :-
    length(VisitedNodes, L),
    L > N,
    !.
bfs(Queue, N, VisitedNodes, Results) :-
    queue_cons(Node, RQ, Queue),
    (bagof(Child, expand_node(Node, Queue, VisitedNodes, Child), Children) ; Children = []),
    !,
    % duplicates may appear in children /!\
    list_to_set(Children, DistinctChildren),
    % appends to the queue the children
    queue_append(RQ, DistinctChildren, NewQueue),
    bfs(NewQueue, N, [Node|VisitedNodes], Results).

expand_node(node(State, Depth), Queue, VisitedNodes, node(NextState, NextDepth)) :-
    progress(State, NextState),
    % checks that we have never visited NextState nor have already planned to visit it
    \+ memberchk(node(NextState, _), VisitedNodes),
    \+ queue_memberchk(node(NextState, _), Queue),
    % child node instanciation
    NextDepth is Depth + 1.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RANDOM WALKS PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% callable random walks generator.
walks_generator(N, Results) :-
    get_source_result(SourceResult),
    length(SourceResult, MaximumWalkLength),
    get_cache_name(CacheName),
    load_nodes_or_generate(CacheName, Nodes),
    compute_reachable_nodes(N, MaximumWalkLength, Nodes, NewN),
    random_walks_generator(NewN, MaximumWalkLength, Results).

% naive generation strategy that returns N nodes reached by random walks in the state space.
random_walks_generator(N, MaximumWalkLength, Results) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    random_walks(InitialState, MaximumWalkLength, N, [], Results).

random_walks(_, _, 0, Results, Results).
random_walks(StartState, MaximumWalkLength, N, Accumulator, Results) :-
    random(1, MaximumWalkLength, WalkLength),
    % format('starts random walk of length ~d.\n', [WalkLength]),
    random_walk(StartState, WalkLength, NbActionsLeft, FinalState),
    \+ memberchk(node(FinalState, _), Accumulator),
    !, % since random_walk/4 unifies a single time (see the cut below), if the state returned has been already gathered, the current call fails: then a new random walk of different length is proceeded
    NewN is N - 1,
    Cost is WalkLength - NbActionsLeft,
    random_walks(StartState, MaximumWalkLength, NewN, [node(FinalState, Cost)|Accumulator], Results).
random_walks(StartState, MaximumWalkLength, N, Accumulator, Results) :-
    % write('repeats random walks.\n'),
    random_walks(StartState, MaximumWalkLength, N, Accumulator, Results). % /!\ repeat backtracking: infinite loop if no more states can be gathered (e.g, N is larger than the size of the state space)

random_walk(FinalState, 0, 0, FinalState) :-
    !. % see comment above on the cut in random_walks/5
random_walk(State, NbActions, NbActionsLeft, FinalState) :-
    bagof(NS, progress(State, NS), NextStates),
    random_member(NextState, NextStates),
    !,
    NewNbActions is NbActions - 1,
    random_walk(NextState, NewNbActions, NbActionsLeft, FinalState).
random_walk(LastState, NbActionsLeft, NbActionsLeft, LastState) :-
    format('no successor found. ~d actions left.\n', [NbActionsLeft]).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INDEXES SELECTION PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% mimics select_bfs/2 by returning the indexes of the nodes selected. Useful for a 'static' execution of the framework
select_bfs_indexes(N, MaximumCost, Indexes) :-
    get_cache_name(CacheName),
    load_nodes_or_generate(CacheName, Nodes),
    filter_nodes(Nodes, N, MaximumCost, SelectedNodes),
    (
        foreach(Node, SelectedNodes),
        foreach(I, Indexes),
        param(Nodes)
    do
        nth0(I, Nodes, Node)
    ).

%% mimics select_random/2 by returning the indexes of the nodes selected. Useful for a 'static' execution of the framework
select_random_indexes(N, MaximumCost, Indexes) :-
    get_cache_name(CacheName),
    load_nodes_or_generate(CacheName, Nodes),
    random_permutation(Nodes, ShuffledNodes),
    filter_nodes(ShuffledNodes, N, MaximumCost, SelectedNodes),
    (
        foreach(Node, SelectedNodes),
        foreach(I, Indexes),
        param(Nodes)
    do
        nth0(I, Nodes, Node)
    ).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SELECTION PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% selects the N first nodes whose costs are lower than the one of the source result.
select_bfs(N, Results) :-
    get_source_result(SourceResult),
    length(SourceResult, SourceResultCost),
    get_cache_name(CacheName),
    load_nodes_or_generate(CacheName, Nodes),
    filter_nodes(Nodes, N, SourceResultCost, Results).

% selects the nodes whose computed distance with the initial state is minimal.
select_min_dist_i(N, Results) :-
    get_source_result(SourceResult),
    length(SourceResult, SourceResultCost),
    get_cache_name(NodesCacheName),
    atom_concat(Tmp, '.txt', NodesCacheName),
    atom_concat(Tmp, '_min_dist_i.txt', CacheName),
    (load_nodes(CacheName, []) -> make_select_cache(NodesCacheName, CacheName, min_dist_i, Nodes) ; load_nodes(CacheName, Nodes)),
    filter_nodes(Nodes, N, SourceResultCost, Results).

% selects the nodes whose computed distance with the initial state is maximal.
select_max_dist_i(N, Results) :-
    get_source_result(SourceResult),
    length(SourceResult, SourceResultCost),
    get_cache_name(NodesCacheName),
    atom_concat(Tmp, '.txt', NodesCacheName),
    atom_concat(Tmp, '_max_dist_i.txt', CacheName),
    (load_nodes(CacheName, []) -> make_select_cache(NodesCacheName, CacheName, max_dist_i, Nodes) ; load_nodes(CacheName, Nodes)),
    filter_nodes(Nodes, N, SourceResultCost, Results).


% selects the nodes whose computed distance with the final state is minimal.
select_min_dist_g(N, Results) :-
    get_source_result(SourceResult),
    length(SourceResult, SourceResultCost),
    get_cache_name(NodesCacheName),
    atom_concat(Tmp, '.txt', NodesCacheName),
    atom_concat(Tmp, '_min_dist_g.txt', CacheName),
    (load_nodes(CacheName, []) -> make_select_cache(NodesCacheName, CacheName, min_dist_g, Nodes) ; load_nodes(CacheName, Nodes)),
    filter_nodes(Nodes, N, SourceResultCost, Results).


% selects the nodes whose computed distance with the final state is maximal.
select_max_dist_g(N, Results) :-
    get_source_result(SourceResult),
    length(SourceResult, SourceResultCost),
    get_cache_name(NodesCacheName),
    atom_concat(Tmp, '.txt', NodesCacheName),
    atom_concat(Tmp, '_max_dist_g.txt', CacheName),
    (load_nodes(CacheName, []) -> make_select_cache(NodesCacheName, CacheName, max_dist_g, Nodes) ; load_nodes(CacheName, Nodes)),
    filter_nodes(Nodes, N, SourceResultCost, Results).


% selects a random list of states.
select_random(N, Results) :-
    get_source_result(SourceResult),
    length(SourceResult, SourceResultCost),
    get_cache_name(CacheName),
    load_nodes_or_generate(CacheName, Nodes),
    random_permutation(Nodes, ShuffledNodes),
    filter_nodes(ShuffledNodes, N, SourceResultCost, Results).

% selects the nodes which successfully kill a list of mutated planners.
select_mutants_killers(N, Results) :-
    get_source_result(SourceResult),
    length(SourceResult, SourceResultCost),
    get_cache_name(NodesCacheName),
    atom_concat(Tmp, '.txt', NodesCacheName),
    atom_concat(Tmp, '_mt.txt', CacheName),
    load_nodes(CacheName, Nodes),
    Nodes \== [],
    !,
    filter_nodes(Nodes, N, SourceResultCost, Results).
select_mutants_killers(_, []) :-
    write('no mutants killers nodes found in cache. Nodes must be generated (by executing the framework with --cache argument) before.\n').

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPER PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% progress(+State, -NextState).
progress(State, NextState) :-
    % retrieves from the ground actions a possible operator
    get_operators(Operators),
    member(Operator, Operators),
    action_preconditions(Operator, TmpPreconditions),
    sort(TmpPreconditions, Preconditions),
    ord_subset(Preconditions, State),
    % applies the operator
    action_positive_effects(Operator, TmpPE),
    sort(TmpPE, PE),
    action_negative_effects(Operator, TmpNE),
    sort(TmpNE, NE),
    ord_subtract(State, NE, TmpState),
    ord_union(TmpState, PE, NextState).

%% progress(+State, +Operator, -NextState).
progress(State, Operator, NextState) :-
    generate_action(Action),
    untyped_action(Action, Operator),
    action_preconditions(Action, TmpPreconditions),
    sort(TmpPreconditions, Preconditions),
    ord_subset(Preconditions, State),
    % applies the operator
    action_positive_effects(Action, TmpPE),
    sort(TmpPE, PE),
    action_negative_effects(Action, TmpNE),
    sort(TmpNE, NE),
    ord_subtract(State, NE, TmpState),
    ord_union(TmpState, PE, NextState).

%% progress_from_state_with_plan(+StartState, +Plan, -Path).
progress_from_state_with_plan(FinalState, [], [FinalState]).
progress_from_state_with_plan(State, [ActionDef|T1], [State|T2]) :-
    progress(State, ActionDef, NextState),
    progress_from_state_with_plan(NextState, T1, T2).

validate_plan(Plan) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    problem_goal_state(Problem, GoalState),
    progress_from_state_with_plan(InitialState, Plan, SourceStates),
    append(_, [FinalState], SourceStates),
    !,
    ord_subset(GoalState, FinalState),
    set_final_state(FinalState).

filter_nodes([], _, _, []) :-
    !.
filter_nodes(_, 0, _, []) :-
    !.
filter_nodes([node(State, Cost)|T1], N, MaximumCost, [node(State, Cost)|T2]) :-
    Cost < MaximumCost,
    NewN is N - 1,
    !,
    filter_nodes(T1, NewN, MaximumCost, T2).
filter_nodes([_|T], N, MaximumCost, FilteredNodes) :-
    filter_nodes(T, N, MaximumCost, FilteredNodes).

filter_nodes([], _, []) :-
    !.
filter_nodes([node(State, Cost)|T1], StateToNotInclude, [node(State, Cost)|T2]) :-
    \+ ord_subset(StateToNotInclude, State),
    !,
    filter_nodes(T1, StateToNotInclude, T2).
filter_nodes([_|T], StateToNotInclude, FilteredNodes) :-
    filter_nodes(T, StateToNotInclude, FilteredNodes).


% computes the number of reachable states with respect to Maximum
compute_reachable_nodes(N, Maximum, Nodes, NewN) :-
    (
        foreach(node(_, NodeCost), Nodes),
        fromto(0, In, Out, NN),
        param(Maximum)
    do
        (NodeCost < Maximum -> Out is In + 1 ; Out is In)
    ),
    (NN < N -> NewN is NN ; NewN is N).