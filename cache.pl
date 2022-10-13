:- module(cache,
    [
        save_nodes/2, load_nodes/2, load_nodes_or_generate/2, make_select_cache/4, compute_distance_between_states/3
    ]).

:- use_module(library(ordsets), [ord_intersect/2, ord_intersection/3]).
:- use_module(library(lists), [keys_and_values/3, reverse/2]).

:- ensure_loaded(pddl_parser).
:- ensure_loaded(pddl_serialiser).
:- ensure_loaded(blackboard_data).
:- ensure_loaded(problem).
:- ensure_loaded(generators).

make_select_cache(OriginalCacheName, TargetCacheName, min_dist_i, AscendingOrderedNodes) :-
    load_nodes_or_generate(OriginalCacheName, Nodes),
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    (
        foreach(node(State, Cost), Nodes),
        foreach(H-node(State, Cost), H_Nodes),
        param(InitialState)
    do
        compute_distance_between_states(InitialState, State, H)
    ),
    sort(H_Nodes, Sorted_H_Nodes),
    keys_and_values(Sorted_H_Nodes, _, AscendingOrderedNodes),
    save_nodes(TargetCacheName, AscendingOrderedNodes),
    format('~a cache file successfully created.\n', [TargetCacheName]).
make_select_cache(OriginalCacheName, TargetCacheName, max_dist_i, DescendingOrderedNodes) :-
    load_nodes_or_generate(OriginalCacheName, Nodes),
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    (
        foreach(node(State, Cost), Nodes),
        foreach(H-node(State, Cost), H_Nodes),
        param(InitialState)
    do
        compute_distance_between_states(InitialState, State, H)
    ),
    sort(H_Nodes, Sorted_H_Nodes),
    keys_and_values(Sorted_H_Nodes, _, AscendingOrderedNodes),
    reverse(AscendingOrderedNodes, DescendingOrderedNodes),
    save_nodes(TargetCacheName, DescendingOrderedNodes),
    format('~a cache file successfully created.\n', [TargetCacheName]).
make_select_cache(OriginalCacheName, TargetCacheName, min_dist_g, AscendingOrderedNodes) :-
    load_nodes_or_generate(OriginalCacheName, Nodes),
    get_final_state(FinalState),
    (
        foreach(node(State, Cost), Nodes),
        foreach(H-node(State, Cost), H_Nodes),
        param(FinalState)
    do
        compute_distance_between_states(FinalState, State, H)
    ),
    sort(H_Nodes, Sorted_H_Nodes),
    keys_and_values(Sorted_H_Nodes, _, AscendingOrderedNodes),
    save_nodes(TargetCacheName, AscendingOrderedNodes),
    format('~a cache file successfully created.\n', [TargetCacheName]).
make_select_cache(OriginalCacheName, TargetCacheName, max_dist_g, DescendingOrderedNodes) :-
    load_nodes_or_generate(OriginalCacheName, Nodes),
    get_final_state(FinalState),
    (
        foreach(node(State, Cost), Nodes),
        foreach(H-node(State, Cost), H_Nodes),
        param(FinalState)
    do
        compute_distance_between_states(FinalState, State, H)
    ),
    sort(H_Nodes, Sorted_H_Nodes),
    keys_and_values(Sorted_H_Nodes, _, AscendingOrderedNodes),
    reverse(AscendingOrderedNodes, DescendingOrderedNodes),
    save_nodes(TargetCacheName, DescendingOrderedNodes),
    format('~a cache file successfully created.\n', [TargetCacheName]).

load_nodes_or_generate(Filename, Nodes) :-
    load_nodes(Filename, []),
    !,
    format('empty cache file found (~a). State generation launched.\n', [Filename]),
    bfs_generator(500, Nodes),
    save_nodes(Filename, Nodes).
load_nodes_or_generate(Filename, Nodes) :-
    load_nodes(Filename, Nodes).

% lowest nodes loading predicate
load_nodes(Filename, Nodes) :-
    deserialise_states(Filename, States),
    (
        foreach(Cost-State, States),
        foreach(node(State, Cost), Nodes)
    do
        true
    ).

% lowest nodes saving predicate
save_nodes(Filename, Nodes) :-
    (
        foreach(node(State, Cost), Nodes),
        foreach(Cost-State, States)
    do
        true
    ),
    serialise_states(Filename, States).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPER PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% compute_distance_between_states(+State1, +State2, -DistinctElements).
compute_distance_between_states(State1, State2, DistinctElements) :-
    ord_intersection(State1, State2, Intersection),
    length(Intersection, IntersectionLength),
    length(State1, Length1),
    length(State2, Length2),
    DistinctElements is (Length1 - IntersectionLength) + (Length2 - IntersectionLength).