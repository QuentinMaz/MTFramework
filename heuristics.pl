:- module(heuristics,
    [
        h_zero/2, h_state_length/2,
        h_distance_with_i/2, h_distance_with_g/2,
        h_random/2,
        write_heuristics/0
    ]).

:- use_module(library(ordsets), [ord_intersection/3]).
:- use_module(library(random), [random/3]).

:- ensure_loaded(blackboard_data).
:- ensure_loaded(problem).
:- ensure_loaded(generators).
:- ensure_loaded(domain).

write_heuristics :-
    Predicates =
        [
            h_zero, h_state_length,
            h_distance_with_i, h_distance_with_g,
            h_random
        ],
    format('\nheuristics available :\n~@\n', [write_list(Predicates)]).

write_list([]).
write_list([H|T]) :-
    write(H),
    nl,
    write_list(T).

%% h_zero(+Node, -Heuristic).
h_zero(node(_State, _Cost, _Heuristics), 0).

%% h_random(+Node, -Heuristic).
h_random(node(_State, _Cost, _Heuristics), Heuristic) :-
    get_nb_nodes(NumberOfNodes),
    Maximum is 10 * NumberOfNodes,
    random(0, Maximum, Heuristic).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NODE-BASED HEURISTICS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% h_state_length(+Node, -Heuristic).
h_state_length(node(State, _Cost, _Heuristics), StateLength) :-
    length(State, StateLength).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HEURISTICS BASED ON DISTANCE ESTIMATION WITH THE INITIAL STATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% h_distance_with_i(+Node, -Heuristic).
% The heuristic value is the sum of distinct literals of the initial and the current state.
h_distance_with_i(node(State, _Cost, _Heuristics), DistinctElements) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    compute_distance_between_states(InitialState, State, DistinctElements).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HEURISTICS BASED ON DISTANCE ESTIMATION WITH THE GOAL STATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% h_distance_with_g(+Node, -Heuristic).
% The heuristic value is the sum of distinct literals of the initial and the current state.
h_distance_with_g(node(State, _Cost, _Heuristics), DistinctElements) :-
    get_final_state(FinalState),
    compute_distance_between_states(FinalState, State, DistinctElements).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% compute_distance_between_states(+State1, +State2, -DistinctElements).
compute_distance_between_states(State1, State2, DistinctElements) :-
    ord_intersection(State1, State2, Intersection),
    length(Intersection, IntersectionLength),
    length(State1, Length1),
    length(State2, Length2),
    DistinctElements is (Length1 - IntersectionLength) + (Length2 - IntersectionLength).