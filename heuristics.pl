:- module(heuristics,
    [
        h_zero/2, h_cost/2, h_reversed_cost/2, h_state_length/2,
        h_subtract_with_i/2, h_subtract_with_g/2, h_subtract_with_path/2,
        h_distance_with_i/2, h_distance_with_g/2, h_distance_with_path/2,
        h_plus_with_i/2, h_plus_with_g/2, h_random/2,
        write_heuristics/0
    ]).

:- use_module(library(ordsets), [ord_subtract/3, ord_intersection/3, ord_union/2, ord_subset/2]).
:- use_module(library(sets), [is_set/1]).
:- use_module(library(random), [random/3]).

:- ensure_loaded(blackboard_data).
:- ensure_loaded(problem).
:- ensure_loaded(generators).
:- ensure_loaded(domain).

write_heuristics :-
    Predicates =
        [
            h_zero, h_cost, h_reversed_cost, h_state_length,
            h_subtract_with_i, h_subtract_with_g, h_subtract_with_path,
            h_distance_with_i, h_distance_with_g, h_distance_with_path,
            h_plus_with_i, h_plus_with_g, h_random
        ],
    format('\nheuristics available :\n~@\n', [write_list(Predicates)]).

write_list([]).
write_list([H|T]) :-
    write(H),
    nl,
    write_list(T).

%% h_zero(+Node, -NodeWithHeuristic).
h_zero(node(State, Cost, _), node(State, Cost, 0)).

h_random(node(State, Cost, _), node(State, Cost, Heuristic)) :-
    get_nb_nodes(NumberOfNodes),
    Maximum is 10 * NumberOfNodes,
    random(0, Maximum, Heuristic).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NODE-BASED HEURISTICS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% h_cost(+Node, -NodeWithHeuristic).
% The higher the cost to reach the node, the better the heuristic of the node is.
h_cost(node(State, Cost, _), node(State, Cost, Cost)).

%% h_reversed_cost(+Node, -NodeWithHeuristic).
% The lower the cost to reach the node, the better the heuristic of the node is.
h_reversed_cost(node(State, Cost, _), node(State, Cost, Heuristic)) :-
    get_source_result(SourceResult),
    length(SourceResult, SourceResultLength),
    Heuristic is SourceResultLength - Cost,
    Heuristic > 0,
    !.
% If its cost is higher than the size of the source result, then it is pointless (so it unifies 0).
h_reversed_cost(node(State, Cost, _), node(State, Cost, 0)).

%% h_state_length(+Node, -NodeWithHeuristic).
h_state_length(node(State, Cost, _), node(State, Cost, StateLength)) :-
    length(State, StateLength).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HEURISTICS BASED ON DISTANCE ESTIMATION WITH THE INITIAL STATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% h_subtract_with_i(+Node, -NodeWithHeuristic).
% The heuristic value is the size of the difference of the initial state and the current state.
h_subtract_with_i(node(State, Cost, _), node(State, Cost, DifferenceLength)) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    ord_subtract(InitialState, State, Difference),
    length(Difference, DifferenceLength).

%% h_distance_with_i(+Node, -NodeWithHeuristic).
% The heuristic value is the sum of distinct literals of the initial and the current state.
h_distance_with_i(node(State, Cost, _), node(State, Cost, DistinctElements)) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    compute_distance_between_states(InitialState, State, DistinctElements).

h_plus_with_i(node(State, Cost, _), node(State, Cost, Heuristic)) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    compute_h_plus_between_states(State, InitialState, Heuristic).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HEURISTICS BASED ON DISTANCE ESTIMATION WITH THE GOAL STATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% h_subtract_with_g(+Node, -NodeWithHeuristic).
% The heuristic value is the size of the intersection of the goal state and the current state.
h_subtract_with_g(node(State, Cost, _), node(State, Cost, DifferenceLength)) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    get_source_result(SourceResult),
    progress_from_state_with_plan(InitialState, SourceResult, SourceStates),
    append(_FirstStates, [GoalState], SourceStates),
    ord_subtract(GoalState, State, Difference),
    length(Difference, DifferenceLength).

%% h_distance_with_g(+Node, -NodeWithHeuristic).
% The heuristic value is the sum of distinct literals of the initial and the current state.
h_distance_with_g(node(State, Cost, _), node(State, Cost, DistinctElements)) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    get_source_result(SourceResult),
    progress_from_state_with_plan(InitialState, SourceResult, SourceStates),
    append(_FirstStates, [GoalState], SourceStates),
    compute_distance_between_states(GoalState, State, DistinctElements).

h_plus_with_g(node(State, Cost, _), node(State, Cost, Heuristic)) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    get_source_result(SourceResult),
    progress_from_state_with_plan(InitialState, SourceResult, SourceStates),
    append(_FirstStates, [GoalState], SourceStates),
    compute_h_plus_between_states(State, GoalState, Heuristic).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HEURISTICS BASED ON DISTANCE ESTIMATION WITH THE STATES OF THE PATH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h_subtract_with_path(node(State, Cost, _), node(State, Cost, Result)) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    get_source_result(SourceResult),
    progress_from_state_with_plan(InitialState, SourceResult, SourceStates),
    append([InitialState|FirstStates], [_GoalState], SourceStates),
    (
        foreach(PathState, FirstStates),
        fromto(0, In, Out, Result),
        param(State)
    do
        subtract_states(PathState, State, Distance),
        Out is In + Distance
    ).

h_distance_with_path(node(State, Cost, _), node(State, Cost, Result)) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    get_source_result(SourceResult),
    progress_from_state_with_plan(InitialState, SourceResult, SourceStates),
    append([InitialState|FirstStates], [_GoalState], SourceStates),
    (
        foreach(PathState, FirstStates),
        fromto(0, In, Out, Result),
        param(State)
    do
        compute_distance_between_states(PathState, State, Distance),
        Out is In + Distance
    ).

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

%% subtract_states(+State1, +State2, -DifferenceLength).
subtract_states(State1, State2, DifferenceLength) :-
    ord_subtract(State1, State2, Difference),
    length(Difference, DifferenceLength).

compute_h_plus_between_states(State, GoalState, 0) :-
    ord_subset(GoalState, State),
    !.
compute_h_plus_between_states(State, GoalState, Level) :-
    setof(PE, compute_positive_effects(State, PE), SetOfPE),
    ord_union([State|SetOfPE], TmpState),
    sort(TmpState, NewState),
    NewState \= State, % only needed for some strange problems (see the airport domain)
    compute_h_plus_between_states(NewState, GoalState, NextLevel),
    Level is NextLevel + 1.
% limits the search in case of unreachable goal state
compute_h_plus_between_states(_State, _GoalState, 0).

%% compute_positive_effects(+State, -PositiveEffects).
% generates a possible action (from the state) and retrieves only its positive effects
compute_positive_effects(State, PositiveEffects) :-
    generate_action(Action),
    action_parameters(Action, Parameters),
    instantiate_parameters(Parameters),
    is_set(Parameters),
    action_preconditions(Action, TmpPreconditions),
    sort(TmpPreconditions, Preconditions),
    ord_subset(Preconditions, State),
    % gets only its positive effects
    action_positive_effects(Action, TmpPositiveEffects),
    sort(TmpPositiveEffects, PositiveEffects).