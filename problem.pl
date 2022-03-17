:- module(problem,
    [
        problem_name/2, problem_initial_state/2, problem_goal_state/2, problem_objects/2,
        sort_problem/2, process_problem/5
    ]).

:- use_module(library(ordsets), [is_ordset/1, ord_union/2]).
:- use_module(library(lists), [maplist/3]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PROBLEM STRUCTURE AND ACCESSORS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% problem(_Name, _Domain, _Requirements, _ObjectsDeclaration, _I, _G, _C, _MS, _LS).

problem_name(problem(Name, _, _, _, _, _, _, _, _), Name).
problem_initial_state(problem(_, _, _, _, Init, _, _, _, _), Init).
problem_goal_state(problem(_, _, _, _, _, Goal, _, _, _), Goal).
problem_objects(problem(_, _, _, Objects, _, _, _, _, _), Objects).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PROBLEM-RELATED PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sort_problem(problem(N, D, R, OD, I, G, C, MS, LS), problem(N, D, R, OD, SI, SG, C, MS, LS)) :-
    sort(I, SI),
    is_ordset(SI),
    sort(G, SG),
    is_ordset(SG).

% processes the problem by returning the ground rigid relations of the initial state as well as the number of each static fluent
%% process_problem(+problem, +RigidRelationNames, +StaticRelationNames, -RigidRelations, -NumberForEachStaticFluent).
process_problem(Problem, RigidRelationNames, StaticFluentNames, RigidRelations, NumberForEachStaticFluent) :-
    problem_initial_state(Problem, InitialState),
    maplist(predicates_from_name(InitialState), RigidRelationNames, TmpRigidRelations),
    ord_union(TmpRigidRelations, RigidRelations),
    % format('rigid relations of problem are :\n~p\n', [RigidRelations]),
    maplist(predicates_from_name(InitialState), StaticFluentNames, StaticFluents),
    maplist(length, StaticFluents, NumberForEachStaticFluent).
    % format('number of static fluents of problem : ~p\n', [NumberForEachStaticFluent]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% accumulates the predicates whose name matches Name
%% predicates_from_name(Pred, Name, Res).
predicates_from_name([], _, []).
predicates_from_name([H|T1], Name, [H|T2]) :-
    H =.. [Name|_],
    !,
    predicates_from_name(T1, Name, T2).
predicates_from_name([_|T1], Name, Results) :-
    predicates_from_name(T1, Name, Results).