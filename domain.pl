:- module(domain,
    [
        domain_actions/2, domain_constants/2, domain_name/2, domain_predicates/2,
        action_parameters/2, action_preconditions/2, action_positive_effects/2, action_negative_effects/2, action_definition/2,
        generate_action/1,
        instantiate_parameters/1,
        process_domain/3, check_rigid_relations/3, check_static_fluents/3
    ]).

:- use_module(library(ordsets), [ord_subtract/3, ord_union/2, ord_subset/2]).
:- use_module(library(lists), [maplist/3]).

:- ensure_loaded(blackboard_data).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DOMAIN STRUCTURE AND ACCESSORS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% domain(_Name, _Requirements, _Types, _Constants, _Predicates, _Functions, _Constraints, _SructureDefinitions).

domain_name(domain(Name, _, _, _, _, _, _, _), Name).
domain_predicates(domain(_, _, _, _, Predicates, _, _, _), Predicates).
domain_actions(domain(_, _, _, _, _, _, _, Actions), Actions).
domain_constants(domain(_, _, _, Constants, _, _, _, _), Constants).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ACTIONS PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

action_parameters(action(_, Parameters, _, _, _, _), Parameters).
action_preconditions(action(_, _, Preconditions, _, _, _), Preconditions).
action_positive_effects(action(_, _, _, PositiveEffects, _, _), PositiveEffects).
action_negative_effects(action(_, _, _, _, NegativeEffects, _), NegativeEffects).
action_definition(action(Name, Parameters, _, _, _, _), ActionDefinition) :-
    untype_parameters(Parameters, UntypedParameters),
    ActionDefinition =.. [Name|UntypedParameters].

%% untype_parameters(+Parameters, -UntypedParameters).
untype_parameters([], []).
untype_parameters([TypedHead|T1], [UntypedHead|T2]) :-
    compound(TypedHead),
    TypedHead =.. [_Type, UntypedHead], % reminder : type(x)
    !,
untype_parameters(T1, T2).
untype_parameters([H|T1], [H|T2]) :-
    untype_parameters(T1, T2).

%% generate_action(-Action).
generate_action(Action) :-
    get_actions(As),
    generate_action(As, Action).

%% generate_action(+PDDLActions, -Action).
% generates a free action from a list of pddl-like action
generate_action(PDDLActions, Action) :-
    member(ActionPDDL, PDDLActions),
    copy_pddl_terms(ActionPDDL, Action).

% Special version of copy_term. variable x represented as ?(x)
% All occurs of ?(x) are replaced with real prolog variables.
% Modified version of code published by Bartak: http://kti.mff.cuni.cz/~bartak/prolog/data_struct.html
copy_pddl_terms(A, B) :- copy_pddl_terms(A, [], B, _).

copy_pddl_terms(A, Vars, A, Vars) :-
    atomic(A),
    A \= ?(_). % A does NOT represent a term to turn into a variable
copy_pddl_terms(?(V), Vars, NV, NVars) :-
    atomic(V), % A does represent a term to turn into a variable
    register_variable(V, Vars, NV, NVars). % ... and so we either get the associated variable or register a new one
copy_pddl_terms(Term, Vars, NTerm, NVars):-
    compound(Term),
    Term \= ?(_),
    Term =.. [F|Args],
    copy_pddl_arguments(Args, Vars, NArgs, NVars),
    NTerm =.. [F|NArgs].

copy_pddl_arguments([H|T], Vars, [NH|NT], NVars) :-
    copy_pddl_terms(H, Vars, NH, SVars),
    copy_pddl_arguments(T, SVars, NT, NVars).
copy_pddl_arguments([], Vars, [], Vars).

%% register_variable(+T, +L, -N, -NL).
% browses the list of couples term/var L to retrieve the variable N associated to the term T.
% If there is no such association yet, then it registers a new variable (ie, a new couple is added to NL).
register_variable(V, [X/H|T], N, [X/H|NT]) :-
    V \== X , % different variables
    register_variable(V, T, N, NT).
register_variable(V, [X/H|T], H, [X/H|T]) :-
    V == X. % same variables
% registers a new variable N to the term V
register_variable(V, [], N, [V/N]).

%% instantiate_parameters(-GroundParameters).
instantiate_parameters(Parameters) :-
    get_objects(Objects),
    get_constants(Constants),
    instantiate_parameters(Objects, Constants, Parameters).

%% instantiate_parameters(+Objects, +Constants, -GroundParameters).
% makes a ground instance of a list of parameters
instantiate_parameters(_, _, []).
% already ground case (the parameter is a constant)
instantiate_parameters(Objects, Constants, [Param|Ps]) :-
    ground(Param),
    !,
    member(Param, Constants),
    instantiate_parameters(Objects, Constants, Ps).
% untyped case : it unifies Param with one of the untyped objects
instantiate_parameters(Objects, Constants, [Param|Ps]) :-
    var(Param),
    !,
    member(Param, Objects),
    instantiate_parameters(Objects, Constants, Ps).
% typed case : it unifies Param with one of the matching typed objects
instantiate_parameters(Objects, Constants, [Param|Ps]) :-
    \+ ground(Param),
    Param =.. [TypeName, Var], % type(var)
    TypedObject =.. [TypeName, Var], % type(object)
    member(TypedObject, Objects),
    instantiate_parameters(Objects, Constants, Ps).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MODEL CHECKING PREDICATES (STATE VALIDITY)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% processes the domain by returning the names of the rigid relations and static fluents
%% process_domain(+Domain, -RigidRelationNames, -StaticFluentNames).
process_domain(Domain, RigidRelationNames, StaticFluentNames) :-
    domain_predicates(Domain, Predicates),
    domain_actions(Domain, Actions),
    % format('\n~p\n~p\n', [Predicates, Actions]),
    % finds state-invariant predicates aka rigid relations
    % vs fluents = flexible relations
    (
        foreach(Action, Actions),
        fromto([], In, Out, Fluents)
    do
        action_positive_effects(Action, TmpPos),
        sort(TmpPos, Pos),
        action_negative_effects(Action, TmpNeg),
        sort(TmpNeg, Neg),
        ord_union([Pos, Neg, In], Out)
    ),
    predicates_names(Fluents, FNames),
    predicates_names(Predicates, PNames),
    sort(FNames, FluentNames),
    sort(PNames, PredicateNames),
    ord_subtract(PredicateNames, FluentNames, RigidRelationNames),
    % format('\npredicates of domain are :\n~p\nfluents of domain are :\n~p\nrigid relations of domain are :\n~p', [PredicateNames, FluentNames, RigidRelationNames]),
    % finds static fluents : fluents whose parameters can change but their number of instances never change
    (
        foreach(FluentName, FluentNames),
        fromto([], In, Out, SFNames),
        param(Actions)
    do
        (is_static_fluent(FluentName, Actions) -> Out = [FluentName|In] ; Out = In)
    ),
    sort(SFNames, StaticFluentNames).
    % format('\nstatic fluents of domain are :\n~p\n\n', [StaticFluentNames]).

% is_static_fluent(FluentName, Actions).
is_static_fluent(_, []).
is_static_fluent(Name, [Action|T]) :-
    action_positive_effects(Action, Pos),
    sort(Pos, SPos),
    predicates_from_name(SPos, Name, Tmp1),
    action_negative_effects(Action, Neg),
    sort(Neg, SNeg),
    predicates_from_name(SNeg, Name, Tmp2),
    length(Tmp1, L1),
    length(Tmp2, L2),
    R is L2 - L1,
    R == 0,
    is_static_fluent(Name, T).

%% check_rigid_relations(+State, +RigidRelationNames, +RigidRelations).
check_rigid_relations(State, RigidRelationNames, RigidRelations) :-
    maplist(predicates_from_name(State), RigidRelationNames, Tmp),
    ord_union(Tmp, FlatTmp),
    % format('\nrigid relations are :\n~p\nrigid relations of S are :\n~p\n', [RigidRelations, FlatTmp]),
    ord_subset(FlatTmp, RigidRelations).

% checks for each static fluent that its number of instances in State are lesser or equal wrt NumberOfStaticFluents
%% check_static_fluents(+State, +StaticFluentNames, +NumberForEachStaticFluent).
check_static_fluents(State, StaticFluentNames, NumberForEachStaticFluent) :-
    maplist(predicates_from_name(State), StaticFluentNames, Tmp),
    maplist(length, Tmp, StaticFluentNumbers),
    maplist(=<, StaticFluentNumbers, NumberForEachStaticFluent).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% retrieves all the names of the terms in Predicates (/!\ may have duplicates)
%% predicates_names(Predicates, PredicatesNames).
predicates_names([], []).
predicates_names([H|T1], [Name|T2]) :-
    H =.. [Name|_],
    predicates_names(T1, T2).

% accumulates the predicates whose name matches Name
%% predicates_from_name(Pred, Name, Res).
predicates_from_name([], _, []).
predicates_from_name([H|T1], Name, [H|T2]) :-
    H =.. [Name|_],
    !,
    predicates_from_name(T1, Name, T2).
predicates_from_name([_|T1], Name, Results) :-
    predicates_from_name(T1, Name, Results).