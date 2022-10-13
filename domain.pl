:- module(domain,
    [
        domain_actions/2, domain_constants/2, domain_name/2, domain_predicates/2, domain_types/2,
        action_parameters/2, action_preconditions/2, action_positive_effects/2, action_negative_effects/2, untyped_action/2,
        ground_predicates/4, ground_actions/3, generate_action/1
    ]).

:- use_module(library(ordsets), [ord_subtract/3, ord_union/2, ord_subset/2]).
:- use_module(library(lists), [maplist/3]).
:- use_module(library(sets), [is_set/1]).

:- ensure_loaded(blackboard_data).
:- ensure_loaded(problem).
:- ensure_loaded(utils).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DOMAIN STRUCTURE AND ACCESSORS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% domain(_Name, _Requirements, _Types, _Constants, _Predicates, _Functions, _Constraints, _SructureDefinitions).

domain_name(domain(Name, _, _, _, _, _, _, _), Name).
domain_actions(domain(_, _, _, _, _, _, _, Actions), Actions).
domain_constants(domain(_, _, _, Constants, _, _, _, _), Constants).
domain_predicates(domain(_, _, _, _, Predicates, _, _, _), Predicates).
domain_types(domain(_, _, Types, _, _, _, _, _), Types).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ACTIONS PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

action_parameters(action(_, Parameters, _, _, _, _), Parameters).
action_preconditions(action(_, _, Preconditions, _, _, _), Preconditions).
action_positive_effects(action(_, _, _, PositiveEffects, _, _), PositiveEffects).
action_negative_effects(action(_, _, _, _, NegativeEffects, _), NegativeEffects).
untyped_action(action(Name, Parameters, _, _, _, _), UntypedAction) :-
    untype_parameters(Parameters, UntypedParameters),
    UntypedAction =.. [Name|UntypedParameters].

%% untype_parameters(+Parameters, -UntypedParameters).
untype_parameters([], []).
untype_parameters([TypedHead|T1], [UntypedHead|T2]) :-
    compound(TypedHead),
    TypedHead =.. [_Type, UntypedHead], % reminder : type(x)
    !,
untype_parameters(T1, T2).
untype_parameters([H|T1], [H|T2]) :-
    untype_parameters(T1, T2).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ACTIONS GENERATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PARAMETERS PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% instantiate_parameters(-GroundParameters).
% makes a ground instance of a list of parameters
instantiate_parameters([]).
% already ground case (the parameter is... a constant ?)
instantiate_parameters([Parameter|Ps]) :-
    ground(Parameter),
    !,
    % to do : check whether Parameter is declared in the constants of the problem (/!\ constants is a typed list !)
    instantiate_parameters(Ps).
% untyped case: it unifies Parameter with one of the untyped parameters
instantiate_parameters([Parameter|Ps]) :-
    var(Parameter),
    !,
    get_untyped_variables(UntypedParameters),
    member(Parameter, UntypedParameters),
    instantiate_parameters(Ps).
% typed case: it unifies Parameter with one of the matching typed parameters
instantiate_parameters([Parameter|Ps]) :-
    \+ ground(Parameter),
    !,
    Parameter =.. [TypeName, Var], % type(var)
    TypedParameter =.. [TypeName, Var], % type(parameter)
    get_typed_variables(TypedParameters),
    member(TypedParameter, TypedParameters),
    instantiate_parameters(Ps).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GROUNDING PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ground_predicates(Domain, Problem, RigidPredicatesNames, RigidFacts) :-
    domain_predicates_names(Domain, _PN, _FN, RigidPredicatesNames),
    % format('predicates:\n~p\n\nfluents:\n~p\n\nrigid predicates:\n~p\n\n', [PN, FN, RigidPredicatesNames]),
    % ground the rigid predicates of the current problem
    problem_initial_state(Problem, InitialState),
    filter_predicates_with_names(InitialState, RigidPredicatesNames, Tmp),
    ord_union(Tmp, RigidFacts).

ground_actions(RigidPredicatesNames, RigidFacts, Operators) :-
    findall(A,
    (
        generate_action(A), action_parameters(A, Parameters), instantiate_parameters(Parameters), % is_set(Parameters),
        action_preconditions(A, P), filter_predicates_with_names(P, RigidPredicatesNames, TmpRF), ord_union(TmpRF, RF), ord_subset(RF, RigidFacts)
    )
    , Operators).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

domain_predicates_names(Domain, PredicatesNames, FluentNames, RigidPredicatesNames) :-
    domain_predicates(Domain, Predicates),
    domain_actions(Domain, Actions),
    % finds state-invariant predicates aka rigid predicates
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
    sort(PNames, PredicatesNames),
    ord_subtract(PredicatesNames, FluentNames, RigidPredicatesNames).