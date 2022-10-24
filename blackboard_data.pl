:- module(blackboard_data,
    [
        set_blackboard/4, set_blackboard/3, set_source_result/1, set_nb_tests/1, set_final_state/1,
        get_problem/1, get_domain/1, get_actions/1, get_objects/1, get_constants/1,
        get_source_result/1, get_configuration/1, get_cache_name/1, get_final_state/1,
        get_operators/1, get_untyped_variables/1, get_typed_variables/1,

        get_mf/1
    ]).

:- use_module(library(sets), [subtract/3]).

:- ensure_loaded(domain).
:- ensure_loaded(problem).
:- ensure_loaded(configuration).
:- ensure_loaded(utils).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BLACKBOARD SETTERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% set_blackboard(+Configuration, +Domain, +Problem, +CacheName).
set_blackboard(Configuration, Domain, Problem, CacheName) :-
    bb_put(configuration, Configuration),
    configuration_nb_tests(Configuration, NumberOfTests),
    bb_put(number_of_tests, NumberOfTests),
    set_blackboard(Domain, Problem, CacheName).

set_blackboard(Domain, Problem, CacheName) :-
    bb_put(domain, Domain),
    bb_put(source_problem, Problem),
    bb_put(cache_name, CacheName),
    domain_actions(Domain, Actions),
    bb_put(actions, Actions),
    domain_constants(Domain, Constants),
    bb_put(constants, Constants),
    problem_objects(Problem, Objects),
    bb_put(objects, Objects),
    domain_predicates(Domain, Predicates),
    bb_put(predicates, Predicates),
    compute_variables(Domain, Problem),
    compute_rigid_predicates(Domain, Problem, RigidPredicatesNames, RigidFacts),
    % length(RigidFacts, LRF), format('~d ground rigid facts found.\n', [LRF]),
    ground_actions(RigidPredicatesNames, RigidFacts, Operators),
    % length(Operators, LO), format('~d ground actions found.\n', [LO]),
    bb_put(operators, Operators),
    problem_initial_state(Problem, InitialState),
    compute_mandatory_facts(InitialState, Operators, MandatoryFacts),
    bb_put(mf, MandatoryFacts).


set_source_result(SourceResult) :-
    bb_put(source_result, SourceResult).

set_nb_tests(NumberOfTests) :-
    bb_put(number_of_tests, NumberOfTests).

set_final_state(FinalState) :-
    bb_put(final_state, FinalState).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BLACKBOARD GETTERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

get_configuration(Configuration) :- bb_get(configuration, Configuration).

get_domain(Domain) :- bb_get(domain, Domain).

get_problem(Problem) :- bb_get(source_problem, Problem).

get_actions(Actions) :- bb_get(actions, Actions).

get_objects(Objects) :- bb_get(objects, Objects).

get_constants(Constants) :- bb_get(constants, Constants).

get_source_result(SourceResult) :- bb_get(source_result, SourceResult).

get_cache_name(CacheName) :- bb_get(cache_name, CacheName).

get_final_state(FinalState) :- bb_get(final_state, FinalState).

get_untyped_variables(UntypedVariables) :- bb_get(untyped_variables, UntypedVariables).

get_typed_variables(TypedVariables) :- bb_get(typed_variables, TypedVariables).

get_operators(Operators) :- bb_get(operators, Operators).

get_mf(MandatoryFacts) :- bb_get(mf, MandatoryFacts).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% VARIABLES PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

compute_variables(Domain, Problem) :-
    domain_constants(Domain, Constants),
    problem_objects(Problem, Objects),
    append(Constants, Objects, Variables),
    (
        foreach(Variable, Variables),
        fromto([], TypedIn, TypedOut, TypedVariables),
        fromto([], UntypedIn, UntypedOut, UntypedVariables)
    do
        (   atom(Variable) -> (UntypedOut = [Variable|UntypedIn], TypedOut = TypedIn)
        ;
            (UntypedOut = UntypedIn, TypedOut = [Variable|TypedIn])
        )
    ),
    bb_put(untyped_variables, UntypedVariables),
    % format('untyped variables: ~p\n', [UntypedVariables]),
    % format('Typed variables: ~p\n', [TypedVariables]),
    domain_types(Domain, Types),
    % format('Types: ~p\n', [Types]),
    (
        foreach(TypedVariable, TypedVariables),
        fromto([], In, Out, NewTypedVariables),
        param(Types)
    do
        TypedVariable =.. [Type, Variable],
        % #TODO : inherited types are each time re-computed
        get_inherited_types([Type], Types, InheritedTypes),
        % mapping the list of inherited types with the current variable
        (
            foreach(InheritedType, InheritedTypes),
            foreach(InheritedTypedVariable, InheritedTypedVariables),
            param(Variable)
        do
            InheritedTypedVariable =.. [InheritedType, Variable]
        ),
        append(In, InheritedTypedVariables, Out)
    ),
    append(NewTypedVariables, TypedVariables, FinalTypedVariables),
    filter_predicates(FinalTypedVariables, object, VariablesOfTypeObject),
    subtract(FinalTypedVariables, VariablesOfTypeObject, Results),
    % format('Typed variables: ~p\n', [Results]),
    bb_put(typed_variables, Results).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% inherited_types(+Type, +Types, -InheritedTypes).
inherited_types(_, [], []).
inherited_types(Type, [H|T], [InheritedType|R]) :-
    H =.. [InheritedType, Type],
    !,
    inherited_types(Type, T, R).
inherited_types(Type, [_|T], InheritedTypes) :-
    inherited_types(Type, T, InheritedTypes).

get_inherited_types([], _, []).
get_inherited_types([Type|T], Types, InheritedTypes) :-
    inherited_types(Type, Types, IT1),
    append(T, IT1, NewT),
    get_inherited_types(NewT, Types, IT2),
    append(IT1, IT2, InheritedTypes).