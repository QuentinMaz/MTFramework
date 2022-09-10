:- module(blackboard_data,
    [
        set_blackboard/3, set_source_result/1, set_nb_nodes/1, set_nb_tests/1, set_final_state/1,
        get_problem/1, get_domain/1, get_actions/1, get_objects/1, get_constants/1,
        get_source_result/1, get_configuration/1, get_nb_tests/1, get_nb_nodes/1, get_final_state/1,
        update_nb_nodes/1, keep_generating/0,
        get_rigid_relations/1, get_rigid_relation_names/1, get_static_fluent_names/1, get_static_fluent_numbers/1
    ]).

:- ensure_loaded(domain).
:- ensure_loaded(problem).
:- ensure_loaded(configuration).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BLACKBOARD SETTERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% set_blackboard(+Configuration, +Domain, +Problem).
set_blackboard(Configuration, Domain, Problem) :-
    process_domain(Domain, RigidRelationNames, StaticFluentNames),
    process_problem(Problem, RigidRelationNames, StaticFluentNames, RigidRelations, NumberForEachStaticFluent),
    bb_put(rigid_relations, RigidRelations),
    bb_put(rigid_relation_names, RigidRelationNames),
    bb_put(static_fluent_names, StaticFluentNames),
    bb_put(number_for_each_static_fluent, NumberForEachStaticFluent),
    bb_put(configuration, Configuration),
    configuration_nb_tests(Configuration, NumberOfTests),
    bb_put(number_of_tests, NumberOfTests),
    bb_put(number_of_nodes, 0),
    bb_put(domain, Domain),
    bb_put(source_problem, Problem),
    domain_actions(Domain, Actions),
    bb_put(actions, Actions),
    domain_constants(Domain, Constants),
    bb_put(constants, Constants),
    problem_objects(Problem, Objects),
    bb_put(objects, Objects).

set_source_result(SourceResult) :-
    bb_put(source_result, SourceResult).

set_final_state(FinalState) :-
    bb_put(final_state, FinalState). 

set_nb_tests(NumberOfTests) :-
    bb_put(number_of_tests, NumberOfTests).

set_nb_nodes(NumberOfNodes) :-
    bb_put(number_of_nodes, NumberOfNodes).

update_nb_nodes(N) :-
    bb_get(number_of_nodes, NumberOfNodes),
    UpdatedNbNodes is NumberOfNodes + N,
    bb_put(number_of_nodes, UpdatedNbNodes).

keep_generating :-
    NB_NODES_MAX = 5000,
    bb_get(number_of_nodes, NumberOfNodes),
    NumberOfNodes =< NB_NODES_MAX.

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

get_nb_tests(NumberOfTests) :- bb_get(number_of_tests, NumberOfTests).

get_nb_nodes(NumberOfNodes) :- bb_get(number_of_nodes, NumberOfNodes).

get_rigid_relation_names(RigidRelationNames) :- bb_get(rigid_relation_names, RigidRelationNames).

get_rigid_relations(RigidRelations) :- bb_get(rigid_relations, RigidRelations).

get_static_fluent_names(StaticFluentNames) :- bb_get(static_fluent_names, StaticFluentNames).

get_static_fluent_numbers(NumberForEachStaticFluent) :- bb_get(number_for_each_static_fluent, NumberForEachStaticFluent).

get_final_state(FinalState) :- bb_get(final_state, FinalState).  