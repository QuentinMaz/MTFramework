:- module(configuration,
    [
        deserialise_configuration/2,
        configuration_planner_command/2, configuration_domain_filename/2, configuration_problem_filename/2,
        configuration_result_filename/2, configuration_nb_tests/2, configuration_generators/2,
        configuration_heuristic/2, configuration_metamorphic_relation/2, configuration_run_all_tests/2
    ]).

:- use_module(read_file).
:- use_module(library(plunit)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CONFIGURATION STRUCTURE AND ACCESSORS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% configuration(_PlannerCommand, _DomainFilename, _ProblemFilename, _ResultFilename, _MetamorphicRelation, _NumberOfTests, _RunAllTests, _GeneratorsPredicates, _HeuristicPredicate).

configuration_planner_command(configuration(PlannerCommand, _, _, _, _, _, _, _, _), PlannerCommand).
configuration_domain_filename(configuration(_, DomainFilename, _, _, _, _, _, _, _), DomainFilename).
configuration_problem_filename(configuration(_, _, ProblemFilename, _, _, _, _, _, _), ProblemFilename).
configuration_result_filename(configuration(_, _, _, ResultFilename, _, _, _, _, _), ResultFilename).
configuration_metamorphic_relation(configuration(_, _, _, _, MetamorphicRelation, _, _, _, _), MetamorphicRelation).
configuration_nb_tests(configuration(_, _, _, _, _, NumberOfTests, _, _, _), NumberOfTests).
configuration_run_all_tests(configuration(_, _, _, _, _, _, RunAllTests, _, _), RunAllTests).
configuration_generators(configuration(_, _, _, _, _, _, _, GeneratorsPredicates, _), GeneratorsPredicates).
configuration_heuristic(configuration(_, _, _, _, _, _, _, _, HeuristicPredicate), HeuristicPredicate).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CONFIGURATION PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% deserialise_configuration(+Filename, -Configuration).
deserialise_configuration(Filename, Configuration) :- deserialise_configuration(Filename, Configuration, []).

%% deserialise_configuration(+Filename, -Configuration, -RestOfFile).
deserialise_configuration(Filename, Configuration, RestOfFile) :-
    read_file(Filename, List),
    json_configuration(Configuration, List, RestOfFile),
    % grounds free variables of the configuration (in case of optional parameters)
    instantiate_optional_parameters(Configuration).

json_configuration(configuration(PC, DF, PF, RF, MR, NbTests, RAT, Gen, H))
    --> ['{'], key_string_value(command, PC), [','],
        key_string_value(domain, DF), [','],
        key_string_value(problem, PF), [','],
        (key_string_value(result, RF), [','] ; []), % result filename is optional
        key_string_value(metamorphic_relation, MR), [','],
        (key_integer_value(number_of_tests, NbTests), [','] ; []), % number of tests optional
        (key_boolean_value(run_all_tests, RAT), [','] ; []), % run_all_tests is optional
        key_objects(generators, argument, Gen), [','],
        key_object_value(heuristic, argument, H),
        ['}'].

key_string_value(Key, StringValue) --> [Key], [':'], string_value(StringValue).
key_integer_value(Key, IntegerValue) --> [Key], [':'], integer_value(IntegerValue).
key_boolean_value(Key, BooleanValue) --> [Key], [':'], boolean_value(BooleanValue).
key_object_value(Key, Object, ObjectValue) --> [Key], [':'], object_value(Object, ObjectValue).
key_objects(Key, Object, Objects) --> [Key], [':'], ['['], maybe_more(Object, Objects), [']'].

object_value(ObjectName, ObjectValue, Input, Output) :-
    Rule =.. [ObjectName, ObjectValue, Input, Output],
    Rule.

maybe_more(_Object, []) --> [].
maybe_more(Object, [Element|Tail]) --> object_value(Object, Element), [','], maybe_more(Object, Tail).
maybe_more(Object, [Element]) --> object_value(Object, Element).

%% rules called by object_value/2 must return atom and not list !
argument(Argument) --> ['{'], key_string_value(name, Name), [','], key_objects(arguments, element, Args), ['}'], {Argument =.. [Name|Args]}.
argument(ArgumentWithoutParameter) --> ['{'], key_string_value(name, ArgumentWithoutParameter), ['}'].
element(Element) --> [Element].

%% tokens
boolean_value(true) --> [true].
boolean_value(false) --> [false].
integer_value(I) --> [I], {integer(I)}.
string_value(V) --> [V], {integer(V), !, fail}.
string_value(V) --> [V], {float(V), !, fail}.
string_value(V) --> [V].

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CONFIGURATION HELPER PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

instantiate_optional_parameters(Configuration) :-
    configuration_result_filename(Configuration, ResultFilename),
    (ground(ResultFilename) -> true ; ResultFilename = 'test.csv'),
    configuration_nb_tests(Configuration, NumberOfTests),
    (ground(NumberOfTests) -> true ; NumberOfTests = 15),
    configuration_run_all_tests(Configuration, RAT),
    (ground(RAT) -> true ; RAT = true).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLUNIT TESTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(basic_configuration).

get_configuration(configuration('planner_command', 'domain_filename', 'problem_filename', 'result_filename', mr0, 42, false, [generator0(2), generator1(0,2)], heuristic0)).

test(planner_command_accessor, [setup(get_configuration(Config))]) :-
    configuration_planner_command(Config, 'planner_command').

test(domain_filename_accessor, [setup(get_configuration(Config))]) :-
    configuration_domain_filename(Config, 'domain_filename').

test(problem_filename_accessor, [setup(get_configuration(Config))]) :-
    configuration_problem_filename(Config, 'problem_filename').

test(result_filename_accessor, [setup(get_configuration(Config))]) :-
    configuration_result_filename(Config, 'result_filename').

test(metamorphic_relation_accessor, [setup(get_configuration(Config))]) :-
    configuration_metamorphic_relation(Config, mr0).

test(nb_tests_accessor, [setup(get_configuration(Config))]) :-
    configuration_nb_tests(Config, 42).

test(run_all_tests_accessor, [setup(get_configuration(Config))]) :-
    configuration_run_all_tests(Config, false).

test(generators_accessor, [setup(get_configuration(Config))]) :-
    configuration_generators(Config, [generator0(2),generator1(0,2)]).

test(heuristic_accessor, [setup(get_configuration(Config))]) :-
    configuration_heuristic(Config, heuristic0).

:- end_tests(basic_configuration).