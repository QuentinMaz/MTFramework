:- module(configuration,
    [
        deserialise_configuration/2,
        configuration_planner_command/2, configuration_domain_filename/2, configuration_problem_filename/2,
        configuration_result_filename/2, configuration_output_filename/2, configuration_nb_tests/2, configuration_generators/2,
        configuration_heuristic/2, configuration_metamorphic_relation/2, configuration_run_all_tests/2
    ]).

:- use_module(read_file).
:- use_module(library(plunit)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CONFIGURATION STRUCTURE AND ACCESSORS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% configuration(_PlannerCommand, _DomainFilename, _ProblemFilename, _ResultFilename, _OutputFilename, _NumberOfTests, _GeneratorsPredicates).

configuration_planner_command(configuration(PlannerCommand, _, _, _, _, _, _), PlannerCommand).
configuration_domain_filename(configuration(_, DomainFilename, _, _, _, _, _), DomainFilename).
configuration_problem_filename(configuration(_, _, ProblemFilename, _, _, _, _), ProblemFilename).
configuration_result_filename(configuration(_, _, _, ResultFilename, _, _, _), ResultFilename).
configuration_output_filename(configuration(_, _, _, _, OutputFilename, _, _), OutputFilename).
configuration_nb_tests(configuration(_, _, _, _, _, NumberOfTests, _), NumberOfTests).
configuration_generators(configuration(_, _, _, _, _, _, GeneratorsPredicates), GeneratorsPredicates).

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

json_configuration(configuration(PC, DF, PF, RF, OF, NbTests, Gen))
    --> ['{'], key_string_value(planner_command, PC), [','],
        key_string_value(domain, DF), [','],
        key_string_value(problem, PF), [','],
        (key_string_value(result, RF), [','] ; []), % result filename is optional
        (key_string_value(output, OF), [','] ; []), % output filename is optional
        (key_integer_value(number_of_tests, NbTests), [','] ; []), % number of tests is optional
        key_objects(generators, argument, Gen),
        ['}'].

key_string_value(Key, StringValue) --> [Key], [':'], string_value(StringValue).
key_integer_value(Key, IntegerValue) --> [Key], [':'], integer_value(IntegerValue).
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
    configuration_output_filename(Configuration, OutputFilename),
    (ground(OutputFilename) -> true ; OutputFilename = 'tmp/output.txt').

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLUNIT TESTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(basic_configuration).

get_configuration(configuration('planner_command', 'domain_filename', 'problem_filename', 'result_filename', 'output_filename', 42, [generator0(2), generator1(0,2)])).

test(planner_command_accessor, [setup(get_configuration(Config))]) :-
    configuration_planner_command(Config, 'planner_command').

test(domain_filename_accessor, [setup(get_configuration(Config))]) :-
    configuration_domain_filename(Config, 'domain_filename').

test(problem_filename_accessor, [setup(get_configuration(Config))]) :-
    configuration_problem_filename(Config, 'problem_filename').

test(result_filename_accessor, [setup(get_configuration(Config))]) :-
    configuration_result_filename(Config, 'result_filename').

test(output_filename_accessor, [setup(get_configuration(Config))]) :-
    configuration_output_filename(Config, 'output_filename').

test(nb_tests_accessor, [setup(get_configuration(Config))]) :-
    configuration_nb_tests(Config, 42).

test(generator_accessor, [setup(get_configuration(Config))]) :-
    configuration_generators(Config, [generator0(2),generator1(0,2)]).

:- end_tests(basic_configuration).

:- begin_tests(parsing_configuration).

test(planner_command_accessor, [setup(deserialise_configuration('configurations/test.json', Config))]) :-
    configuration_planner_command(Config, 'planner_command').

test(domain_filename_accessor, [setup(deserialise_configuration('configurations/test.json', Config))]) :-
    configuration_domain_filename(Config, 'domain_filename').

test(problem_filename_accessor, [setup(deserialise_configuration('configurations/test.json', Config))]) :-
    configuration_problem_filename(Config, 'problem_filename').

test(result_filename_accessor, [setup(deserialise_configuration('configurations/test.json', Config))]) :-
    configuration_result_filename(Config, 'result_filename').

test(output_filename_accessor, [setup(deserialise_configuration('configurations/test.json', Config))]) :-
    configuration_output_filename(Config, 'output_filename').

test(nb_tests_accessor, [setup(deserialise_configuration('configurations/test.json', Config))]) :-
    configuration_nb_tests(Config, 42).

test(generator_accessor, [setup(deserialise_configuration('configurations/test.json', Config))]) :-
    configuration_generators(Config, [generator(42)]).

:- end_tests(parsing_configuration).