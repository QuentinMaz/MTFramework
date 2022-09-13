:- module(configuration,
    [
        deserialise_configuration/2,
        configuration_planners_commands/2, configuration_domain_filename/2, configuration_problem_filename/2,
        configuration_results_filenames/2, configuration_nb_tests/2, configuration_generator/2,
        configuration_heuristics/2, configuration_metamorphic_relation/2, configuration_run_all_tests/2
    ]).

:- use_module(read_file).
:- use_module(library(plunit)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CONFIGURATION STRUCTURE AND ACCESSORS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% configuration(_PlannersCommands, _DomainFilename, _ProblemFilename, _ResultFilenames, _MetamorphicRelation, _NumberOfTests, _RunAllTests, _GeneratorPredicate, _HeuristicsPredicates).

configuration_planners_commands(configuration(PlannersCommands, _, _, _, _, _, _, _, _), PlannersCommands).
configuration_domain_filename(configuration(_, DomainFilename, _, _, _, _, _, _, _), DomainFilename).
configuration_problem_filename(configuration(_, _, ProblemFilename, _, _, _, _, _, _), ProblemFilename).
configuration_results_filenames(configuration(_, _, _, ResultsFilenames, _, _, _, _, _), ResultsFilenames).
configuration_metamorphic_relation(configuration(_, _, _, _, MetamorphicRelation, _, _, _, _), MetamorphicRelation).
configuration_nb_tests(configuration(_, _, _, _, _, NumberOfTests, _, _, _), NumberOfTests).
configuration_run_all_tests(configuration(_, _, _, _, _, _, RunAllTests, _, _), RunAllTests).
configuration_generator(configuration(_, _, _, _, _, _, _, GeneratorsPredicate, _), GeneratorsPredicate).
configuration_heuristics(configuration(_, _, _, _, _, _, _, _, HeuristicsPredicates), HeuristicsPredicates).

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

json_configuration(configuration(PCs, DF, PF, RF, MR, NbTests, RAT, Gen, H))
    --> ['{'], key_string_values(planners_commands, PCs), [','],
        key_string_value(domain, DF), [','],
        key_string_value(problem, PF), [','],
        (key_string_values(results, RF), [','] ; []), % results filenames is optional
        key_string_value(metamorphic_relation, MR), [','],
        (key_integer_value(number_of_tests, NbTests), [','] ; []), % number of tests optional
        (key_boolean_value(run_all_tests, RAT), [','] ; []), % run_all_tests is optional
        key_object_value(generator, argument, Gen), [','],
        key_objects(heuristics, argument, H),
        ['}'].

key_string_value(Key, StringValue) --> [Key], [':'], string_value(StringValue).
key_string_values(Key, StringValues) --> [Key], [':'], ['['], string_values(StringValues), [']'].
key_integer_value(Key, IntegerValue) --> [Key], [':'], integer_value(IntegerValue).
key_boolean_value(Key, BooleanValue) --> [Key], [':'], boolean_value(BooleanValue).
key_object_value(Key, Object, ObjectValue) --> [Key], [':'], object_value(Object, ObjectValue).
key_objects(Key, Object, Objects) --> [Key], [':'], ['['], maybe_more(Object, Objects), [']'].

object_value(ObjectName, ObjectValue, Input, Output) :-
    Rule =.. [ObjectName, ObjectValue, Input, Output],
    Rule.

string_values([StringValue]) --> string_value(StringValue).
string_values([String|Tail]) --> string_value(String), [','], string_values(Tail).

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
    configuration_results_filenames(Configuration, ResultsFilenames),
    (   
        ground(ResultsFilenames) -> 
            true 
        ;
        (
            configuration_planners_commands(Configuration, PlannersCommands),
            length(PlannersCommands, N),
            (
                for(Index, 1, N),
                foreach(ResultsFilename, Filenames)
            do
                number_codes(Index, Codes),
                atom_codes(IndexAtom, Codes),
                atom_concat('test', IndexAtom, Tmp),
                atom_concat(Tmp, '.csv', ResultsFilename)
            ),
            ResultsFilenames = Filenames
        )
    ), 
    configuration_nb_tests(Configuration, NumberOfTests),
    (ground(NumberOfTests) -> true ; NumberOfTests = 15),
    configuration_run_all_tests(Configuration, RAT),
    (ground(RAT) -> true ; RAT = true).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLUNIT TESTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(accessors_predicates).

get_configuration(configuration(['planner_command'], 'domain_filename', 'problem_filename', ['result_filename'], mr0, 42, false, generator0(2), [heuristic0])).

test(planners_commands_accessor, [setup(get_configuration(Config))]) :-
    configuration_planners_commands(Config, ['planner_command']).

test(domain_filename_accessor, [setup(get_configuration(Config))]) :-
    configuration_domain_filename(Config, 'domain_filename').

test(problem_filename_accessor, [setup(get_configuration(Config))]) :-
    configuration_problem_filename(Config, 'problem_filename').

test(result_filename_accessor, [setup(get_configuration(Config))]) :-
    configuration_results_filenames(Config, ['result_filename']).

test(metamorphic_relation_accessor, [setup(get_configuration(Config))]) :-
    configuration_metamorphic_relation(Config, mr0).

test(nb_tests_accessor, [setup(get_configuration(Config))]) :-
    configuration_nb_tests(Config, 42).

test(run_all_tests_accessor, [setup(get_configuration(Config))]) :-
    configuration_run_all_tests(Config, false).

test(generator_accessor, [setup(get_configuration(Config))]) :-
    configuration_generator(Config, generator0(2)).

test(heuristics_accessor, [setup(get_configuration(Config))]) :-
    configuration_heuristics(Config, [heuristic0]).

:- end_tests(accessors_predicates).

:- begin_tests(parsing_configuration).

test(planners_commands_accessor, [setup(deserialise_configuration('configurations/test.json', Config))]) :-
    configuration_planners_commands(Config, ['planner_command1', 'planner_command2', 'planner_command3']).

test(domain_filename_accessor, [setup(deserialise_configuration('configurations/test.json', Config))]) :-
    configuration_domain_filename(Config, 'domain_filename').

test(problem_filename_accessor, [setup(deserialise_configuration('configurations/test.json', Config))]) :-
    configuration_problem_filename(Config, 'problem_filename').

test(result_filename_accessor, [setup(deserialise_configuration('configurations/test.json', Config))]) :-
    configuration_results_filenames(Config, ['test1.csv', 'test2.csv', 'test3.csv']).

test(metamorphic_relation_accessor, [setup(deserialise_configuration('configurations/test.json', Config))]) :-
    configuration_metamorphic_relation(Config, mr0).

test(nb_tests_accessor, [setup(deserialise_configuration('configurations/test.json', Config))]) :-
    configuration_nb_tests(Config, 42).

test(run_all_tests_accessor, [setup(deserialise_configuration('configurations/test.json', Config))]) :-
    configuration_run_all_tests(Config, false).

test(generator_accessor, [setup(deserialise_configuration('configurations/test.json', Config))]) :-
    configuration_generator(Config, generator0(2)).

test(heuristics_accessor, [setup(deserialise_configuration('configurations/test.json', Config))]) :-
    configuration_heuristics(Config, [heuristic0]).

:- end_tests(parsing_configuration).