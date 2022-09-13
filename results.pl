:- module(results, [set_csv_result/5, write_csv_results/2]).

:- ensure_loaded(domain).
:- ensure_loaded(problem).
:- ensure_loaded(configuration).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RESULT PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% result(+Node, +Failure, +Error, +ExecutionTime, +ResultCost).
result(_Node, _Failure, _Error, _ExecutionTime, _ResultCost).

%% result_values(+Result, -Values).
result_values(result(node(State, Cost, HeuristicName, HeuristicValue), Failure, Error, ExecutionTime, ResultCost), ResultValues) :-
    ResultValues = [State, Cost, HeuristicName, HeuristicValue, Failure, Error, ExecutionTime, ResultCost].

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% WRITE HELPER PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% write_csv_header(+CsvFileStream, +Domain, +Problem, +Configuration, +SourceResultCost).
write_csv_header(CsvFileStream, Domain, Problem, Configuration, SourceResultCost) :-
    domain_name(Domain, DomainName),
    problem_name(Problem, ProblemName),
    configuration_metamorphic_relation(Configuration, MetamorphicRelation),
    configuration_nb_tests(Configuration, NumberOfTests),
    configuration_heuristics(Configuration, Heuristics),
    configuration_generator(Configuration, Generator),
    append([DomainName, ProblemName, SourceResultCost, MetamorphicRelation, NumberOfTests, Generator], Heuristics, CsvHeader),
    list_to_comma_seperated_list(CsvHeader, CsvHeaderRecord),
    write_list(CsvFileStream, CsvHeaderRecord).

write_csv_results_header(CsvFileStream) :-
    CsvResultsHeader = [node_cost, heuristic_name, heuristic_value, failure, error, 'execution_time(sec)', result_cost],
    list_to_comma_seperated_list(CsvResultsHeader, CsvResultsHeaderRecord),
    write_list(CsvFileStream, CsvResultsHeaderRecord).

%% write_csv_record_results(+CsvFileStream, +Results).
write_csv_record_results(_, []).
write_csv_record_results(CsvFileStream, [Result|Tail]) :-
    write_csv_record_result(CsvFileStream, Result),
    write_csv_record_results(CsvFileStream, Tail).

%% write_csv_record_result(+CsvFileStream, +Result).
write_csv_record_result(CsvFileStream, Result) :-
    result_values(Result, [_State|CsvResult]),
    list_to_comma_seperated_list(CsvResult, CsvResultRecord),
    write_list(CsvFileStream, CsvResultRecord).

%% list_to_comma_seperated_list(+List, -ListWithCommas).
list_to_comma_seperated_list([LastElement], [LastElement, '\n']).
list_to_comma_seperated_list([Head|T1], [Head, ','|T2]) :-
    list_to_comma_seperated_list(T1, T2).

%% write_list(+FileStream, +List).
write_list(_, []).
write_list(FileStream, [Head|Tail]) :-
    write(FileStream, Head),
    write_list(FileStream, Tail).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RESULTS EXPORTING PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% set_csv_result(+Domain, +Problem, +Configuration, +CsvFilename, +SourceResultCost).
set_csv_result(Domain, Problem, Configuration, CsvFilename, SourceResultCost) :-
    open(CsvFilename, write, CsvFileStream),
    write_csv_header(CsvFileStream, Domain, Problem, Configuration, SourceResultCost),
    write_csv_results_header(CsvFileStream),
    close(CsvFileStream).

%% write_csv_results(+CsvFilename, +Results) :-
write_csv_results(CsvFilename, Results) :-
    open(CsvFilename, append, CsvFileStream),
    write_csv_record_results(CsvFileStream, Results),
    close(CsvFileStream).