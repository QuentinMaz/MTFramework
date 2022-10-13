:- module(pddl_serialiser, [serialise_problem/2, serialise_states/2]).

%% serialise_states(+Filename, +States).
serialise_states(Filename, States) :-
    open(Filename, write, FileStream),
    set_output(FileStream),
    write_states_with_cost(States),
    flush_output(FileStream),
    told.

write_states_with_cost([]).
write_states_with_cost([Cost-State|T]) :-
    format('~d,[~@]\n', [Cost, write_state(State)]),
    write_states_with_cost(T).

write_state([Fact]) :-
    Fact =.. [Name|Variables],
    format('~a~@', [Name, write_list(Variables)]).
write_state([Fact|T]) :-
    Fact =.. [Name|Variables],
    format('~a~@,', [Name, write_list(Variables)]),
    write_state(T).

serialise_problem(Problem, Filename) :-
    open(Filename, write, FileStream),
    set_output(FileStream),
    write_problem(Problem),
    flush_output(FileStream),
    told.

write_problem(problem(Name, Domain, _R, Objects, IS, GS, _C, _MS, _LS)) :-
    format('(define ~@\n~@\n~@\n~@\n~@)\n',
        [
            write_problem_name(Name),
            write_problem_domain(Domain),
            write_problem_objects(Objects),
            write_problem_init(IS),
            write_problem_goal(GS)
        ]
    ).

write_problem_name(Name) :-
    format('(problem ~a)', [Name]).

write_problem_domain(Domain) :-
    format('(:domain ~a)', [Domain]).

write_problem_objects(Objects) :-
    format('(:objects~@)', [write_typed_list(Objects)]).

write_problem_init(IS) :-
    format('(:init~@)', [write_predicates(IS)]).

write_problem_goal(GS) :-
    format('(:goal (and~@))', write_predicates(GS)).

write_predicates([]).
write_predicates([Predicate|T]) :-
    Predicate =.. [Name|Arguments],
    format('\n(~a~@)', [Name, write_list(Arguments)]),
    write_predicates(T).

write_typed_list([]).
write_typed_list([TypedHead|T]) :-
    compound(TypedHead),
    !,
    TypedHead =.. [Type, Object], % type(object)
    format(' ~a - ~a', [Object, Type]),
    write_typed_list(T).
write_typed_list([UntypedHead|T]) :-
    write(' '),
    write(UntypedHead),
    write_list(T).

write_list([]).
write_list([H|T]) :-
    write(' '),
    write(H),
    write_list(T).