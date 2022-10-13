:- module(utils, [predicates_names/2, filter_predicates/3, filter_predicates_with_names/3]).

% retrieves all the names of the terms in Predicates (/!\ may have duplicates)
%% predicates_names(+Predicates, -PredicatesNames).
predicates_names([], []).
predicates_names([H|T1], [Name|T2]) :-
    H =.. [Name|_],
    predicates_names(T1, T2).

% accumulates the predicates whose name matches Name
%% filter_predicates(+Pred, +Name, -Res).
filter_predicates([], _, []).
filter_predicates([H|T1], Name, [H|T2]) :-
    H =.. [Name|_],
    !,
    filter_predicates(T1, Name, T2).
filter_predicates([_|T1], Name, Results) :-
    filter_predicates(T1, Name, Results).

% accumulates the result of filter_predicates/3 in Res (/!\ not not flattened)
%% filter_predicates_with_names(-Pred, +Names, -Res).
filter_predicates_with_names(_, [], []).
filter_predicates_with_names(Predicates, [Name|T1], [MatchedPredicates|T2]) :-
    filter_predicates(Predicates, Name, MP),
    sort(MP, MatchedPredicates),
    filter_predicates_with_names(Predicates, T1, T2).