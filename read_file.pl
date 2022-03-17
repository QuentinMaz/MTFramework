:- module(read_file, [read_file/2]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  File reader for .pddl files.
%%  It reads the input file character by character and parse it
%%  into a list. Brackets, comma, period and question marks
%%  are treated as separate words. White spaces separe words.
%%
%%  Similar to read_sent in Pereira and Shieber, Prolog and
%%      Natural Language Analysis, CSLI, 1987.
%%
%%  Examples:
%%      :- read_file('input.txt', L).
%%      input.txt > The sky was blue, after the rain.
%%      L = [the, sky, was, blue, ',', after, the, rain, '.']
%%
%%      :- read_file('domain.pddl', L).
%%      domain.pddl >
%%          (define (domain BLOCKS)
%%          (:requirements :strips :typing :action-costs)
%%          (:types block)
%%          (:predicates (on ?x - block ?y - block)
%%          ...
%%      L = ['(', define, '(', domain, blocks, ')', '(', :, requirements|...].
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% read_file(+File, -List).
read_file(File, Words) :-
    see(File), % sets File as the new stream input
    get_code(C), % gets the code of the next character
    read_rest(C, Words),
    seen. % closes the stream and sets the standard input as the new stream input

%% read_rest(+Code, -List).
% unifies [] when input ends
read_rest(-1, []) :- !.

read_rest(C, [Word|Words]) :-
    C = 34,
    !,
    get_code(C1),
    read_json_value(C1, Chars, Next),
    name(Word, Chars),
    read_rest(Next, Words). % keeps track of the last character read (Next)


% ignores spaces, tabs and newlines
read_rest(C, Words) :-
    (C = 32 ; C = 10 ; C = 9 ; C = 13 ; C = 92),
    !,
    get_code(C1),
    read_rest(C1, Words).

% brackets, comma, period or question marks are treated as separed words
read_rest(C, [Char|Words]) :-
    (C = 40 ; C = 41 ; C = 44 ; C = 45 ; C = 46 ; C = 63 ; C = 58 ; C = 91 ; C = 93),
    name(Char, [C]),
    !,
    get_code(C1),
	read_rest(C1, Words).
% reads comments to the end of line
read_rest(59, Words) :-
    get_code(Next),
    !,
    read_comment(Next, Last), % keeps track of the last character read
    read_rest(Last, Words).
% otherwise gets all of the next word
read_rest(C, [Word|Words]) :-
    read_word(C, Chars, Next),
    name(Word, Chars),
    read_rest(Next, Words). % keeps track of the last character read (Next)

%% read_json_value(+Code, -Characters, -NextCode).
% stops if " found
read_json_value(34, [], Next) :-
    get_code(Next),
    !.
% otherwise, gets characters and converts them to lower case
read_json_value(C, [LC|Chars], Last) :-
    lower_case(C, LC),
    get_code(Next),
    read_json_value(Next, Chars, Last).

%% read_comment(+Code, -LastCode).
% keep reading as long you don't find end-of-line or end-of-file
read_comment(10, 10) :- !.
read_comment(-1, -1) :- !.
read_comment(_, Last) :-
    get_code(Next),
	read_comment(Next, Last).

%% read_word(+Code, -Characters, -NextCode).
% spaces, commas, newlines, periods, end-of-file or question marks separate words
read_word(C, [], C) :-
    (C = 32 ; C = 44 ; C = 10 ; C = 9 ; C = 13 ;
    C = 46 ; C = 63 ; C = 40 ; C = 41 ; C = 58 ;
    C = -1 ; C = 93),
    !.
% otherwise, gets characters and converts them to lower case
read_word(C, [LC|Chars], Last) :-
    lower_case(C, LC),
    get_code(Next),
    read_word(Next, Chars, Last).

%% lower_case(+Code, -LowerCaseCode).
% converts to lower case if necessary
lower_case(C, C) :-
    (C < 65 ; C > 90),
    !.
lower_case(C, LC) :-
    LC is C + 32.