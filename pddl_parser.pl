:- module(pddl_parser, [parse_domain/2, parse_problem/2, deserialise_plan/2, deserialise_plans/2]).

:- ensure_loaded(read_file).

% defines operator ?. It is a syntax sugar to mark PDDL variables (example : ?x)
:- op(300, fy, ?).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXPORTED PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% parseDomain(+File, -Output).
% parses PDDL domain File and turns it into re-written Prolog syntax
parse_domain(File, Domain) :-
    parse_domain(File, FreeDomain, []),
    FreeDomain =.. [domain|Data],
    (
        foreach(Term, Data),
        foreach(GroundTerm, GroundData)
    do
        (var(Term) -> GroundTerm = [] ; GroundTerm = Term)
    ),
    Domain =.. [domain|GroundData].

%% parseDomain(+File, -Output, -RestOfFile).
% same as parseDomain/2 but also returns the rest of the file. It can be useful if the domain and problem are in one file
parse_domain(File, Output, R) :-
    read_file(File, List),
    domainBNF(Output, List, R).

%% parse_problem(+File, -Output).
% parses PDDL problem File and turns it into re-written Prolog syntax
parse_problem(File, Problem) :-
    parse_problem(File, FreeProblem, []),
    FreeProblem =.. [problem|Data],
    (
        foreach(Term, Data),
        foreach(GroundTerm, GroundData)
    do
        (var(Term) -> GroundTerm = [] ; GroundTerm = Term)
    ),
    Problem =.. [problem|GroundData].

%% parseProblem(+File, -Output, -RestOfFile).
% same as parseProblem/2 but also returns the rest of the file. It can be useful if the domain and problem are in one file
parse_problem(F, O, R) :-
	read_file(F, L),
	problemBNF(O, L, R),
    !.

%% deserialise_plan(+Filename, -Plan).
deserialise_plan(Filename, Plan) :- deserialise_plan(Filename, Plan, []).

%% deserialise_plan(+Filename, -Plan, -RestOfFile).
deserialise_plan(Filename, Plan, RestOfFile) :-
    catch(read_file(Filename, List), _, List = []),
    ipc_style_plan(Plan, List, RestOfFile).

%% deserialise_plans(+PlansFilenames, -Plans).
deserialise_plans([], []).
deserialise_plans([PlanFilename|T1], [Plan|T2]) :-
    deserialise_plan(PlanFilename, Plan),
    deserialise_plans(T1, T2).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DCG RULES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% List of DCG rules describing structure of domain file in language PDDL.
% BNF description was obtain from http://www.cs.yale.edu/homes/dvm/papers/pddl-bnf.pdf
% This parser does not fully support PDDL 3.0.
domainBNF(domain(N, R, T, C, P, F, C, S))
    --> ['(', 'define', '(', 'domain'], name(N), [')'],
        (require_def(R)	; []),
        (types_def(T) ; []),
        (constants_def(C) ; []),
        (predicates_def(P) ; []),
        (functions_def(F) ; []), % :fluents
        % (constraints(C) ; []), % :constraints
        zero_or_more(structure_def, S),
        [')'].

%% List of DCG rules describing structure of problem file in language PDDL.
% BNF description was obtain from http://www.cs.yale.edu/homes/dvm/papers/pddl-bnf.pdf
% This parser does not fully NOT support PDDL 3.0.
problemBNF(problem(Name, Domain, Requirements, ObjectsDeclaration, I, G, _, MS, LS))
    --> ['(', define, '(', problem, Name, ')',
        '(', ':', domain, Domain, ')'],
        (require_def(Requirements) ; []),
        (object_declaration(ObjectsDeclaration)	; []),
        init(I),
        goal(G),
        % (constraints(C) ; []), % :constraints
        (metric_spec(MS) ; []),
        (length_spec(LS) ; []),
        [')'].

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DOMAIN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

require_def(R) --> ['(', ':', 'requirements'], one_or_more(require_key, R), [')'].
require_key(strips) --> [':', strips].
require_key(typing) --> [':', typing].
% require_key('negative-preconditions') --> [':', 'negative-preconditions'].
% require_key('disjunctive-preconditions') --> [':', 'disjunctive-preconditions'].
require_key(equality) --> [':', 'equality'].
require_key('existential-preconditions') --> [':', 'existential-preconditions'].
require_key('universal-preconditions') --> [':', 'universal-preconditions'].
require_key('quantified-preconditions')	--> [':', 'quantified-preconditions'].
require_key('conditional-effects') --> [':', 'conditional-effects'].
require_key(fluents) --> [':', 'fluents'].
require_key(adl) --> [':', 'adl'].
require_key('durative-actions') --> [':', 'durative-actions'].
require_key('derived-predicates') --> [':', 'derived-predicates'].
require_key('timed-initial-literals') --> [':', 'timed-initial-literals'].
require_key(preferences) --> [':', 'preferences'].
require_key(constraints) --> [':', 'constraints'].

types_def(L) --> ['(', ':', types], typed_list(name, L), [')'].

constants_def(L) --> ['(', ':', constants], zero_or_more(name, L), [')'].

predicates_def(P) --> ['(', ':', predicates], one_or_more(atomic_formula_skeleton, P), [')'].

atomic_formula_skeleton(F) --> ['('], predicate(P), typed_list(variable, L), [')'], {F =.. [P|L]}.

predicate(P) --> name(P).

variable(V) --> ['?'], name(N), {V =.. [?, N]}.

atomic_function_skeleton(f(S, L)) --> ['('], function_symbol(S), typed_list(variable, L), [')'].

function_symbol(S) --> name(S).

functions_def(F) --> ['(', ':', functions], function_typed_list(atomic_function_skeleton, F), [')']. %:fluents

% constraints(C) --> ['(', ':', constraints], con_GD(C), [')']. % :constraints

structure_def(A) --> action_def(A).
% structure_def(D) --> durative_action_def(D). % :durative actions
% structure_def(D) --> derived_def(D). % :derived predicates

typed_list(W, R) --> one_or_more(W, N), ['-'], type(T), !, typed_list(W, Ns), {type_list(T, N, TN), append(TN, Ns, R)}.
typed_list(W, N) --> zero_or_more(W, N).

primitive_type(N) --> name(N).

type(either(PT)) --> ['(', either], !, one_or_more(primitive_type, PT), [')'].
type(PT) --> primitive_type(PT).

function_typed_list(W, [F|Ls]) --> one_or_more(W, L), ['-'], function_type(T), !, function_typed_list(W, Ls), {F =.. [T|L]}. % :typing
function_typed_list(W, L) --> zero_or_more(W, L).

function_type(number) --> [number].

emptyOr(_) --> ['(', ')'].
emptyOr(W) --> W.

%% action_def(-Action).
action_def(action(S, L, Precon, Pos, Neg, Assign))
    --> ['(', ':', action], action_symbol(S),
        [':', parameters, '('], typed_list(variable, L), [')'],
        action_def_body(Precon, Pos, Neg, Assign),
        [')'].

action_symbol(N) --> name(N).

action_def_body(P, Pos, Neg, Assign) --> [':', precondition], pre_GD(P), [':', effect], emptyOr(effect(Pos, Neg, Assign)).
action_def_body([], Pos, Neg, Assign) --> ([':', precondition, '(', ')'] ; []), [':', effect], emptyOr(effect(Pos, Neg, Assign)).

pre_GD([F]) --> atomic_formula(term, F), !.
pre_GD(P) --> pref_GD(P).
pre_GD(P) --> ['(', and], pre_GD(P), [')'].
% pre_GD(forall(L, P)) --> ['(', forall, '('], typed_list(variable, L), [')'], pre_GD(P), [')']. % :universal-preconditions

% pref_GD(preference(N, P))	--> ['(', preference], (pref_name(N) ; []), gd(P), [')']. % :preferences
pref_GD(P) --> gd(P).

pref_name(N) --> name(N).

gd(F) --> atomic_formula(term, F).	% this option is covered by gd(L)
% gd(L) --> literal(term, L). % :negative-preconditions
gd(P) --> ['(', and], zero_or_more(gd, P), [')'].
% gd(or(P)) --> ['(', or], zero_or_more(gd, P), [')']. % :disjuctive-preconditions
% gd(not(P)) --> ['(', not], gd(P), [')']. % :disjuctive-preconditions
% gd(imply(P1, P2)) --> ['(', imply], gd(P1), gd(P2), [')']. % :disjuctive-preconditions
% gd(exists(L, P)) --> ['(', exists, '('], typed_list(variable, L), [')'], gd(P), [')']. % :existential-preconditions
% gd(forall(L, P)) --> ['(', forall, '('], typed_list(variable, L), [')'], gd(P), [')']. % :universal-preconditions
gd(F) --> f_comp(F). % :fluents

f_comp(compare(C, E1, E2)) --> ['('], binary_comp(C), f_exp(E1), f_exp(E2), [')'].

literal(T, F) --> atomic_formula(T, F).
literal(T, not(F)) --> ['(', not], atomic_formula(T, F), [')'].

atomic_formula(_, F) --> ['('], predicate(P), zero_or_more(term, T), [')'], {F =.. [P|T]}.

term(N)	--> name(N).
term(V)	--> variable(V).

f_exp(N) --> number(N).
f_exp(op(O, E1, E2)) --> ['('], binary_op(O), f_exp(E1), f_exp(E2), [')'].
f_exp('-'(E)) --> ['(', '-'], f_exp(E), [')'].
f_exp(H) --> f_head(H).

f_head(F) --> ['('], function_symbol(S), zero_or_more(term, T), [')'], {F =.. [S|T]}.
f_head(S) --> function_symbol(S).

binary_op(O) --> multi_op(O).
binary_op(45) --> [45]. % 45 = minus = '-'
binary_op('/') --> ['/'].

multi_op('*') --> ['*'].
multi_op('+') --> ['+'].

binary_comp('>') --> ['>'].
binary_comp('<') --> ['<'].
binary_comp('=') --> ['='].
binary_comp('>=') --> ['>='].
binary_comp('<=') --> ['<='].

number(N) --> [N], {integer(N)}.
number(N) --> [N], {float(N)}.

effect(P, N, A) --> ['(', and], c_effect(P, N, A), [')'].
effect(P, N, A) --> c_effect(P, N, A).

c_effect(P, N, A) --> p_effect(P, N, A).

p_effect([], [], []) --> [].
p_effect(Ps, Ns, [F|As])
    --> ['('], assign_op(O), f_head(H), f_exp(E), [')'],
        p_effect(Ps, Ns, As), {F =.. [O, H, E]}.
p_effect(Ps, [F|Ns], As) --> ['(', not], atomic_formula(term, F), [')'], p_effect(Ps, Ns, As).
p_effect([F|Ps], Ns, As) --> atomic_formula(term, F), p_effect(Ps, Ns, As).

assign_op(assign) --> [assign].
assign_op(scale_up) --> [scale_up].
assign_op(scale_down) --> [scale_down].
assign_op(increase) --> [increase].
assign_op(decrease) --> [decrease].

%% one_or_more(+Word, -Results, +Input, -Output).
% DCG extension that overcomes the <term>+ operator (cf the BNF description)
% mimics the behavior of the predicate 'C'/3 (which is part of the underlying DCG processing)
one_or_more(Word, [R|Rs], Input, Output) :-
    F =.. [Word, R, Input, TmpOutput], % builds a functor with R and Input as first variables
    F, % calls the predicate
    (one_or_more(Word, Rs, TmpOutput, Output) ; (Rs = [] , Output = TmpOutput)).

%% zero_or_more(+Word, -Rest).
% DCG extension that overcomes the <term>* operator (cf the BNF description)
zero_or_more(W, R) --> one_or_more(W, R).
zero_or_more(_, []) --> [].

%% type_list(+Type, +List, -TypedList).
type_list(_, [], []).
type_list(Type, [Head|T1], [TypedHead|T2]) :-
    TypedHead =.. [Type, Head],
    type_list(Type, T1, T2).

%% name(+Name).
% Name is everything that is not number, bracket or question mark.
% Those rules are not necessary, but rapidly speed up parsing process.
name(N) --> [N], {integer(N), !, fail}.
name(N) --> [N], {float(N), !, fail}.
name(N)	--> [N], {N = ')', !, fail}.
name(N)	--> [N], {N = '(', !, fail}.
name(N)	--> [N], {N = '?', !, fail}.
name(N)	--> [N], {N = '-', !, fail}.
name(N)	--> [N].

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PROBLEM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

object_declaration(L) --> ['(', ':', objects], typed_list(name, L), [')'].

init(I) --> ['(', ':', init], zero_or_more(init_el, I), [')'].

init_el(I) --> literal(name, I).
init_el(set(H, N)) --> ['(', '='], f_head(H), number(N), [')']. % fluents
init_el(at(N, L)) --> ['(', at], number(N), literal(name, L), [')']. % timed-initial literal

goal(G)	--> ['(', ':', goal], pre_GD(G), [')'].

/** for constraints. Not seen yet.
% constraints(C) --> ['(', ':', constraints], pref_con_GD(C), [')']. % constraints

pref_con_GD(and(P))		--> ['(',and], zero_or_more(pref_con_GD, P), [')'].
%pref_con_GD(foral(L, P))	--> ['(',forall,'('], typed_list(variable, L), [')'], pref_con_GD(P), [')'].	%universal-preconditions
%pref_con_GD(prefernce(N, P))	--> ['(',preference], (pref_name(N) ; []), con_GD(P), [')'].			%prefernces
pref_con_GD(P)			--> con_GD(P).

con_GD(and(L))			--> ['(',and], zero_or_more(con_GD, L), [')'].
con_GD(forall(L, P))		--> ['(',forall,'('], typed_list(variable, L),[')'], con_GD(P), [')'].
con_GD(at_end(P))		--> ['(',at,end],	gd(P), [')'].
con_GD(always(P))		--> ['(',always],	gd(P), [')'].
con_GD(sometime(P))		--> ['(',sometime],	gd(P), [')'].
con_GD(within(N, P))		--> ['(',within], number(N), gd(P), [')'].

con_GD(at_most_once(P))		--> ['(','at-most-once'], gd(P),[')'].
con_GD(some_time_after(P1, P2))	--> ['(','sometime-after'], gd(P1), gd(P2), [')'].
con_GD(some_time_before(P1, P2))--> ['(','sometime-before'], gd(P1), gd(P2), [')'].
con_GD(always_within(N, P1, P2))--> ['(','always-within'], number(N), gd(P1), gd(P2), [')'].
con_GD(hold_during(N1, N2, P))	--> ['(','hold-during'], number(N1), number(N2), gd(P), [')'].
con_GD(hold_after(N, P))	--> ['(','hold-after'], number(N), gd(P),[')'].
*/

metric_spec(metric(O, E)) --> ['(', ':', metric], optimization(O), metric_f_exp(E), [')'].

optimization(minimize) --> [minimize].
optimization(maximize) --> [maximize].

metric_f_exp(E) --> ['('], binary_op(O), metric_f_exp(E1), metric_f_exp(E2), [')'], {E =..[O, E1, E2]}.
metric_f_exp(multi_op(O,[E1|E])) --> ['('], multi_op(O), metric_f_exp(E1), one_or_more(metric_f_exp, E), [')']. % I dont see meanful of this rule, in additional is missing in f-exp
metric_f_exp(E) --> ['(','-'], metric_f_exp(E1), [')'], {E=.. [-, E1]}.
metric_f_exp(N)	--> number(N).
metric_f_exp(F)	--> ['('], function_symbol(S), zero_or_more(name, Ns), [')'], { F=..[S|Ns]}. % concat_atom([S|Ns], '-', F).
metric_f_exp(function(S)) --> function_symbol(S).
metric_f_exp(total_time) --> ['total-time'].
metric_f_exp(is_violated(N)) --> ['(', 'is-violated'], pref_name(N), [')'].

% Work arround
length_spec([])	--> [not_defined]. % there is no definition ???

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLAN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ipc_style_plan(Plan) --> zero_or_more(action, Plan).
action(Action) --> ['('], action_name(N), zero_or_more(action_parameter, P), [')'], {Action =.. [N|P]}.
action_name(Name) --> name(Name).
action_parameter(Parameter) --> name(Parameter).