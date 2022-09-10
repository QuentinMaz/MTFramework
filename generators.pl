:- module(generators,
    [
        generator1/1, generator2/1, generator3/1, generator4/1, generator5/1,

        bfs_from_initial_state/2, progress_from_result_path/2, regress_from_goal_state/2, regress_from_result_path/2,
        progress_from_state_with_plan/3, progress_randomly_from_initial_state/2, write_generators/0,

        validate_plan/1
    ]).

:- use_module(library(ordsets), [ord_subtract/3, ord_union/3, ord_union/2, ord_subset/2, ord_add_element/3, ord_intersect/2, ord_disjoint/2]).
:- use_module(library(queues), [queue_cons/3, list_queue/2, queue_append/3, queue_memberchk/2, empty_queue/1]).
:- use_module(library(sets), [is_set/1, list_to_set/2]).
:- use_module(library(lists), [selectchk/3, reverse/2, maplist/3]).
:- use_module(library(random), [random_member/2, maybe/1]).
:- use_module(library(plunit)).

:- ensure_loaded(domain).
:- ensure_loaded(blackboard_data).

write_generators :-
    Predicates =
    [
        'bfs_from_initial_state(_MaxDepth)', 'progress_from_result_path(_MaxActionIndex)',
        'regress_from_goal_state(_Depth)', 'regress_from_result_path(_MaxActionIndex)',
        'progress_randomly_from_initial_state(_MaxDepth)'
    ],
    format('\ngenerators available :\n~@\n', [write_list(Predicates)]).

write_list([]).
write_list([H|T]) :-
    write(H),
    nl,
    write_list(T).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPER PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% regress(+State, +NumberOfActions, -NextState).
% regresses from a state to another one until the remaining number of actions is zero.
regress(ResultState, 0, ResultState) :-
    !.
regress(State, NumberOfActions, ResultState) :-
    regress(State, NewState),
    NewNumberOfActions is NumberOfActions - 1,
    regress(NewState, NewNumberOfActions, ResultState).

%% regress(+State, -PreviousState).
regress(State, PreviousState) :-
    generate_action(Action),
    action_parameters(Action, Parameters),
    instantiate_parameters(Parameters),
    is_set(Parameters),
    % gets its effects
    action_positive_effects(Action, TmpPE),
    sort(TmpPE, PE),
    action_negative_effects(Action, TmpNE),
    sort(TmpNE, NE),
    %% relevance checking of the given instancied action A
    % checks that the intersection between PE and G is not empty
    % (it means that the given action makes at least one the goal's literals true)
    ord_intersect(State, PE),
    % checks that the intersection between NE and G is empty
    % (it means that the given action does not make any of goal's literals false)
    ord_disjoint(State, NE),
    % retrieves the preconditions of the action once checking is done
    action_preconditions(Action, TmpP),
    sort(TmpP, Preconditions),
    % PreviousState is made of State (from which we subtract the own effects of A) with the preconditions of A
    ord_subtract(State, PE, TmpState),
    ord_union(TmpState, Preconditions, PreviousState).

%% regress_from_state_with_plan(+StartState, +Plan, -Path).
regress_from_state_with_plan(FinalState, [], [FinalState]).
regress_from_state_with_plan(State, [ActionDef|T1], [State|T2]) :-
    regress_with_action_definition(State, ActionDef, PreviousState),
    regress_from_state_with_plan(PreviousState, T1, T2).

%% regress_with_action_definition(+State, +ActionDefinition, -PreviousState).
regress_with_action_definition(State, ActionDefinition, PreviousState) :-
    generate_action(Action),
    action_definition(Action, ActionDefinition),
    action_positive_effects(Action, TmpPE),
    sort(TmpPE, PE),
    action_preconditions(Action, TmpP),
    sort(TmpP, Preconditions),
    ord_subtract(State, PE, TmpState),
    ord_union(TmpState, Preconditions, PreviousState).

regress_with_goal_filter(State, GoalState, PreviousState) :-
    regress(State, PreviousState),
    \+ ord_subset(GoalState, PreviousState).

validate_goal_states(States, Cost, Nodes) :-
    get_rigid_relation_names(RigidRelationNames),
    get_rigid_relations(RigidRelations),
    get_static_fluent_names(StaticFluentNames),
    get_static_fluent_numbers(NumberForEachStaticFluent),
    (
        foreach(State, States),
        fromto([], In, Out, Nodes),
        param(Cost, RigidRelationNames, RigidRelations, StaticFluentNames, NumberForEachStaticFluent)
    do
        (   (check_rigid_relations(State, RigidRelationNames, RigidRelations), check_static_fluents(State, StaticFluentNames, NumberForEachStaticFluent)) -> Out = [node(State, Cost, _)|In]
        ;   Out = In
        )
    ).

%% progress(+State, -NextState).
% progresses from a state to another with a possible action
progress(State, NextState) :-
    % generates a possible action
    generate_action(Action),
    action_parameters(Action, Parameters),
    instantiate_parameters(Parameters),
    is_set(Parameters),
    action_preconditions(Action, TmpPreconditions),
    sort(TmpPreconditions, Preconditions),
    ord_subset(Preconditions, State),
    % applies the action
    action_positive_effects(Action, TmpPE),
    sort(TmpPE, PE),
    action_negative_effects(Action, TmpNE),
    sort(TmpNE, NE),
    ord_subtract(State, NE, TmpState),
    ord_union(TmpState, PE, NextState).

%% progress(+State, +NumberOfActions, -NextState).
% progresses from a state to another one until the remaining number of actions is zero.
progress(ResultState, 0, ResultState) :-
    !.
progress(State, NumberOfActions, ResultState) :-
    progress(State, NewState),
    NewNumberOfActions is NumberOfActions - 1,
    progress(NewState, NewNumberOfActions, ResultState).

%% progress_from_state_with_plan(+StartState, +Plan, -Path).
progress_from_state_with_plan(FinalState, [], [FinalState]).
progress_from_state_with_plan(State, [ActionDef|T1], [State|T2]) :-
    progress_with_action_definition(State, ActionDef, NextState),
    progress_from_state_with_plan(NextState, T1, T2).

%% progress_with_action_definition(+State, +ActionDefinition, -NextState).
progress_with_action_definition(State, ActionDefinition, NextState) :-
    generate_action(Action),
    action_definition(Action, ActionDefinition),
    action_preconditions(Action, TmpPreconditions),
    sort(TmpPreconditions, Preconditions),
    ord_subset(Preconditions, State),
    % applies the action
    action_positive_effects(Action, TmpPE),
    sort(TmpPE, PE),
    action_negative_effects(Action, TmpNE),
    sort(TmpNE, NE),
    ord_subtract(State, NE, TmpState),
    ord_union(TmpState, PE, NextState).

validate_plan(Plan) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    problem_goal_state(Problem, GoalState),
    progress_from_state_with_plan(InitialState, Plan, SourceStates),
    append(_, [FinalState], SourceStates),
    !,
    ord_subset(GoalState, FinalState),
    set_final_state(FinalState).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GENERATORS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GENERATOR#0
%% name : bfs_from_initial_state
%% description : it explores the states space from the initial state
%% of the problem with a BFS until the given maximum depth is achieved.
%% The cost is of each node is thus the length of the shortest path
%% to the associated state. Generator used to test mr0.
%% parameter :
%%      - MaxDepth : integer that defines the depth of the BFS exploration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bfs_from_initial_state(MaxDepth, Results) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    StartNode = node(InitialState, 0, _),
    list_queue([StartNode], Queue),
    bfs(Queue, MaxDepth, [], Nodes),
    % removes the start node (as it is the initial state...)
    selectchk(StartNode, Nodes, Results).

%% bfs(+Queue, +MaxDepth, +VisitedNodes, -Results).
% stops when empty queue
bfs(Queue, _MaxDepth, VisitedNodes, VisitedNodes) :-
    empty_queue(Queue),
    !.
bfs(_Queue, _MaxDepth, VisitedNodes, VisitedNodes) :-
    \+ keep_generating,
    !.
bfs(Queue, MaxDepth, VisitedNodes, Results) :-
    queue_cons(Node, RQ, Queue),
    (bagof(Child, expand_node(Node, Queue, VisitedNodes, MaxDepth, Child), Children) ; Children = []),
    !,
    % duplicates may appear in children /!\
    list_to_set(Children, DistinctChildren),
    % appends to the queue the children
    queue_append(RQ, DistinctChildren, NewQueue),
    % adds the current node in the list of the visited ones
    ord_add_element(VisitedNodes, Node, NewVisitedNodes),
    length(NewVisitedNodes, NumberOfNodes),
    set_nb_nodes(NumberOfNodes),
    bfs(NewQueue, MaxDepth, NewVisitedNodes, Results).

%% expand_node(+Node, +Queue, +VisitedNodes, +MaxDepth, -NextNode).
expand_node(node(State, Depth, _), Queue, VisitedNodes, MaxDepth, node(NextState, NextDepth, _)) :-
    Depth < MaxDepth,
    progress(State, NextState),
    % checks that we have never visited NextState nor have already planned to visit it
    \+ memberchk(node(NextState, _, _), VisitedNodes),
    \+ queue_memberchk(node(NextState, _, _), Queue),
    % child node instanciation
    NextDepth is Depth + 1.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GENERATOR#1
%% name : progress_from_result_path
%% description : it explores the states space from the (beginning of the) result path. For each state of the path, it progresses
%% of one possible action and gathers the reachable states. Generator used to test mr0.
%% parameter :
%%      - MaxActionIndex : integer that defines the maximum index to iterate to from the initial state (in case the source result is too long)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% progress_from_result_path(-MaxActionIndex, -Results).
progress_from_result_path(MaxActionIndex, Results) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    get_source_result(SourceResult),
    progress_from_state_with_plan(InitialState, SourceResult, SourceStates),
    length(SourceStates, PathLength),
    (PathLength - 3 >= MaxActionIndex -> Maximum is MaxActionIndex ; Maximum is PathLength - 3),
    progress_and_gather_from_path(SourceStates, 1, Maximum, [], Results).

progress_and_gather_from_path(_, Index, Maximum, Results, Results) :-
    Index > Maximum,
    !.
progress_and_gather_from_path(_, _, _, Results, Results) :-
    \+ keep_generating,
    !.
progress_and_gather_from_path([State|T], Cost, Maximum, Nodes, Results) :-
    get_problem(P),
    problem_initial_state(P, IS),
    setof(node(NextState, Cost, _), progress_with_state_filtering(State, IS, NextState), NextNodes),
    ord_union(Nodes, NextNodes, TmpNewNodes),
    list_to_set(TmpNewNodes, NewNodes),
    length(NewNodes, NumberOfNodes),
    set_nb_nodes(NumberOfNodes),
    NewCost is Cost + 1,
    progress_and_gather_from_path(T, NewCost, Maximum, NewNodes, Results).

progress_with_state_filtering(State, FilterState, NextState) :-
    progress(State, NextState),
    NextState \= FilterState.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GENERATOR#2
%% name : regress_from_goal_state
%% description : it explores the states space from the goal state using regression.
%% The cost of each state is thus the number of actions performed. Generator used to test mr1.
%% Note that the current implementation is not optimized (the regression (re)starts from the goal state at each stage).
%% parameter :
%%      - Depth : integer that defines the depth of the exploration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

regress_from_goal_state(Depth, Results) :-
    get_problem(Problem),
    problem_goal_state(Problem, GoalState),
    regress_and_gather_from_states([GoalState], [], 1, Depth, Results).

regress_and_gather_from_states([], _, _, _, []).
regress_and_gather_from_states(_, _, Depth, MaximumDepth, []) :-
    Depth > MaximumDepth.
regress_and_gather_from_states(_, _, _, _, []) :-
    \+ keep_generating.
regress_and_gather_from_states([State|T], VisitedStates, Depth, MaximumDepth, Results) :-
    % regresses from one action
    get_problem(Problem),
    problem_goal_state(Problem, GoalState),
    (bagof(S, (regress_with_goal_filter(State, GoalState, S), \+ memberchk(S, VisitedStates)), PreviousStates) ; PreviousStates = []),
    % length(PreviousStates, PS), format('around me : ~d\n', [PS]),
    % sets all the states reached as visited
    append([State|PreviousStates], VisitedStates, NewVisitedStates),
    % length(NewVisitedStates, VS), format('now a total of visited states : ~d\n', [VS]),
    % gets proper and valid nodes from the regression step
    validate_goal_states(PreviousStates, Depth, Nodes),
    % updates the number of nodes to gather
    length(Nodes, N),
    update_nb_nodes(N),
    % adds the relevant states to the list
    maplist(arg(1), Nodes, StatesFromNodes),
    append(T, StatesFromNodes, NewStates),
    NewDepth is Depth + 1,
    % format('got : ~d\n', [N]),
    regress_and_gather_from_states(NewStates, NewVisitedStates, NewDepth, MaximumDepth, NewResults),
    !,
    append(Nodes, NewResults, Results).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GENERATOR#3
%% name : regress_from_result_path
%% description : it explores the states space from the (end of the) result path. For each state of the path, it regresses
%% of one possible action and gathers the reachable states. Generator used to test mr1.
%% parameter :
%%      - MaxActionIndex : integer that defines the maximum index to iterate to from the goal state (in case the source result is too long)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

regress_from_result_path(MaxActionIndex, Results) :-
    get_problem(Problem),
    problem_goal_state(Problem, GoalState),
    get_source_result(SourceResult),
    reverse(SourceResult, ReversedSourceResult),
    regress_from_state_with_plan(GoalState, ReversedSourceResult, SourceStates),
    length(SourceStates, PathLength),
    (PathLength - 3 >= MaxActionIndex -> Maximum is MaxActionIndex ; Maximum is PathLength - 3),
    regress_and_gather_from_path(SourceStates, [], 1, Maximum, Results).

regress_and_gather_from_path(_, _, Index, Maximum, []) :-
    Index > Maximum.
regress_and_gather_from_path(_, _, _, _, []) :-
    \+ keep_generating.
regress_and_gather_from_path([State|T], VisitedStates, Cost, Maximum, Results) :-
    get_problem(Problem),
    problem_goal_state(Problem, GoalState),
    (bagof(S, (regress_with_goal_filter(State, GoalState, S), \+ memberchk(S, VisitedStates)), PreviousStates) ; PreviousStates = []),
    append([State|PreviousStates], VisitedStates, NewVisitedStates),
    % filters the states reached by checking their validity as a new planning problem (since MR1 is expected to be used)
    validate_goal_states(PreviousStates, Cost, Nodes),
    length(Nodes, N),
    update_nb_nodes(N),
    NewCost is Cost + 1,
    regress_and_gather_from_path(T, NewVisitedStates, NewCost, Maximum, NewResults),
    !,
    append(Nodes, NewResults, Results).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GENERATOR#4
%% name : progress_randomly_from_initial_state
%% description : it is said to be the implementation of a "baseline" generator.
%% Intuitively, it should randomly select states of the search space.
%% The current implementation uses the length of the source result
%% in order to progress randomly over a random but reasonable amount of steps.
%% The states are gathered from the path of each progression, and the cost of each state
%% is its position in the path. If the progression fails, the algorithm repeats itself
%% with either the same depth (which is used to set the targeted length of the progression)
%% or the current depth - 1. So, the search necessarily finishes (as its depth converges to zero).
%% parameter :
%%      - MaximumDepth : integer that defines the maximum size of the randomly-driven pathes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% progress_randomly_from_initial_state(+MaximumDepth, -Results).
progress_randomly_from_initial_state(MaximumDepth, Nodes) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    generate_with_depth(InitialState, MaximumDepth, [InitialState], Nodes),
    !.

%% generate_with_depth(+StartState, +MaximumDepth, +VisitedStates, -Results).
generate_with_depth(_, 0, _, []) :-
    !. % prevents from backtracking when Depth is 0
generate_with_depth(_, _, _, []) :-
    \+ keep_generating,
    !. % prevents from backtracking when enough nodes have been generated
generate_with_depth(StartState, Depth, VisitedStates, Results) :-
    % tries to progress randomly (length of the path is Depth)
    progress_and_gather_randomly(StartState, Depth, [StartState|Path]),
    gather_path(Path, 1, VisitedStates, NewVisitedStates, Nodes),
    Nodes \== [],
    length(Nodes, N), % format('got ~d nodes from path (for depth : ~d)\n', [N, Depth]),
    update_nb_nodes(N),
    !,
    generate_with_depth(StartState, Depth, NewVisitedStates, NewNodes),
    append(Nodes, NewNodes, Results).
generate_with_depth(StartState, Depth, VisitedStates, Results) :-
    (   maybe(0.9) -> NewDepth = Depth % (format('retry with depth ~d\n', [Depth]), NewDepth = Depth)
    ;   NewDepth is Depth - 1 % (NewDepth is Depth - 1, format('surrender and continue with depth ~d\n', [NewDepth]))
    ),
   generate_with_depth(StartState, NewDepth, VisitedStates, Results).

%% gather_path(+Path, +IndexOrCost, +VisitedStates, -UpdatedVisitedStates, -Nodes).
gather_path([], _, VisitedStates, VisitedStates, []).
gather_path([State|T], Index, VisitedStates, NewVisitedStates, GatheredNodes) :-
    memberchk(State, VisitedStates),
    NewIndex is Index + 1,
    gather_path(T, NewIndex, VisitedStates, NewVisitedStates, GatheredNodes).
gather_path([State|T1], Index, VisitedStates, NewVisitedStates, [node(State, Index, _)|T2]) :-
    \+ memberchk(State, VisitedStates),
    NewIndex is Index + 1,
    gather_path(T1, NewIndex, [State|VisitedStates], NewVisitedStates, T2).

%% progress_and_gather_randomly(+StartState, +NumberOfActions, -Path).
progress_and_gather_randomly(_, 0, []).
progress_and_gather_randomly(State, NumberOfActions, [State|T]) :-
    NumberOfActions > 0,
    bagof(NextState, progress(State, NextState), NextStates),
    random_member(NewState, NextStates),
    NewNumberOfActions is NumberOfActions - 1,
    !,
    progress_and_gather_randomly(NewState, NewNumberOfActions, T).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GENERATORS FOR EXPERIMENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

generator1(Results) :-
    get_source_result(SourceResult),
    length(SourceResult, SourceResultLength),
    MaximumDepth is SourceResultLength // 2 + 1,
    format('generator1 launched with maximum depth of ~d\n', [MaximumDepth]),
    bfs_from_initial_state(MaximumDepth, Results).

generator2(Results) :-
    get_source_result(SourceResult),
    length(SourceResult, SourceResultLength),
    MaximumDepth is SourceResultLength // 2 + 1,
    progress_from_result_path(MaximumDepth, Results),
    !.
generator2([]) :-
    write('generator 2 : source result not valid\n').

generator3(Results) :-
    get_source_result(SourceResult),
    length(SourceResult, SourceResultLength),
    MaximumDepth is SourceResultLength // 2 + 1,
    regress_from_goal_state(MaximumDepth, Results).

generator4(Results) :-
    get_source_result(SourceResult),
    length(SourceResult, SourceResultLength),
    MaximumDepth is SourceResultLength // 2 + 1,
    regress_from_result_path(MaximumDepth, Results),
    !.
generator4([]) :-
    write('generator4 : source result not valid\n').

generator5(Results) :-
    get_source_result(SourceResult),
    length(SourceResult, SourceResultLength),
    % MaximumDepth is SourceResultLength // 2 + 1,
    MaximumDepth is SourceResultLength,
    progress_randomly_from_initial_state(MaximumDepth, Results).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLUNIT TESTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- ensure_loaded(pddl_parser).
:- ensure_loaded(pddl_serialiser).
:- ensure_loaded(problem).

%% setup_input(+DomainFilename, +ProblemFilename).
setup_input(DomainFilename, ProblemFilename) :-
    parse_domain(DomainFilename, Domain),
    parse_problem(ProblemFilename, TmpProblem),
    sort_problem(TmpProblem, Problem),
    set_blackboard(_Configuration, Domain, Problem),
    set_nb_tests(15),
    set_nb_nodes(0).

%% check_if_goal_state_included(+Nodes).
%% Helper predicate used for regression generators. Indeed, when regressing, a state is "equal" to another one
%% if it is entirely included by the other one.
check_if_goal_state_included(Nodes) :-
    get_problem(Problem),
    problem_goal_state(Problem, GoalState),
    (
        foreach(node(State, _, _), Nodes),
        fromto(0, In, Out, BadNodes),
        param(GoalState)
    do
        (ord_subset(GoalState, State) -> Out is In + 1 ; Out = In)
    ),
    BadNodes == 0.

%% input_with_source_result(+DomainFilename, +ProblemFilename, +SourceResult).
input_with_source_result('test/blocks/domain.pddl', 'test/blocks/blocks1.pddl', ['pick-up'(b),stack(b,a),'pick-up'(c),stack(c,b)]).
input_with_source_result('test/blocks/domain.pddl', 'test/blocks/blocks2.pddl', ['pick-up'(b),stack(b,a),'pick-up'(c),stack(c,b),'pick-up'(d),stack(d,c)]).
input_with_source_result('test/blocks/domain.pddl', 'test/blocks/blocks2.pddl', ['pick-up'(c),stack(c,b),'pick-up'(a),stack(a,d),unstack(c,b),'put-down'(c),'pick-up'(b),stack(b,c),unstack(a,d),'put-down'(a),'pick-up'(d),stack(d,a),unstack(b,c),'put-down'(b),unstack(d,a),stack(d,c),'pick-up'(a),stack(a,b),unstack(d,c),'put-down'(d),'pick-up'(c),stack(c,d),unstack(a,b),'put-down'(a),'pick-up'(b),stack(b,a),unstack(c,d),stack(c,b),'pick-up'(d),stack(d,c)]).
input_with_source_result('test/hanoi/domain.pddl', 'test/hanoi/hanoi2.pddl', [move(d1,d2,peg3),move(d2,d3,peg2),move(d1,peg3,d2),move(d3,peg1,peg3),move(d1,d2,peg1),move(d2,peg2,d3),move(d1,peg1,d2)]).
input_with_source_result('test/hanoi/domain.pddl', 'test/hanoi/hanoi2.pddl', [move(d1,d2,peg3),move(d1,peg3,peg2),move(d2,d3,peg3),move(d1,peg2,d2),move(d3,peg1,peg2),move(d1,d2,d3),move(d2,peg3,peg1),move(d1,d3,d2),move(d3,peg2,peg3),move(d1,d2,d3),move(d2,peg1,peg2),move(d1,d3,d2),move(d1,d2,peg1),move(d2,peg2,d3),move(d1,peg1,d2)]).
input_with_source_result('test/gripper/domain.pddl', 'test/gripper/gripper1.pddl', [pick(ball2, rooma, right), move(rooma, roomb), drop(ball2, roomb, right), move(roomb, rooma), pick(ball1, rooma, right), move(rooma, roomb), drop(ball1, roomb, right)]).
input_with_source_result('test/monkey/domain.pddl', 'test/monkey/monkey2.pddl', [goto(l1,l6),pickglass(l6),goto(l6,l4),getknife(l4),goto(l4,l2),pushbox(l2,l5),climb(l5),getwater(l5),descend(l5),pushbox(l5,l3),climb(l3),grabbananas(l3)]).
input_with_source_result('test/monkey/domain.pddl', 'test/monkey/monkey2.pddl', [goto(l1,l6),goto(l6,l5),goto(l5,l4),goto(l4,l3),goto(l3,l2),pushbox(l2,l6),goto(l6,l5),goto(l5,l4),getknife(l4),goto(l4,l6),pickglass(l6),pushbox(l6,l5),climb(l5),getwater(l5),descend(l5),pushbox(l5,l4),pushbox(l4,l6),pushbox(l6,l3),climb(l3),grabbananas(l3)]).
input_with_source_result('test/typed_gripper/domain.pddl', 'test/typed_gripper/typed_gripper2.pddl', [pick(ball1, rooma, left), move(rooma, roomb), drop(ball1, roomb, left), move(roomb, rooma), pick(ball2, rooma, left), move(rooma, roomb), drop(ball2, roomb, left), move(roomb, rooma), pick(ball3, rooma, left), move(rooma, roomb), drop(ball3, roomb, left), move(roomb, rooma), pick(ball4, rooma, left), move(rooma, roomb), drop(ball4, roomb, left)]).
input_with_source_result('test/travel/domain.pddl', 'test/travel/problem2.pddl', [drive(nj,wv,pe,'car-0'),walk(wv,pe),walk(pe,wv),walk(wv,ky),walk(ky,wv),walk(wv,pe),walk(pe,nj),walk(nj,pe),walk(pe,wv),walk(wv,ky),walk(ky,tn),walk(tn,ky),'fly-red'(ky,og,'plane-0'),walk(og,ca),'fly-blue'(ca,tx,'plane-1')]).
input_with_source_result('test/travel/domain.pddl', 'test/travel/problem5.pddl', ['fly-red'(nv,ky,'plane-0'),'fly-red'(ky,id,'plane-1'),'fly-red'(id,fl,'plane-3'),'fly-red'(fl,nj,'plane-2')]).
input_with_source_result('test/travel/domain.pddl', 'test/travel/problem6.pddl', ['fly-blue'(nv,wa,'plane-8'),'fly-blue'(wa,ms,'plane-7'),'fly-blue'(ms,mi,'plane-6'),'fly-blue'(mi,wa,'plane-5'),'fly-blue'(wa,hi,'plane-4'),walk(hi,ak),'fly-red'(ak,nj,'plane-3'),'fly-red'(nj,ky,'plane-2'),'fly-red'(ky,id,'plane-1')]).
input_with_source_result('test/travel/domain.pddl', 'test/travel/problem7.pddl', [drive(sd,wn,mn,'car-3'),walk(wn,mi),walk(mi,wn),drive(wn,il,ia,'car-2'),drive(il,wn,ia,'car-1'),drive(wn,ia,mn,'car-0'),walk(ia,mn),walk(mn,sd),walk(sd,nd),walk(nd,mo),walk(mo,id),walk(id,wa),'fly-blue'(wa,hi,'plane-1'),'fly-blue'(hi,wa,'plane-3'),'fly-blue'(wa,ok,'plane-2'),walk(ok,tx),walk(tx,ok),'fly-blue'(ok,fl,'plane-0')]).
input_with_source_result('test/travel/domain.pddl', 'test/travel/problem7.pddl', [drive(sd,wn,mn,'car-3'),drive(wn,sd,mn,'car-2'),drive(sd,wn,mn,'car-1'),drive(wn,sd,mn,'car-0'),walk(sd,nd),walk(nd,sd),walk(sd,mo),walk(mo,sd),walk(sd,nd),walk(nd,mn),walk(mn,wn),walk(wn,mi),walk(mi,wn),walk(wn,mn),walk(mn,sd),walk(sd,nd),walk(nd,mo),walk(mo,id),walk(id,wa),'fly-blue'(wa,ok,'plane-3'),'fly-blue'(ok,fl,'plane-2')]).
input_with_source_result('test/simple_rover/domain.pddl', 'test/simple_rover/simple_rover1.pddl', [sample(soil,alpha),drive(alpha,gamma),commun(soil),sample(image,gamma),drive(gamma,beta),commun(image),sample(rock,beta),commun(rock)]).
input_with_source_result('test/simple_rover/domain.pddl', 'test/simple_rover/simple_rover1.pddl', [drive(alpha,gamma),drive(gamma,beta),sample(rock,beta),commun(rock),drive(beta,gamma),drive(gamma,alpha),sample(soil,alpha),commun(soil),drive(alpha,gamma),sample(image,gamma),commun(image)]).
input_with_source_result('test/gripper/domain.pddl', 'test/gripper/gripper2.pddl', [pick(ball4,rooma,right),pick(ball3,rooma,left),move(rooma,roomb),drop(ball4,roomb,right),drop(ball3,roomb,left),move(roomb,rooma),pick(ball2,rooma,right),pick(ball1,rooma,left),move(rooma,roomb),drop(ball2,roomb,right),drop(ball1,roomb,left)]).
input_with_source_result('test/gripper/domain.pddl', 'test/gripper/gripper2.pddl', [pick(ball4,rooma,right),move(rooma,roomb),drop(ball4,roomb,right),move(roomb,rooma),pick(ball3,rooma,right),move(rooma,roomb),drop(ball3,roomb,right),move(roomb,rooma),pick(ball2,rooma,right),move(rooma,roomb),drop(ball2,roomb,right),move(roomb,rooma),pick(ball1,rooma,right),move(rooma,roomb),drop(ball1,roomb,right)]).
% input_with_source_result('test/grid/domain.pddl', 'test/grid/grid1.pddl', [move('node2-4','node1-4'),move('node1-4','node0-4'),move('node0-4','node0-3'),move('node0-3','node0-2'),pickup('node0-2',key3),move('node0-2','node1-2'),move('node1-2','node1-3'),unlock('node1-3','node2-3',key3,square),putdown('node1-3',key3),move('node1-3','node2-3'),pickup('node2-3',key0),move('node2-3','node1-3'),move('node1-3','node1-2'),move('node1-2','node1-1'),putdown('node1-1',key0)]).

:- begin_tests(progression_based_generators).

progression_based_generator(bfs_from_initial_state, [50]).
progression_based_generator(progress_from_result_path, [50]).
progression_based_generator(progress_randomly_from_initial_state, [50]).

test(non_empty_sets, [nondet, forall((progression_based_generator(GeneratorName, Parameters), input_with_source_result(DomainFilename, ProblemFilename, SourceResult)))]) :-
    setup_input(DomainFilename, ProblemFilename),
    set_source_result(SourceResult),
    append(Parameters, [Nodes], Arguments),
    Generator =.. [GeneratorName|Arguments],
    Generator,
    !,
    length(Nodes, N),
    N \== 0,
    is_set(Nodes).

test(non_initial_state_inclusion, [nondet, forall((progression_based_generator(GeneratorName, Parameters), input_with_source_result(DomainFilename, ProblemFilename, SourceResult)))]) :-
    setup_input(DomainFilename, ProblemFilename),
    set_source_result(SourceResult),
    append(Parameters, [Nodes], Arguments),
    Generator =.. [GeneratorName|Arguments],
    Generator,
    !,
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    \+ memberchk(node(InitialState, _, _), Nodes).

:- end_tests(progression_based_generators).

:- begin_tests(regression_based_generators).

regression_based_generator(regress_from_goal_state, [50]).
regression_based_generator(regress_from_result_path, [50]).

test(non_empty_sets, [nondet, forall((regression_based_generator(GeneratorName, Parameters), input_with_source_result(DomainFilename, ProblemFilename, SourceResult)))]) :-
    setup_input(DomainFilename, ProblemFilename),
    set_source_result(SourceResult),
    append(Parameters, [Nodes], Arguments),
    Generator =.. [GeneratorName|Arguments],
    Generator,
    !,
    length(Nodes, N),
    N \== 0,
    is_set(Nodes).

test(non_goal_state_inclusion, [nondet, forall((regression_based_generator(GeneratorName, Parameters), input_with_source_result(DomainFilename, ProblemFilename, SourceResult)))]) :-
    setup_input(DomainFilename, ProblemFilename),
    set_source_result(SourceResult),
    append(Parameters, [Nodes], Arguments),
    Generator =.. [GeneratorName|Arguments],
    Generator,
    !,
    check_if_goal_state_included(Nodes).

test(state_validity, [nondet, forall((regression_based_generator(GeneratorName, Parameters), input_with_source_result(DomainFilename, ProblemFilename, SourceResult)))]) :-
    setup_input(DomainFilename, ProblemFilename),
    set_source_result(SourceResult),
    append(Parameters, [Nodes], Arguments),
    Generator =.. [GeneratorName|Arguments],
    Generator,
    !,
    get_static_fluent_numbers(GSFN),
    get_static_fluent_names(SFN),
    get_rigid_relation_names(RRN),
    get_rigid_relations(RR),
    (
        foreach(node(State, _, _), Nodes),
        fromto([], In, Out, ValidStates),
        param(RR, RRN, SFN, GSFN)
    do
        (   (check_rigid_relations(State, RRN, RR), check_static_fluents(State, SFN, GSFN)) -> Out = [State|In]
        ;   Out = In
        )
    ),
    length(Nodes, NL),
    length(ValidStates, VSL),
    NL == VSL.

:- end_tests(regression_based_generators).

:- begin_tests(generators).

generator(generator1).
generator(generator2).
generator(generator3).
generator(generator4).
generator(generator5).

test(all_in_one_test, [nondet, forall((generator(GeneratorName), input_with_source_result(DomainFilename, ProblemFilename, SourceResult)))]) :-
    setup_input(DomainFilename, ProblemFilename),
    set_source_result(SourceResult),
    Generator =.. [GeneratorName, Nodes],
    Generator,
    !,
    %%%%%%%%%%%% STATES VALIDITY CHECKING %%%%%%%%%%%%%
    get_static_fluent_numbers(GSFN),
    get_static_fluent_names(SFN),
    get_rigid_relation_names(RRN),
    get_rigid_relations(RR),
    (
        foreach(node(State, _, _), Nodes),
        fromto([], In, Out, ValidStates),
        param(RR, RRN, SFN, GSFN)
    do
        (   (check_rigid_relations(State, RRN, RR), check_static_fluents(State, SFN, GSFN)) -> Out = [State|In]
        ;   Out = In
        )
    ),
    length(Nodes, NL),
    length(ValidStates, VSL),
    NL == VSL,
    %%%%%%%%%%%% NON EMPTY SET CHECKING %%%%%%%%%%%%%
    NL \== 0,
    is_set(Nodes),
    %%%%%%%%%%%% NON INITIAL AND GOAL STATE INCLUSION CHECKING %%%%%%%%%%%%%
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    \+ memberchk(node(InitialState, _, _), Nodes),
    check_if_goal_state_included(Nodes).
    % export_problems(ValidStates, tests, GeneratorName, 0).

% creates new planning problems for each state.
% It uses Index and GeneratorName to name the files.
%% export_problems(+States, +FolderName, +GeneratorName, +Index).
export_problems([], _, _, _).
export_problems([State|T], FolderName, GeneratorName, Index) :-
    %% creates a new problem
    get_problem(Problem),
    Problem = problem(Name, D, R, OD, IS, _, C, MS, LS),
    NewProblem = problem(Name, D, R, OD, IS, State, C, MS, LS),
    %% gets different information to build its filename
    get_domain(Domain),
    domain_name(Domain, DomainName),
    number_codes(Index, Codes),
    atom_codes(IndexAtom, Codes),
    %% builds the filename
    atom_concat(FolderName, '/', Folder),
    atom_concat(Folder, DomainName, DomainNameInFolder),
    atom_concat(DomainNameInFolder, '/', DomainFolder),
    atom_concat(DomainFolder, GeneratorName, GeneratorInDomainFolder),
    atom_concat(GeneratorInDomainFolder, '__', Tmp1),
    atom_concat(Tmp1, Name, Tmp2),
    atom_concat(Tmp2, IndexAtom, Tmp3),
    atom_concat(Tmp3, '.pddl', Filename),
    % serialises the problem
    serialise_problem(NewProblem, Filename),
    NewIndex is Index + 1,
    export_problems(T, GeneratorName, NewIndex).

:- end_tests(generators).