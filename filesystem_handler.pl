:- module(filesystem_handler, [make_inputs_from_directory/3]).

:- use_module(library(lists), [delete/3, maplist/3, prefix_length/3]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXPORTED PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% make_inputs_from_directory(+Directory, +MaximumNumberOfProblemsPerDomain, -Inputs).
make_inputs_from_directory(Directory, MaximumNumberOfProblemsPerDomain, Inputs) :-
    bagof(D, absolute_file_name(Directory, D, [glob('*'), file_type(directory), solutions(all), file_errors(fail)]), Directories),
    (
        foreach(Dir, Directories),
        fromto([], In, Out, TmpInputs),
        param(MaximumNumberOfProblemsPerDomain)
    do
        %% computes the relative path of the pddl domain
        absolute_file_name(Dir, AbsoluteDomainFilename, [glob('*domain.pddl'), solutions(first), file_errors(fail)]),
        relative_filename(AbsoluteDomainFilename, DomainFilename),
        %% gets all the .pddl files from the directory, removes domain's one and computes their relative paths
        setof(F, absolute_file_name(Dir, F, [glob('*.pddl'), solutions(all), file_errors(fail)]), PddlFilenames),
        delete(PddlFilenames, AbsoluteDomainFilename, AbsoluteProblemFilenames),
        maplist(relative_filename, AbsoluteProblemFilenames, ProblemFilenames),
        % makes the associated predicates for each new pddl problem
        make_input_predicates(DomainFilename, ProblemFilenames, NewInputs),
        %% limits the number of problems gathered (wrt MaximumNumberOfProblemsPerDomain)
        length(NewInputs, NewInputsLength),
        (   NewInputsLength =< MaximumNumberOfProblemsPerDomain -> append(In, NewInputs, Out)
        ;   (prefix_length(NewInputs, Prefix, MaximumNumberOfProblemsPerDomain), append(In, Prefix, Out))
        )
    ),
    sort(TmpInputs, Inputs).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPER PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% relative_filename(+AbsoluteFilename, -RelativeFilename).
relative_filename(AbsoluteFilename, RelativeFilename) :-
    atom_concat(Tmp, RelativeFilename, AbsoluteFilename),
    atom_concat(_, 'mtframework/', Tmp). % uses the name of the application folder to crop the absolute path (dirty...)

%% make_input_predicates(+DomainFilename, +ProblemFilenames, -InputPredicates).
make_input_predicates(_, [], []).
make_input_predicates(DomainFilename, [ProblemFilename|T1], [source_input(DomainFilename, ProblemFilename)|T2]) :-
    make_input_predicates(DomainFilename, T1, T2).