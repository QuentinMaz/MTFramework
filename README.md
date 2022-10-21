# Metamorphic Testing of Optimality in AI Planners (MorphinPlan)

MorphinPlan is a framework that sets up metamorphic testing for checking optimal planning. In the following, we detail the different steps to replicate the experiments done to evaluate this tool, from their execution to the data mining. Please do not rename the folder of the repository when cloning (i.e., use `git clone` command without additional parameter).

## Requirements
We suppose that the machine is running a Windows 10 OS. Also, in order to build MorphinPlan, make sure SICStus is installed (we used version 4.7.0). If it is not the case, the all-in-one executable used for the experiments is available on this present repository (main.exe). Python (version 3.10) as well as Fast-Downward are also needed. Precisely, Python scripts are used to call the mutated planners as well as the Fast-Downward-based ones.

## Python Installation
Download the last version of Python [here](https://www.python.org/). Make sure to add the `python` command (in PATH). Install the additional packages:
- `pip install numpy` (needed for mining data).
- `pip install pandas` (needed for mining data).
- `pip install matplotlib` (needed for mining data).

## Fast-Downward Installation
Follow the instructions of the [Fast-Downward website](https://www.fast-downward.org/). We used the Microsoft Windows Resource Compiler Version 10.0.10011.16384 from VS 2022 in the Native Tools Command Prompt to build the application. Once installed, assign the absolute filepath to the `FD_PATH` constant of the script *fd_planner.py* (found in the *planners* folder).

## Building MorphinPlan
SICStus provides a simple and easy way to build an all-in-one executable from the source code. We mimic the procedure described in its user's manual. Open a terminal in the folder containing this repository and do the following:
- Run the `sicstus` command and successively execute `compile(framework).`,  `save_program('main.sav').` and `halt.`. The traces should look like:
```
sicstus
SICStus [...]
| ?- compile(framework).
% [...]
yes
| ?- save_program('main.sav').
% [...]
yes
| ?- halt.
```
- Build the executable with `spld --output=main.exe --static main.sav` (we used the Microsoft Windows Resource Compiler Version 10.0.10011.16384 from VS 2022 in the Native Tools Command Prompt):
```
spld --output=main.exe --static main.sav
[...]
spldgen_s_14540_1647527992_restore_main.c
spldgen_s_14540_1647527992_main_wrapper.c
spldgen_s_14540_1647527992_prolog_rtable.c
   Creating library main.lib and object main.exp
Created "main.exe"
```
At this point, a proper executable file *main.exe* should have been created.

## Execution of the experiments
MorphinPlan outputs its result in a .csv file. We provide a Python script, *simulate_framework.py*, that handles the entire execution of all the two experiments. Simply run the script with `python simulate_framework.py`. All the subfiles are written in the *results* directory. Please note that some parts of the execution are parallelized: the number of threads can be adjusted with the `NB_THREADS` constant in the script.
The resources presented in the paper are:
- RQ1: *results/coverage_20_10_2.tex*.
- RQ2: *results/overall_efficiency_20_10.png* and *results\n_scaling_mutation_coverage_20_10.png*.
- RQ3: *results/fd_results_10_n_scaling_coverage.png* and *results/fd_results_10_coverage.png*.

They may not present the results in the same manner as done in the MorphinPlan's paper. In any case, they are easily readable.