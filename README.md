# Mutation-guided Metamorphic Testing of Optimality in AI Planning (MorphinPlan)
MorphinPlan is a framework that sets up metamorphic testing for checking optimal planning. In the following, we detail the different steps to replicate the experiments done to evaluate this tool, from their execution to the data mining. Please do not rename the folder of the repository when cloning (i.e., use `git clone` command without additional parameter).

## Requirements
The code supports Windows 10 and Linux-based plateforms. Also, in order to build MorphinPlan, make sure SICStus is installed (we used version 4.9.0). If it is not the case, the all-in-one executable *for Ubuntu 22.04.3 LTS* used for the experiments is available on this present repository (`main.exe`). Python (version 3.10) as well as Fast Downward are also needed. Precisely, Python scripts are used to call the mutated planners as well as the Fast Downward-based ones.

## Python Installation
Download the last version of Python [here](https://www.python.org/). Make sure to add the `python` command (in PATH). Install the additional packages:
- `pip install numpy` (needed for mining data).
- `pip install pandas` (needed for mining data).
- `pip install matplotlib` (needed for mining data).

## Fast-Downward Installation
Follow the instructions of the [Fast Downward website](https://www.fast-downward.org/). We used the GNU Compiler Collection version 11.4.0. Once installed, assign the absolute filepath to the `FD_PATH` constant of the script `fd_planner.py`(found in the `planners/` folder).

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
- Build the executable with `spld --output=main.exe --static main.sav` (we used the used the GNU Compiler Collection version 11.4.0):
```
spld --output=main.exe --static main.sav
[...]
spldgen_s_14540_1647527992_restore_main.c
spldgen_s_14540_1647527992_main_wrapper.c
spldgen_s_14540_1647527992_prolog_rtable.c
   Creating library main.lib and object main.exp
Created "main.exe"
```
At this point, a proper executable file `main.exe` should have been created.

## Replication of the results

### Execution of the two experiments
MorphinPlan outputs its result in `.csv` files. We provide a Python script, *simulate_framework.py*, that handles the entire execution of all the two experiments. Simply run the script with `python simulate_framework.py`. All the subfiles are written in the `results/` directory. Please note that some parts of the execution are parallelized: the number of threads can be adjusted with the `NB_THREADS` constant in the script.
The resources presented in the paper are:
- RQ1: `table_coverage_4.tex` (Table 4).
- RQ2: `figure3.png` and `figure4.png` (Figures 3 and 4).
- RQ3: `second_experiment_results_10.tex`.

They may not present the results in the same manner as done in the MorphinPlan's paper. In any case, they are easily readable.

### Execution time analysis
The execution times reported in the second table of the paper are computed with additional, seperate runs of the framework. Indeed, we couldn't source them from the executions of the first experiment directly since the latter involves random selection/validation splits of mutated planners. Executions times can be obtained by running the script `exec_time.py` (the expected command to do so is `python exec_time.py`). Please note that in those runs, MorphinPlan leverages all the mutants available, which thus corresponds to the worst case scenario *w.r.t* computation costs (i.e, more mutants mean greater workload). The results are exported in the file `results/execution_time_results.csv`. Eventually, although they are averages of multiple executions, please be aware that they may differ from the ones presented in the paper (hardware-dependent results).

## Link to data actually used in the paper
If, for whatever reason, the experiments can't be reproduced, the aforementioned files are uploaded on [Zenodo](https://doi.org/10.5281/zenodo.7615241).
The figures are the exact ones presented in the paper. Regarding the `.csv` files, they correspond to the ones used at the end of the script to build the figures and the latex tables. Therefore, if issues occur to run the experiments, one can always place those files in the `results/` folder and adapt the provided script to make it directly leverage the `.csv` files in order to get the figures and tables back.
