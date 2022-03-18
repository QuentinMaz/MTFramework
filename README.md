# Metamorphic Testing of Optimality in AI Planners (MorphinPlan)

MorphinPlan is a framework that sets up metamorphic testing for checking optimal planning. In the following, we detail the different steps to replicate the experiments done to evaluate this tool, from their execution to the data mining. Please do not rename the folder of the repository when cloning (i.e., use `git clone` command without additional parameter).  

## Requirements
We suppose that the machine is running a Windows 10 OS. Also, in order to build MorphinPlan, make sure SICStus is installed (we used version 4.7.0). If it is not the case, the all-in-one executable used for the experiments is available on this present repository (main.exe). Python (version 3.10) as well as Docker are also needed (to launch the different AI planners). Precisely, Python scripts are used to call  the mutated planners and the online planners ([WebPlanner](https://web-planner.herokuapp.com/) and [plannin.domains](http://editor.planning.domains/)), while Docker is used to run with ease the Fast Downward planners.

## Python Installation
Download the last version of Python [here](https://www.python.org/). Make sure to add the `python` command (in PATH). Install the additional packages:
- `pip install requests` (needed for running the online planners).
- `pip install numpy` (needed for mining data).
- `pip install pandas` (needed for mining data).
- `pip install matplotlib` (needed for mining data).

## Docker Installation
Follow the instructions of the [Docker website](https://docs.docker.com/desktop/windows/install/) to install Docker. Once again, make sure that the command `docker` is now available on the system. Then, pull the official Fast Downward Docker image with `docker pull aibasel/downward`.

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
MorphinPlan outputs its result in a .csv file. We provide a Python script, *results_maker.py*, that handles the execution of all the experiments. Simply run the script with `python results_maker.py`. All the subfiles are written in the *data* directory. At the end of the script, they are regrouped into a single file *results.csv* (directly in the directory of this repository).

## Mining of the raw data
The exploitation of the raw results are proceeded by the script *results_miner.py*, which can be launched with `python results_miner.py`. It reads *results.csv* and outputs the following files:
- *mutation_scores.csv*, the results used in RQ1.
- *generators_scores.csv*, the results related to RQ2.1.
- *heuristics_chart.png*, the bars chart for RQ2.2. 
- *non_optimal_detection.csv*, the results for RQ3.

They may not present the results in the same manner as done in the MorphinPlan's paper. In any case, they are easily readable.

## Link to data actually used in the paper
If, for whatever reason, the experiments can't be reproduced, the aforementioned files can be freely accessed [here](https://doi.org/10.5281/zenodo.6368009).