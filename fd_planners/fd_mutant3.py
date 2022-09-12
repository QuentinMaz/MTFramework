import subprocess
import os
import sys

def run_fastdownward(timeout: int, eval: str, domain: str, problem: str, output: str) -> None:
    sas_filename = 'tmpsasfile'
    command = f'python3 ../downward/fast-downward.py --plan-file {output} --overall-time-limit {timeout}s --overall-memory-limit 4G --sas-file {sas_filename} {domain} {problem} --evaluator "h={eval}()" --search "eager(single(sum([weight(g(), -1), weight(h, 10)])), reopen_closed=false, f_eval=sum([weight(g(), -1), weight(h, 10)]))"'
    process = subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)
    if process.returncode != 0:
        print(f'timeout suspected mutant3({eval}) {str(timeout)}s (error {process.returncode} on {domain} {problem})')
        f = open(output, 'w')
        f.close()
    try:
        os.remove(sas_filename)
    except:
        print(f'SAS file {sas_filename} not found.')

run_fastdownward(int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])