import subprocess
import os
import sys

def run_fastdownward(timeout: int, eval: str, weight: int, domain: str, problem: str, output: str) -> None:
    sas_filename = 'tmpsasfile'
    command = f'python3 ../downward/fast-downward.py --plan-file {output} --overall-time-limit {timeout}s --overall-memory-limit 4G --sas-file {sas_filename} {domain} {problem} --evaluator "h={eval}()" --search "eager_wastar([h()], reopen_closed=false, w={weight})"'
    process = subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)
    if process.returncode != 0:
        print(f'timeout suspected wastar_{weight}({eval}) {str(timeout)}s (error {process.returncode} on {domain} {problem})')
        f = open(output, 'w')
        f.close()
    try:
        os.remove(sas_filename)
    except:
        print(f'SAS file {sas_filename} not found.')

run_fastdownward(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])