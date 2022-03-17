import subprocess
import os
import sys

def run_planner(timeout: int, eval: str, weight: int, domain: str, problem: str, output: str):
    path = os.getcwd().replace('\\', '/').lower()
    command = f'docker run --rm -v {path}:/host aibasel/downward --plan-file /host/{output} --overall-time-limit {timeout}s --overall-memory-limit 2G /host/{domain} /host/{problem} --evaluator "h={eval}()" --search "eager_wastar([h()], reopen_closed=false, w={weight})"'
    process = subprocess.run(command, stdout=subprocess.DEVNULL)
    if process.returncode != 0:
        print(f'timeout suspected wastar_{weight}({eval}) {str(timeout)}s (error {process.returncode} on {domain} {problem})')
        f = open(output, 'w')
        f.close()

run_planner(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])