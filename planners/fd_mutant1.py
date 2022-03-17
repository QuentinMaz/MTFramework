import subprocess
import os
import sys

def run_planner(timeout: int, eval: str, domain: str, problem: str, output: str):
    path = os.getcwd().replace('\\', '/').lower()
    command = f'docker run --rm -v {path}:/host aibasel/downward --plan-file /host/{output} --overall-time-limit {timeout}s --overall-memory-limit 2G /host/{domain} /host/{problem} --evaluator "h={eval}()" --search "eager(single(sum([weight(g(), -1), weight(h, -1)])), reopen_closed=false, f_eval=sum([weight(g(), -1), weight(h, -1)]))"'
    process = subprocess.run(command, stdout=subprocess.DEVNULL)
    if process.returncode != 0:
        print(f'timeout suspected mutant({eval}) {str(timeout)}s (error {process.returncode} on {domain} {problem})')
        f = open(output, 'w')
        f.close()

run_planner(int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])