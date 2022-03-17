import subprocess
import os
import sys

def run_planner(timeout: int, alias: str, domain: str, problem: str, output: str):
    path = os.getcwd().replace('\\', '/').lower()
    command = f'docker run --rm -v {path}:/host aibasel/downward --plan-file /host/{output} --overall-time-limit {timeout}s --overall-memory-limit 2G --alias {alias} /host/{domain} /host/{problem}'
    process = subprocess.run(command, stdout=subprocess.DEVNULL)
    if process.returncode != 0:
        print(f'timeout suspected {alias} {str(timeout)}s (error {process.returncode} on {domain} {problem})')
        f = open(output, 'w')
        f.close()

run_planner(int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])