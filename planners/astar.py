import subprocess
import sys
import os

TIMEOUT = 120

def main(eval: str, domain: str, problem: str, output: str) -> None:
    sas = f'{output.split(".")[0]}.sas'
    command = f'fast-downward.py --plan-file {output} --sas-file {sas} {domain} {problem} --evaluator "h={eval}()" --search "astar(h())"'
    try:
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, timeout=TIMEOUT)
        os.remove(sas)
    except:
        print(f'astar {eval} error on {domain} {problem}.')
        f = open(output, 'w')
        f.close()

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('wrong number of input parameters.')
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
