import requests, sys

data = {
    'domain': open(sys.argv[1], 'r').read(),
    'problem': open(sys.argv[2], 'r').read()
}


response = requests.post('https://web-planner.herokuapp.com/', verify = True, json = data).json()

with open(sys.argv[3], 'w') as f:
    f.write('\n'.join(response['plan']))