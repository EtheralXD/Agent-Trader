# This is the master script I made to run all the data collection 

import subprocess
import sys

scripts = ['data_collecter.py', 'pattern_builder.py', 'label_builder.py']

for script in scripts:
    print(f'--- master.py is Starting {script} ---')
    subprocess.run([sys.executable, script], check= True)
    print(f"--- master.py has Finished {script} ---")

print('Data has all been collected and stored')