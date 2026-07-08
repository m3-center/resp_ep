import os
import pytest
import sys

sys.path.insert(0, os.path.abspath('/home/karl/sciebo/resp_ep'))
from resp_ep import driver

#job_list = ['stage_1.ini', 'stage_1_x.ini']
job_list = ['stage_1.ini']

for job in job_list:
	charges = driver.resp(job)

	print('Unrestrained Electrostatic Potential Charges')
	print(f'{charges[0]}\n')

	print('Restrained Electrostatic Potential (RESP) Charges')
	print(f'{charges[1]}\n')
