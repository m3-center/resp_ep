import os

from resp_ep import driver


ini_files = ['stage_1_a.ini', 'stage_1_b.ini']

for ini in ini_files:
	job_name = os.path.splitext(os.path.basename(ini))[0]

	print('-' * 40)
	print(f'Job: {job_name}')

	charges = driver.resp(ini)

	print(f"\n{'-' * 10} Results {'-' * 10}\n")
	print('Unrestrained Electrostatic Potential Charges')
	print(f'  {charges[0]}\n')

	print('Restrained Electrostatic Potential (RESP) Charges')
	print(f'  {charges[1]}\n')
