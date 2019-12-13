results_dir='results/'
problem_dir = 'starting_kit/ingestion_program/' 

from sys import path; path.append(problem_dir);
import datetime 
from data_io import zipdir


the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
sample_result_submission = 'sample_result_submission_' + the_date + '.zip'
zipdir(sample_result_submission, results_dir)
print("Submit one of these files (codalab.lri.fr):\n" + sample_result_submission)