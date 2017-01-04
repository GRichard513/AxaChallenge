# coding: latin-1

import pandas as pd

input_file = "submission_test_antoine.txt"
output_file = "submission_test_antoine_modif.txt"

submission = pd.read_csv(input_file, sep="\t", encoding ='latin1')
print("File read.")
submission['DATE'] = [dd + ".000" for dd in submission['DATE']]
print("Date modified.")
submission.to_csv(output_file, sep="\t", index=False)
print("All done.")
