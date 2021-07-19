# Created by Antonio Di Mariano (antonio.dimariano@gmail.com) at 15/12/2019
import pandas as pd

DIRECTORY = 'csv_results/'
def build_csv(test_users,Y_pred,filename):
    """
    Build a csv
    """
    try:
        print("[] Starting creating the CSV file ",filename)
        submission = pd.DataFrame({'id': test_users['id'], 'country': Y_pred})
        submission.head()
        submission.to_csv(DIRECTORY+filename, index=False)
        print("[*] Your file is ready  :",DIRECTORY+filename)
    except Exception as error:
        return 0