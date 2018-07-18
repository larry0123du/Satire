import os
from svmclassifier import BASE,FAKE_TEXT,TRUE_TEXT,FAKE_FILE,TRUE_FILE,makedata

def check_fake():
    sample_fake = os.path.join(FAKE_TEXT, FAKE_FILE[0]) 
    data = makedata(sample_fake, 0)

    print(data[0][0])


def check_true():
    sample_true = os.path.join(TRUE_TEXT, TRUE_FILE[9])
    data = makedata(sample_true, 1)

    print(data[0][0])
check_fake()
# check_true()
