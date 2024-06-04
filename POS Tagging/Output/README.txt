The instructions to run the code are as follows:

1. The input files (train, dev, test) are saved in the same directory as the code is saved. So, to access the input files from the code I used open("file_name).

Example: open("train")

2. The output files (vocab.txt, hmm.json, greedy.out, viterbi.out) are saved into the same directory that the code is present. 

The folder structure should be as follows:

Some directory:
--> CSCI544_HW2_1582852239.py
--> train
--> dev
--> test

After the execution of the code, the output files will be saved into the same directory, hence the directory looks like:

Some directory:
--> CSCI544_HW2_1582852239.py
--> train
--> dev
--> test
--> vocab.txt
--> hmm.json
--> greedy.out
--> viterbi.out