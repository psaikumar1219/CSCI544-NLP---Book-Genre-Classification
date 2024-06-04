The following are the instructions to run the code:

1. The input files (train, dev, test) are saved in the same directory as the code is saved. So, to access the input files from the code I used open("file_name").

Example: with open('train', 'r')

2. The output files (dev1.out, dev2.out, test1.out, test2.out, blstm1.pt, blstm2.pt) are saved into the same directory that the code is present. 

The folder structure should be as follows:

Some directory:
--> CSCI544_HW4_1582852239.py
--> train
--> dev
--> test
--> glove.6B.100d

To produce the output files, we need to execute the .py file by using the following command:
--> $ python CSCI544_HW4_1582852239.py

After the execution of the code, the output files will be saved into the same directory, the directory looks like:

Some directory:
--> CSCI544_HW4_1582852239.py
--> train
--> dev
--> test
--> dev1.out
--> dev2.out
--> test1.out
--> test2.out
--> blstm1.pt
--> blstm2.pt

3. For Executing the code using the perl command, the conll03eval script has to be in the same directory as the generated output files (dev1.out, dev2.out).