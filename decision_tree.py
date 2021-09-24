#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    #Age: Young = 1          Spectacle Prescription: Myope = 1         Astigmatism: Yes = 1  Tear Production Rate: Reduced = 1
    #     Presbyopic = 2                             Hypermetrope = 2               No = 2                       Normal = 2
    #     Prepresbyopic =3

    transform = {'Young': 1, 'Presbyopic': 2, 'Prepresbyopic': 3, 
                 'Myope': 1, 'Hypermetrope': 2, 'Yes': 1, 'No': 2, 
                 'Reduced': 1, 'Normal': 2}
    temp = 0
    tempList = []
    X = []

    for i in range(len(dbTraining)):
        for j in range(4):
            temp = transform[dbTraining[i][j]]
            tempList.append(temp)
        X.append(tempList)
        tempList = []
    


    #transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    temp = 0
    Y = []

    for i in range(len(dbTraining)):
        Y.append(transform[dbTraining[i][4]])
       


    accLow = 0
    #loop your training and test tasks 10 times here
    for i in range (10):
       accList = []
       

       #fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
       clf = clf.fit(X, Y)

       #read the test data and add this data to dbTest
       #--> add your Python code here
       dbTest = []
       with open('contact_lens_test.csv') as csvfile:
           reader = csv.reader(csvfile)
           for j, row in enumerate(reader):
               if j > 0: #skipping the header
                   dbTest.append (row)
       
       
       for data in dbTest:
           test = []
           classPredict = []
           
           #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
           #class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
           for j in range(len(dbTest)):
               for k in range(5):
                   temp = transform[dbTest[j][k]]
                   tempList.append(temp)
               test.append(tempList)
               tempList = []


           for j in range(len(test)):
               class_predicted = clf.predict([test[j][:4]])[0]
               classPredict.append(class_predicted)


           
           
           #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
           correct = 0
           accuracy = 0
           
           
           for j in range(len(classPredict)):
                   if classPredict[j] == test[j][4]:
                       correct += 1
           
           accuracy = correct / 8
           accList.append(accuracy)
           

        #find the lowest accuracy of this model during the 10 runs (training and test set)
        #--> add your Python code here
       if i == 1:
        accLow = accList[0]

       for j in range(len(accList)):
           if accList[j] < accLow:
               accLow = accList[j]


    #print the lowest accuracy of this model during the 10 runs (training and test set) and save it.
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print("final accuracy when training on", ds,":", accLow, "\n")



