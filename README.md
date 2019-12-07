# Java Project - *Decision Tree*

## Purpose:

The program is one of the projects for the Data Structure course. The program was developed for analyzing the accuracy of the decision tree by increasing its depth and leaves. The program utilized the training data to create a **classification model** decision tree. Then, it loaded the testing data for computing the correctness of the prediction by the built tree.

## Process:

- Determines data whether continuous or categorical by file names
- Stores the training data and the testing data by implementing ArrayList.
- Determines the majority vote as the divided point on each level of the tree.
- Creates a decision tree base on the majority vote by implementing the priority queue.
- Predicts the results by loading the testing data.
- Computes the accuracy by comparing the results with the label of the training data.

## Test:

**Input**: train_feature_file train_label_file  test_feature_file test_label_file max_height max_leaves

**Output**: the accurarcy of the correctness on testing data (the best output is 1 as 100%).

<img src='https://github.com/lxy878/DecisionTree/blob/master/TestingDT.gif' title='Testing Video' alt='Testing Video' />
GIF created with [LiceCap](http://www.cockos.com/licecap/).
