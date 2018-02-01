# Machine Learning Engineer Nanodegree
## Supervised Learning
## Project:
Building a Student Intervention System

Welcome to the second project of the Machine Learning Engineer Nanodegree! In
this notebook, some template code has already been provided for you, and it will
be your job to implement the additional functionality necessary to successfully
complete this project. Sections that begin with **'Implementation'** in the
header indicate that the following block of code will require additional
functionality which you must provide. Instructions will be provided for each
section and the specifics of the implementation are marked in the code block
with a `'TODO'` statement. Please be sure to read the instructions carefully!
In addition to implementing code, there will be questions that you must answer
which relate to the project and your implementation. Each section where you will
answer a question is preceded by a **'Question X'** header. Carefully read each
question and provide thorough answers in the following text boxes that begin
with **'Answer:'**. Your project submission will be evaluated based on your
answers to each of the questions and the implementation you provide.
>**Note:** Code and Markdown cells can be executed using the **Shift + Enter**
keyboard shortcut. In addition, Markdown cells can be edited by typically
double-clicking the cell to enter edit mode.

### Question 1 - Classification vs. Regression
*Your goal for this project is to
identify students who might need early intervention before they fail to
graduate. Which type of supervised learning problem is this, classification or
regression? Why?*

**Answer: **
It is classification problem, because the task is to divide
students into two classes: those who might need early intervention and those who
not

## Exploring the Data
Run the code cell below to load necessary Python libraries
and load the student data. Note that the last column from this dataset,
`'passed'`, will be our target label (whether the student graduated or didn't
graduate). All other columns are features about each student.

```python
# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"
```

### Implementation: Data Exploration
Let's begin by investigating the dataset to
determine how many students we have information on, and learn about the
graduation rate among these students. In the code cell below, you will need to
compute the following:
- The total number of students, `n_students`.
- The total
number of features for each student, `n_features`.
- The number of those
students who passed, `n_passed`.
- The number of those students who failed,
`n_failed`.
- The graduation rate of the class, `grad_rate`, in percent (%).

```python
# Calculate number of students
n_students = len(student_data)

# Calculate number of features
n_features = student_data.shape[1] - 1

# Calculate passing students
n_passed = len(student_data[student_data['passed'] == 'yes'])

# Calculate failing students
n_failed = len(student_data[student_data['passed'] == 'no'])

# Calculate graduation rate
grad_rate = n_passed*100.0 / n_students

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)
```

## Preparing the Data
In this section, we will prepare the data for modeling,
training and testing.

### Identify feature and target columns
It is often the
case that the data you obtain contains non-numeric features. This can be a
problem, as most machine learning algorithms expect numeric data to perform
computations with.

Run the code cell below to separate the student data into
feature and target columns to see if any features are non-numeric.

```python
# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()
```

### Preprocess Feature Columns

As you can see, there are several non-numeric
columns that need to be converted! Many of them are simply `yes`/`no`, e.g.
`internet`. These can be reasonably converted into `1`/`0` (binary) values.
Other columns, like `Mjob` and `Fjob`, have more than two values, and are known
as _categorical variables_. The recommended way to handle such a column is to
create as many columns as possible values (e.g. `Fjob_teacher`, `Fjob_other`,
`Fjob_services`, etc.), and assign a `1` to one of them and `0` to all others.
These generated columns are sometimes called _dummy variables_, and we will use
the [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-
docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies)
function to perform this transformation. Run the code cell below to perform the
preprocessing routine discussed in this section.

```python
def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))
```

### Implementation: Training and Testing Data Split
So far, we have converted
all _categorical_ features into numeric values. For the next step, we split the
data (both features and corresponding labels) into training and test sets. In
the following code cell below, you will need to implement the following:
-
Randomly shuffle and split the data (`X_all`, `y_all`) into training and testing
subsets.
  - Use 300 training points (approximately 75%) and 95 testing points
(approximately 25%).
  - Set a `random_state` for the function(s) you use, if
provided.
  - Store the results in `X_train`, `X_test`, `y_train`, and `y_test`.

```python
# Import any additional functionality you may need here
from sklearn import model_selection as ms

# Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = ms.train_test_split(X_all, y_all, test_size=num_test, random_state=42)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])
```

## Training and Evaluating Models
In this section, you will choose 3 supervised
learning models that are appropriate for this problem and available in `scikit-
learn`. You will first discuss the reasoning behind choosing these three models
by considering what you know about the data and each model's strengths and
weaknesses. You will then fit the model to varying sizes of training data (100
data points, 200 data points, and 300 data points) and measure the F<sub>1</sub>
score. You will need to produce three tables (one for each model) that shows the
training set size, training time, prediction time, F<sub>1</sub> score on the
training set, and F<sub>1</sub> score on the testing set.

**The following
supervised learning models are currently available in** [`scikit-
learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may
choose from:**
- Gaussian Naive Bayes (GaussianNB)
- Decision Trees
- Ensemble
Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
- K-Nearest
Neighbors (KNeighbors)
- Stochastic Gradient Descent (SGDC)
- Support Vector
Machines (SVM)
- Logistic Regression

### Question 2 - Model Application
*List three supervised learning models that
are appropriate for this problem. For each model chosen*
- Describe one real-
world application in industry where the model can be applied. *(You may need to
do a small bit of research for this — give references!)* 
- What are the
strengths of the model; when does it perform well? 
- What are the weaknesses of
the model; when does it perform poorly?
- What makes this model a good candidate
for the problem, given what you know about the data?

**Answer: **

For this task, I will choose SVM, Naive Bayes and AdaBoost.


1)
SVM can be aplied to classification and regression tasks. For example,
handwritting recognition, image clasification, spam filtering. It has a lot of
real-world industry applications, and, among them, there is SVM application in
mechanical faults diagnostic, described here
http://www.sciencedirect.com/science/article/pii/S0957417410013801.
Pros of SVM
a) Accuracy
b) Works well on smaller cleaner datasets
c) It can be more
efficient because it uses a subset of training points

Cons of SVM
a) Isn’t
suited to larger datasets as the training time with SVMs can be high
b) Less
effective on noisier datasets with overlapping classes

I think it will be a
good choice for this task because it has high accureacy, pretty small dataset,
classes are not overlapping.

2) Naive Bayes can be applied to classification
and regression tasks too. Real world applications are Text classification, Spam
Filtering, Sentiment Analysis. For example, it is used by google for spam
filtering and google Play apps categorization
https://arxiv.org/pdf/1608.08574.pdf
Pros of NB
It is easy and fast to predict
class of test data set. It also perform well in multi class prediction
When
assumption of independence holds, a Naive Bayes classifier performs better
compare to other models like logistic regression and you need less training
data.
It perform well in case of categorical input variables compared to
numerical variable(s). For numerical variable, normal distribution is assumed
(bell curve, which is a strong assumption).

Cons of NB
If categorical variable
has a category (in test data set), which was not observed in training data set,
then model will assign a 0 (zero) probability and will be unable to make a
prediction. This is often known as “Zero Frequency”. To solve this, we can use
the smoothing technique. One of the simplest smoothing techniques is called
Laplace estimation.
On the other side naive Bayes is also known as a bad
estimator, so the probability outputs from predict_proba are not to be taken too
seriously.
Another limitation of Naive Bayes is the assumption of independent
predictors. In real life, it is almost impossible that we get a set of
predictors which are completely independent.

I think NB can be good classifier
for this task, because it is fast and dataset is small. 

3) AdaBoost can be
applied to classification and regression tasks too. It has a lot of real world
applications, similarly to the models above. It can be used for face detection,
traffic sign recognition (http://www.maia.ub.es/~sergio/files/Transport09.pdf)
Pros of AdaBoost 
a) Fast
b) Simple and easy to program
c) No parameters to tune
(except T)
d) No prior knowledge needed about weak learner
e) Provably effective
given Weak Learning Assumption
f) versatile
Cons of AdaBoost
a) Weak classifiers
too complex leads to overfitting.
b) Weak classifiers too weak can lead to low
margins,and can also lead to overfitting.
c) From empirical evidence,AdaBoost is
particularly vulnerable to uniform noise. 

I think AdaBoost will perform well,
it is not so fast as models above, but it can bring more accuracy too.

### Setup
Run the code cell below to initialize three helper functions which you
can use for training and testing the three supervised learning models you've
chosen above. The functions are as follows:
- `train_classifier` - takes as
input a classifier and training data and fits the classifier to the data.
-
`predict_labels` - takes as input a fit classifier, features, and a target
labeling and makes predictions using the F<sub>1</sub> score.
- `train_predict`
- takes as input a classifier, and the training and testing data, and performs
`train_clasifier` and `predict_labels`.
 - This function will report the
F<sub>1</sub> score for both the training and testing data separately.

```python
def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))
```

### Implementation: Model Performance Metrics
With the predefined functions
above, you will now import the three supervised learning models of your choice
and run the `train_predict` function for each one. Remember that you will need
to train and predict on each classifier for three different training set sizes:
100, 200, and 300. Hence, you should expect to have 9 different outputs below —
3 for each model using the varying training set sizes. In the following code
cell, you will need to implement the following:
- Import the three supervised
learning models you've discussed in the previous section.
- Initialize the three
models and store them in `clf_A`, `clf_B`, and `clf_C`.
 - Use a `random_state`
for each model you use, if provided.
 - **Note:** Use the default settings for
each model — you will tune one specific model in a later section.
- Create the
different training set sizes to be used to train each model.
 - *Do not
reshuffle and resplit the data! The new training points should be drawn from
`X_train` and `y_train`.*
- Fit each model with each training set size and make
predictions on the test set (9 in total).  
**Note:** Three tables are provided
after the following code cell which can be used to store your results.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

# Initialize the three models
clf_A = GaussianNB()
clf_B = SVC(random_state=33)
clf_C = AdaBoostClassifier(random_state=42)

# Set up the training set sizes
X_train_100 = X_train[:100]
y_train_100 = y_train[:100]

X_train_200 = X_train[:200]
y_train_200 = y_train[:200]

X_train_300 = X_train[:300]
y_train_300 = y_train[:300]

# Execute the 'train_predict' function for each classifier and each training set size

train_predict(clf_A, X_train_100, y_train_100, X_test, y_test)
print "\n"
train_predict(clf_A, X_train_200, y_train_200, X_test, y_test)
print "\n"
train_predict(clf_A, X_train_300, y_train_300, X_test, y_test)
print "------------------------------------------------------\n"
train_predict(clf_B, X_train_100, y_train_100, X_test, y_test)
print "\n"
train_predict(clf_B, X_train_200, y_train_200, X_test, y_test)
print "\n"
train_predict(clf_B, X_train_300, y_train_300, X_test, y_test)
print "------------------------------------------------------\n"
train_predict(clf_C, X_train_100, y_train_100, X_test, y_test)
print "\n"
train_predict(clf_C, X_train_200, y_train_200, X_test, y_test)
print "\n"
train_predict(clf_C, X_train_300, y_train_300, X_test, y_test)
```

### Tabular Results
Edit the cell below to see how a table can be designed in
[Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-
Cheatsheet#tables). You can record your results from above in the tables
provided.


** Classifer 1 - GaussianNB **  

| Training Set Size | Training Time |
Prediction Time (test) | F1 Score (train) | F1 Score (test) |
|
:---------------: | :---------------------: | :--------------------: |
:--------------: | :-------------: |
| 100               | 0.0020
| 0.0000                 | 0.8467           | 0.8029          |
| 200
| 0.0020                  | 0.0010                 | 0.8406           | 0.7244
|
| 300               | 0.0030                  | 0.0010                 |
0.8038           | 0.7634          |

** Classifer 2 - SVC **  

| Training Set
Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score
(test) |
| :---------------: | :---------------------: | :--------------------:
| :--------------: | :-------------: |
| 100               | 0.0020
| 0.0010                 | 0.8777           | 0.7746          |
| 200
| 0.0050                  | 0.0020                 | 0.8679           | 0.7815
|
| 300               | 0.0100                  | 0.0030                 |
0.8761           | 0.7838          |

** Classifer 3 - AdaBoostClassifier **
| Training Set Size | Training Time | Prediction Time (test) | F1 Score (train)
| F1 Score (test) |
| :---------------: | :---------------------: |
:--------------------: | :--------------: | :-------------: |
| 100
| 0.1330                  | 0.0060                 | 0.9481           | 0.7669
|
| 200               | 0.1370                  | 0.0040                 |
0.8927           | 0.8281          | 
| 300               | 0.1640
| 0.0060                 | 0.8637           | 0.7820          |

## Choosing the Best Model
In this final section, you will choose from the three
supervised learning models the *best* model to use on the student data. You will
then perform a grid search optimization for the model over the entire training
set (`X_train` and `y_train`) by tuning at least one parameter to improve upon
the untuned model's F<sub>1</sub> score.

### Question 3 - Choosing the Best Model
*Based on the experiments you performed
earlier, in one to two paragraphs, explain to the board of supervisors what
single model you chose as the best model. Which model is generally the most
appropriate based on the available data, limited resources, cost, and
performance?*

**Answer: **
As seen from the experiments above, the best model in terms of
speed and performance is SVC. I think support vectors will generally be the most
appropriate choice given this amount of data. It is also pretty fast and
accurate model. Maybe when there is more data and more complex relations between
the features, AdaBoost would outperform SVC, but on this task it is not.

### Question 4 - Model in Layman's Terms
*In one to two paragraphs, explain to
the board of directors in layman's terms how the final model chosen is supposed
to work. Be sure that you are describing the major qualities of the model, such
as how the model is trained and how the model makes a prediction. Avoid using
advanced mathematical or technical jargon, such as describing equations or
discussing the algorithm implementation.*

**Answer: **

Support vector machines is ML algorithm which finds the best lines
to separate data into classes. In the task above this algoriths will separate
data into two classes: students those who might need early intervention and
those who not. It finds the best way to build that line by maximizing distance
from this plane to nearest datapoints from two classes - this distance is called
margin. Maximizing margin is a problem which can be solved using optimization
techniques. If the classes which one wants to predict for given data point
cannot be separated by the line(they are called non lineary separable), one can
use kernel trick. Kernel trick is a technique, which makes data lineary
separable in higher dimmensional space. When model makes a prediction it checks
if given test point located in one side of a line or another. Those sides
correspond to classes.

### Implementation: Model Tuning
Fine tune the chosen model. Use grid search
(`GridSearchCV`) with at least one important parameter tuned with at least 3
different values. You will need to use the entire training set for this. In the
code cell below, you will need to implement the following:
- Import
[`sklearn.grid_search.GridSearchCV`](http://scikit-
learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) and
[`sklearn.metrics.make_scorer`](http://scikit-
learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
- Create a
dictionary of parameters you wish to tune for the chosen model.
 - Example:
`parameters = {'parameter' : [list of values]}`.
- Initialize the classifier
you've chosen and store it in `clf`.
- Create the F<sub>1</sub> scoring function
using `make_scorer` and store it in `f1_scorer`.
 - Set the `pos_label`
parameter to the correct value!
- Perform grid search on the classifier `clf`
using `f1_scorer` as the scoring method, and store it in `grid_obj`.
- Fit the
grid search object to the training data (`X_train`, `y_train`), and store it in
`grid_obj`.

```python
# Import 'GridSearchCV' and 'make_scorer'
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

# Create the parameters list you wish to tune
parameters = {'C': [0.9, 1, 1.05, 1.1, 1.15], 'gamma': [0.05, 0.06, 0.07, 0.09, 0.1, 0.11],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'] }

# Initialize the classifier
clf = SVC()

# Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label='yes')

# Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, parameters, scoring=f1_scorer, cv=5)

# Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

print grid_obj.best_params_

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))
```

### Question 5 - Final F<sub>1</sub> Score
*What is the final model's
F<sub>1</sub> score for training and testing? How does that score compare to the
untuned model?*

**Answer: **

Final f1 score for training set is 0.9762, for test set is 0.7895.
It produced the best output using rbf kernel, when C=1 and gamma=0.1. It has
only a bit f1 score for testing set comparing to untunned model, but it also has
much better performance on training set.

> **Note**: Once you have completed all of the code implementations and
successfully answered each question above, you may finalize your work by
exporting the iPython Notebook as an HTML document. You can do this by using the
menu above and navigating to  
**File -> Download as -> HTML (.html)**. Include
the finished document along with this notebook as your submission.
