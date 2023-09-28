import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeNode:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.prediction = None

def entropy(y):
    if len(y) == 0:
        return 0
    p0 = len([1 for label in y if label == 0]) / len(y)
    p1 = 1 - p0
    if p0 == 0 or p1 == 0:
        return 0
    return -p0 * math.log2(p0) - p1 * math.log2(p1)

def information_gain_ratio(X, y, feature_idx, threshold):
    n = len(y)
    left_indices = X[:, feature_idx] >= threshold
    right_indices = ~left_indices

    if np.all(left_indices) or np.all(right_indices):
        return 0

    entropy_parent = entropy(y)
    entropy_left = entropy(y[left_indices])
    entropy_right = entropy(y[right_indices])

    gain = entropy_parent - (sum(left_indices) / n) * entropy_left - (sum(right_indices) / n) * entropy_right
    split_information = -(sum(left_indices) / n) * math.log2(sum(left_indices) / n) - (sum(right_indices) / n) * math.log2(sum(right_indices) / n)

    if split_information == 0:
        return 0

    return gain / split_information

def build_decision_tree(X, y):
    if len(np.unique(y)) == 1:
        leaf = DecisionTreeNode()
        leaf.is_leaf = True
        leaf.prediction = y[0]
        return leaf

    if len(y) == 0:
        leaf = DecisionTreeNode()
        leaf.is_leaf = True
        leaf.prediction = 1  # Predict y=1 when there's no majority class
        return leaf

    best_gain_ratio = 0
    best_feature_idx = None
    best_threshold = None

    for feature_idx in range(X.shape[1]):
        unique_values = np.unique(X[:, feature_idx])
        for threshold in unique_values:
            gain_ratio = information_gain_ratio(X, y, feature_idx, threshold)
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_feature_idx = feature_idx
                best_threshold = threshold

    if best_gain_ratio == 0:
        leaf = DecisionTreeNode()
        leaf.is_leaf = True
        leaf.prediction = 1  # Predict y=1 when there's no majority class
        return leaf
    
    left_indices = X[:, best_feature_idx] >= best_threshold
    right_indices = ~left_indices

    tree = DecisionTreeNode()
    tree.feature_idx = best_feature_idx
    tree.threshold = best_threshold
    tree.left = build_decision_tree(X[left_indices], y[left_indices])
    tree.right = build_decision_tree(X[right_indices], y[right_indices])

    return tree

def predict(tree, x):
    if tree.is_leaf:
        return tree.prediction
    if x[tree.feature_idx] >= tree.threshold:
        return predict(tree.left, x)
    else:
        return predict(tree.right, x)
    
def print_tree(tree):
    if tree.is_leaf:
        print("leaf")
        print(tree.prediction)
        return 
    else:
        print("node feature_id: ", tree.feature_idx, "threshold: ", tree.threshold)
        print("left tree")
        print_tree(tree.left)
        print("right tree") 
        print_tree(tree.right)   

def num_nodes(tree):
    if tree.is_leaf:
        return 1
    else:
        return 1 + num_nodes(tree.left) + num_nodes(tree.right)
    
def visualize_tree(tree):
    if tree.is_leaf:
        return
    elif tree.feature_idx == 0:
        plt.axhline(y = tree.threshold)
        visualize_tree(tree.left)
        visualize_tree(tree.right)
    else:
        plt.axvline(x = tree.threshold)
        visualize_tree(tree.left)
        visualize_tree(tree.right)


# Load your data here (X and y)
# X should be a numpy array with shape (n_samples, n_features)
# y should be a numpy array with shape (n_samples,)
#d = np.loadtxt("Homework 2 data/D2.txt", dtype="float")
#X = d[:,:2]
#y = d[:,2:].astype(int)
# Build the decision tree
#tree = build_decision_tree(X, y)



# Make predictions
# dt = np.loadtxt("Homework 2 data/D1.txt", dtype="float")
# Xt = dt[:,:2]
# yt = dt[:,2:].astype(int)
# test_example = Xt[1]  # Replace x1 and x2 with your test data
# prediction = predict(tree, test_example)
# print("Predicted class:", prediction)

d = np.loadtxt("Homework 2 data/Dbig.txt", dtype="float")
X = d[:,:2]
y = d[:,2:].astype(int)

permutation = np.random.permutation(len(X))

train_size = 8192
X_train, X_test, y_train, y_test = train_test_split(X[permutation], y[permutation], train_size=train_size, shuffle=False)

# Initialize lists to store results
n_values = [32, 128, 512, 2048, 8192]  # Training set sizes
num_nodes_list = []  # Number of nodes in decision trees
test_errors = []  # Test set errors

# # Train decision trees for different training set sizes
# for n in n_values:
#     # Take the first n items from the training set
#     X_train_n = X_train[:n]
#     y_train_n = y_train[:n]

#     # Train a decision tree classifier
#     tree = build_decision_tree(X_train_n, y_train_n)
#     print("tree built")
#     errors = 0
#     for i in range(len(X_test)):
#        if predict(tree, X_test[i]) != y_test[i]:
#            errors += 1

#     print("n val: ", n, " errors: ", errors, " num nodes: ", num_nodes(tree))
#     plt.scatter(X_test[y_test.flatten()==0][:, 0], X_test[y_test.flatten()==0][:, 1], label='Class 0', marker='o', color='blue')
#     plt.scatter(X_test[y_test.flatten() == 1][:, 0], X_test[y_test.flatten() == 1][:, 1], label='Class 1', marker='x', color='red')
#     visualize_tree(tree)
#     plt.show()

for n in n_values:
# Take the first n items from the training set
    X_train_n = X_train[:n]
    y_train_n = y_train[:n]

    # Train a decision tree classifier
    tree = DecisionTreeClassifier()
    tree.fit(X_train_n, y_train_n)
    errors = 0
    for i in range(len(X_test)):
       if tree.predict([X_test[i]]) != y_test[i]:
           errors += 1
    print("n val: ", n, " errors: ", errors, " num nodes: ", tree.tree_.node_count)


