import os
import math
import copy
import random

class decision_tree:
    def __init__(self, data):   # data = [data1,data2,...], data1 = [[feature],label]
        self.lchild = None      # lchild could be another tree, or simply a float
        self.rchild = None
        self.feature_id = None
        self.feature_thres = None

        best_feature = None
        best_theta = None    # does not store actual theta, stores i for (i,i+1)
        best_gini_impurity = 10  # gini impurity <= 1


        for feature_i in range(len(data[0][0])):
            data = sorted(data, key=lambda one_data:one_data[0][feature_i])


            for i in range(len(data)-1):

                theta = (data[i][0][feature_i] + data[i+1][0][feature_i]) /2

                gini,_,_ = self.gini(data[:i+1],data[i+1:])


                if gini < best_gini_impurity:
                    best_feature = feature_i
                    best_theta = i
                    best_gini_impurity = gini


        data = sorted(data, key=lambda one_data: one_data[0][best_feature])
        new_data_left = copy.deepcopy(data[:best_theta+1])
        new_data_right = copy.deepcopy(data[best_theta+1:])

        theta = (data[best_theta][0][best_feature] + data[best_theta + 1][0][best_feature]) / 2

        self.feature_id = best_feature
        # print(best_feature)
        self.feature_thres = theta
        # print(theta)

        # print(len(data[:best_theta+1]),len(data[best_theta+1:]))
        gini, left_gini, right_gini = self.gini(data[:best_theta+1], data[best_theta+1:])
        if left_gini == 0:
            self.lchild = data[best_theta][1]
        else:
            self.lchild = decision_tree(new_data_left)

        if right_gini == 0:
            self.rchild = data[best_theta+1][1]
        else:
            self.rchild = decision_tree(new_data_right)



    def eval(self,data):    # data = [feature]
        if data[self.feature_id] >= self.feature_thres:
            return self.rchild if type(self.rchild) is int else self.rchild.eval(data)
        else:
            return self.lchild if type(self.lchild) is int else self.lchild.eval(data)


    def gini(self,data1,data2):     # return root_gini, left_gini, right_gini

        l = [0,0]
        for data in data1:
            if data[1] == 1:
                l[1] += 1
            else:
                l[0] += 1

        left_gini = 1-pow((l[0]/(l[0]+l[1])),2)-pow((l[1]/(l[0]+l[1])),2)



        r = [0, 0]
        for data in data2:
            if data[1] == 1:
                r[1] += 1
            else:
                r[0] += 1

        right_gini = 1 - pow((r[0] / (r[0] + r[1])), 2) - pow((r[1] / (r[0] + r[1])), 2)



        l_total = l[0]+l[1]
        r_total = r[0]+r[1]

        total_gini = left_gini * (l_total/(l_total+r_total)) + right_gini * (r_total/(l_total+r_total))

        return total_gini, left_gini, right_gini


train_data = []
with open('hw6_train.dat','r') as f:
    while True:
        feature = []
        str = f.readline()
        if str == '':
            break

        tokens = str.split(' ')
        for tok in tokens[:-1]:
            feature.append(float(tok))

        label = int(float(tokens[-1][:-1]))

        train_data.append([feature,label])


test_data = []
with open('hw6_test.dat','r') as f:
    while True:
        feature = []
        str = f.readline()
        if str == '':
            break

        tokens = str.split(' ')
        for tok in tokens[:-1]:
            feature.append(float(tok))

        label = int(float(tokens[-1][:-1]))

        test_data.append([feature,label])


# Problem 14
# D = decision_tree(train_data)
# error = 0
# total = 0
# for test in test_data:
#     prediction = D.eval(test[0])
#     total += 1
#     if prediction != test[1]:
#         error += 1
# print(error/total)



# Problem 15~18
forest = []
iterations = 2000


for i in range(iterations):
    if i%1 == 0:
        print(i)
    sampled_data = []
    sampled_id = []
    for j in range(int(len(train_data)/2)):
        id = random.randint(0,len(train_data)-1)
        sampled_data.append(train_data[id])
        sampled_id.append(id)

    D = decision_tree(sampled_data)
    forest.append((D,sampled_id))


total_eout = 0
for tree,_ in forest:
    error = 0
    total = 0
    for test in test_data:
        prediction = tree.eval(test[0])
        total += 1
        if prediction != test[1]:
            error += 1
    total_eout += error/total

print(total_eout/iterations)


error = 0
total = 0
for train in train_data:
    vote = [0,0]
    for tree,_ in forest:
        if tree.eval(train[0]) == 1:
            vote[1] += 1
        else:
            vote[0] += 1
    if (vote[1]-vote[0])*train[1] < 0:    # incorrect prediction
        error += 1
    total += 1

print(f'ein = {error/total}')

error = 0
total = 0
for test in test_data:
    vote = [0,0]
    for tree,_ in forest:
        if tree.eval(test[0]) == 1:
            vote[1] += 1
        else:
            vote[0] += 1
    if (vote[1]-vote[0])*test[1] < 0:    # incorrect prediction
        error += 1
    total += 1

print(f'eout = {error/total}')


error = 0
for i,train in enumerate(train_data):
    forest_gminus = []
    for tree,sampled_id in forest:
        if i not in sampled_id:
            forest_gminus.append(tree)

    if len(forest_gminus) != 0:
        vote = [0,0]
        for tree in forest_gminus:
            if tree.eval(train[0]) == 1:
                vote[1] += 1
            else:
                vote[0] += 1
        if (vote[1]-vote[0])*train[1] < 0:    # incorrect prediction
            error += 1

    else:
        error += 1 if train[1]>0 else 0


print(f'eoob = {error/len(train_data)}')