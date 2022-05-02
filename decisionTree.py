import numpy as np
import sys
from os import write
import math

from numpy.core.fromnumeric import shape

entropy = 0


class Node:

    def __init__(self, entropy, predict, val, classes, labels):
        self.val = val
        self.entropy = entropy
        self.predict = predict
        self.left = None
        self.right = None
        self.classes = classes
        self.labels = labels


class DecisionTreeClassifier:

    def extract_data(self,filename):
        data = np.loadtxt(fname=filename, delimiter="\t", dtype=str)
        feature_name_list = data[:1]
        feature_list = data[:, :-1]
        feature_list = feature_list[1:]
        label_list = data[1:, -1]
        unique_features = np.unique(data[1:, :-1])
        unique_labels = np.unique(data[1:, -1])
        return feature_name_list, feature_list, label_list, unique_features, unique_labels

    def entropy(self, label_list, unique_labels):
        output_label_0 = 0
        output_label_1 = 0
        output_total = len(label_list)
        for label in label_list:
            if label == list(unique_labels)[0]:
                output_label_0 += 1
            else:
                output_label_1 += 1

        if output_label_0 == 0:
            label_0 = 0
        else:
            label_0 = -((output_label_0 / output_total) * math.log2(output_label_0 / output_total))

        if output_label_1 == 0:
            label_1 = 0
        else:
            label_1 = -((output_label_1 / output_total) * math.log2(output_label_1 / output_total))

        entropy = label_0 + label_1
        return entropy

    def mutual_information(self, feature_list, label_list, unique_features, unique_labels, feature_name_list, entropy):
        mutual_info = []
        entropy_list = []

        feature_columns = feature_list.shape[1]
        for i in range(feature_columns):
            feature_0 = []
            feature_1 = []
            feature_0_count_0 = 0
            feature_0_count_1 = 0
            feature_1_count_0 = 0
            feature_1_count_1 = 0
            for index, item in enumerate(feature_list[:, i]):
                if item == list(unique_features)[0]:
                    feature_0.append(label_list[index])
                else:
                    feature_1.append(label_list[index])

            for feature in feature_0:
                if (feature == list(unique_labels)[0]):
                    feature_0_count_0 += 1
                else:
                    feature_0_count_1 += 1

            for feature in feature_1:
                if (feature == list(unique_labels)[0]):
                    feature_1_count_0 += 1
                else:
                    feature_1_count_1 += 1

            if feature_0_count_0 == 0 or feature_0_count_1 == 0:
                label_0 = 0
            else:
                feature_0_given_label_0 = -(
                        (feature_0_count_0 / len(feature_0)) * math.log2(feature_0_count_0 / len(feature_0)))
                feature_0_given_label_1 = -(
                        (feature_0_count_1 / len(feature_0)) * math.log2(feature_0_count_1 / len(feature_0)))
                label_0 = feature_0_given_label_0 + feature_0_given_label_1
            if feature_1_count_0 == 0 or feature_1_count_1 == 0:
                label_1 = 0
            else:
                feature_1_given_label_0 = -(
                        (feature_1_count_0 / len(feature_1)) * math.log2(feature_1_count_0 / len(feature_1)))
                feature_1_given_label_1 = -(
                        (feature_1_count_1 / len(feature_1)) * math.log2(feature_1_count_1 / len(feature_1)))
                label_1 = feature_1_given_label_0 + feature_1_given_label_1
            feature_entropy = (len(feature_0) / len(feature_list[:, i])) * label_0 + (
                    len(feature_1) / len(feature_list[:, i])) * label_1
            entropy_list.append(feature_entropy)

        mutual_info = [entropy - mf for mf in entropy_list]

        return mutual_info

    def grow_tree(self, feature_name_list, feature_list, label_list, unique_features, unique_labels, maxDepth):
        label = [0, 0]
        left_label = [0, 0]
        right_label = [0, 0]
        if maxDepth > feature_list.shape[1]:
            maxDepth = feature_list.shape[1]
        entropy = self.entropy(label_list, unique_labels)
        mutual_info = self.mutual_information(feature_list, label_list, unique_features, unique_labels,
                                              feature_name_list, entropy)
        root_mf = max(mutual_info)
        root_node_value = mutual_info.index(root_mf)
        root_name = feature_name_list[0][root_node_value]
        for label_main in label_list:
            if label_main == list(unique_labels)[0]:
                label[0] += 1
            else:
                label[1] += 1
        if label[0] > label[1]:
            predict = unique_labels[0]
        elif label[0] == label[1]:
            if unique_labels[0] > unique_labels[1]:
                predict = unique_labels[0]
            else:
                predict = unique_labels[1]
        else:
            predict = unique_labels[1]

        node = Node(entropy, predict, root_name, label, unique_features)
        if root_mf > 0 and maxDepth > 0:
            split_left = feature_list[np.where(feature_list[:, root_node_value] == unique_features[0])]
            split_right = feature_list[np.where(feature_list[:, root_node_value] == unique_features[1])]
            split_left_labels = label_list[np.where(feature_list[:, root_node_value] == unique_features[0])]
            split_right_labels = label_list[np.where(feature_list[:, root_node_value] == unique_features[1])]
            for label in split_left_labels:
                if label == list(unique_labels)[0]:
                    left_label[0] += 1
                else:
                    left_label[1] += 1

            for label in split_right_labels:
                if label == list(unique_labels)[0]:
                    right_label[0] += 1
                else:
                    right_label[1] += 1
            pipe_print = "|" * (max_depth_pipe - maxDepth)
            print(pipe_print, root_name, " = ", unique_features[0], " :  [ ", left_label[0], " ", unique_labels[0], "/",
                  left_label[1], " ", unique_labels[1], " ]")
            node.left = self.grow_tree(feature_name_list, split_left, split_left_labels, unique_features, unique_labels,
                                       maxDepth - 1)
            print(pipe_print, root_name, " = ", unique_features[1], " :  [ ", right_label[0], " ", unique_labels[0],
                  "/", right_label[1], " ", unique_labels[1], " ]")
            node.right = self.grow_tree(feature_name_list, split_right, split_right_labels, unique_features,
                                        unique_labels, maxDepth - 1)
        return node

    def predict(self, gTree, feature_label_list, features):
        if gTree.left is None and gTree.right is None:
            output_predict.append(gTree.predict)
            return
        for label in feature_label_list:
            for l in list(label):
                if l == gTree.val:
                    index = list(label).index(l)
        if features[index] == gTree.labels[0]:
            if gTree.left is None:
                output_predict.append(gTree.predict)
                return
            gTree = gTree.left
            self.predict(gTree, feature_label_list, row)
        else:
            if gTree.right is None:
                output_predict.append(gTree.predict)
                return
            gTree = gTree.right
            self.predict(gTree, feature_label_list, row)
    
    def calculate_error(self, output_predict, list):
        error = 0
        for i in range(len(output_predict)):
            if list[i] != output_predict[i]:
                error += 1
        return error


if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    depth_str = sys.argv[3]
    train_output = sys.argv[4]
    test_output = sys.argv[5]
    metrics_out = sys.argv[6]
    print(metrics_out)
    depth = int(depth_str)
    output_predict = []
    dTree = DecisionTreeClassifier()
    feature_name_list, feature_list, label_list, unique_features, unique_labels = dTree.extract_data(train_input)
    unique_label = [0, 0]
    for label_print in label_list:
        if label_print == list(unique_labels)[0]:
            unique_label[0] += 1
        else:
            unique_label[1] += 1
    print("[", unique_label[0], " ", unique_labels[0], "/", unique_label[1], " ", unique_labels[1], "]")
    max_depth_pipe = depth + 1
    gTree = dTree.grow_tree(feature_name_list, feature_list, label_list, unique_features, unique_labels, 3)
    for row in feature_list:
        dTree.predict(gTree, feature_name_list, row)
    with open(train_output, 'w') as out_train_file:
        for data in output_predict:
            out_train_file.write(data+'\n')
    train_error = dTree.calculate_error(output_predict, label_list)
    output_predict.clear()

    feature_name_list_test, feature_list_test, label_list_test, unique_features_test, unique_labels_test = dTree.extract_data(test_input)
    for row in feature_list_test:
        dTree.predict(gTree, feature_name_list_test, row)
    with open(test_output, 'w') as out_test_file:
        for data in output_predict:
            out_test_file.write(data+'\n')
    test_error = dTree.calculate_error(output_predict, label_list_test)
    with open(metrics_out, 'w') as error_file:
        error_file.writelines('error(train): {0:f}\n'.format(train_error / len(feature_list)))
        error_file.writelines('error(test): {0:f}'.format(test_error / len(feature_list_test)))

