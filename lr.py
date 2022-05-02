import numpy as np
import math
import sys
import csv

def data_extract(filename):
    data = np.loadtxt(fname=filename, delimiter="\t", dtype=str)
    rows,features = data.shape
    data_vector = np.zeros((rows,features))
    data_vector[:,1:] = data[:,1:]
    data_vector[:,0] = 1
    labels = data[:,0]
    return data_vector,labels

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def calculate_theta_data(theta,row):
    x = np.dot(row,theta)
    return x

def calculate_sgd(epoch,data_vector,output_labels,val_data_vector,val_labels):
    log_likelihood_train = []
    log_likelihood_val = []
    theta = np.zeros([data_vector.shape[1]])
    learning_rate = np.multiply(data_vector,0.1/data_vector.shape[0])
    log_likelihood = 0
    # val_log_likelihood = 0
    for i in range(epoch):
        sigmoid_list =[]
        for index,row in enumerate(data_vector):
            x = calculate_theta_data(theta,row)
            sig_j = sigmoid(x)
            sigmoid_list.append(float(output_labels[index]) - sig_j)
            gradient_vector = np.dot(learning_rate[index],np.array(sigmoid_list)[index])
            theta = theta + gradient_vector
        for x_train in range(data_vector.shape[0]):
            theta_data = calculate_theta_data(theta,data_vector[x_train])
            log_likelihood = log_likelihood + (-(float(output_labels[x_train]))*(theta_data) + math.log(1 + math.exp(theta_data)))
        # for x_val in range(val_data_vector.shape[0]):
        #     theta_data_val = calculate_theta_data(theta,val_data_vector[x_val])
        #     val_log_likelihood = val_log_likelihood + (-(float(val_labels[x_val]))*(theta_data_val) + math.log(1 + math.exp(theta_data_val)))
        log_likelihood_train.append(log_likelihood/data_vector.shape[0])
        # log_likelihood_val.append(val_log_likelihood/val_data_vector.shape[0])
        log_likelihood = 0
        print(i)
    return theta, log_likelihood_train

def prediction(output_data_vector,theta):
    sigmoid_predict_list = []
    for rows in output_data_vector:
        row = np.array(rows,dtype=float)
        x = calculate_theta_data(theta,row)
        sig_j = sigmoid(x)
        sigmoid_predict_list.append(sig_j)
    return (np.array(sigmoid_predict_list) > 0.5)
    

def error(output_labels, sigmoid_predict):
    count = 0
    for index in range(output_labels.shape[0]):
        if(int(float(output_labels[index])) != sigmoid_predict[index]):
            count = count + 1
    error_rate = count/output_labels.shape[0]
    return error_rate

def write_file(filename, sentence_vectors):
        np.savetxt(filename, sentence_vectors, fmt='%d', delimiter='\t')

def write_error_rate(filename, test_error_rate,train_error_rate):
    with open(filename, 'w') as error_file:
        error_file.writelines('error(train): {0:f}\n'.format(train_error_rate))
        error_file.writelines('error(test): {0:f}'.format(test_error_rate))

if __name__ == '__main__':
    formatted_train_input = sys.argv[1]
    formatted_valid_input = sys.argv[2]
    formatted_test_input = sys.argv[3]
    dict_input = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    epoch = sys.argv[8]

    #train the model
    data_vector,labels = data_extract(formatted_train_input)

    #validation data prediction 
    val_data_vector,val_labels = data_extract(formatted_valid_input)

    theta_final, log_likelihood_train = calculate_sgd(int(epoch),data_vector,labels,val_data_vector,val_labels)
    
    with open('log_likelihood.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(list(zip(log_likelihood_train)))

    #train data predictiom
    train_output_prediction = prediction(data_vector, theta_final)
    train_error_rate = error(labels,train_output_prediction)
    write_file(train_out,train_output_prediction)

    #test data prediction
    test_output_data_vector,test_output_labels = data_extract(formatted_test_input)
    test_output_prediction = prediction(test_output_data_vector,theta_final)
    test_error_rate = error(test_output_labels,test_output_prediction)
    write_file(test_out,test_output_prediction)

    #error rate in file
    write_error_rate(metrics_out,test_error_rate,train_error_rate)
    
        

    

    

    

