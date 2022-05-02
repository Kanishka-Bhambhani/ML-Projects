import numpy as np
import math
import sys
import csv
import time

def extract_data(filename):
    data = np.loadtxt(fname=filename, delimiter=",", dtype=float)
    rows,features = data.shape
    data_vector = np.zeros((rows,features))
    data_vector[:,1:] = data[:,1:]
    data_vector[:,0] = 1
    labels = data[:,0]
    y = np.zeros((labels.shape[0],4))
    for index,row in enumerate(labels): 
        if int(row) == 0:
            y[index,0] = 1
        elif int(row) == 1:
            y[index,1] = 1
        elif int(row) == 2:
            y[index,2] = 1
        else:
            y[index,3] = 1
    return data_vector,y,labels

def SGD(train_data,train_data_labels, valid_data, valid_data_labels, init_param, hidden_units, epochs, learning_rate, filename):
    unique_labels = 4
    alpha,beta = initialization(hidden_units,unique_labels,train_data, init_param)
    s_alpha = np.zeros([alpha.shape[0],alpha.shape[1]])
    s_beta = np.zeros([beta.shape[0],beta.shape[1]])
    epsilon = 0.00001
    cross_entropy_valid_list = []
    cross_entropy_train_list = []
    for i in range(epochs):
        J_train_list = []
        J_valid_list = []
        cross_entropy_train = 0
        cross_entropy_valid = 0
        for index, data in enumerate(train_data):
            x,a,z,b,y_hat,J = NNForward(data, train_data_labels[index], alpha, beta)
            g_alpha,g_beta = NNBackward(alpha,beta,x,train_data_labels[index],a,z,b,y_hat,J)
            s_alpha = s_alpha + np.square(g_alpha)
            s_beta = s_beta + np.square(g_beta)
            alpha = alpha - np.multiply(learning_rate/np.sqrt(s_alpha + epsilon), g_alpha)
            beta = beta - np.multiply(learning_rate/np.sqrt(s_beta + epsilon), g_beta)
        for index, data in enumerate(train_data):
            x,a,z,b,y_hat,J = NNForward(data, train_data_labels[index], alpha, beta)
            J_train_list.append(J)
        for index, data in enumerate(valid_data):
            x,a,z,b,y_hat,J = NNForward(data, valid_data_labels[index], alpha, beta)
            J_valid_list.append(J)
        cross_entropy_train = sum(J_train_list)/train_data.shape[0]
        cross_entropy_valid = sum(J_valid_list)/valid_data.shape[0]
        error_file = open(filename, 'a')
        error_file.write('epoch='+str(i+1)+' crossentropy(train): '+str(cross_entropy_train)+'\n')
        error_file.write('epoch='+str(i+1)+' crossentropy(validation): '+str(cross_entropy_valid)+'\n')
        error_file.close()   
        cross_entropy_train_list.append(cross_entropy_train)
        cross_entropy_valid_list.append(cross_entropy_valid)
    return alpha,beta, cross_entropy_train_list, cross_entropy_valid_list

def prediction(data, data_labels, g_alpha, g_beta):
    y_list = []
    for index, rows in enumerate(data):
        x,a,z,b,y_hat,J = NNForward(rows,data_labels[index], g_alpha, g_beta)
        y_list.append(np.argmax(y_hat))
    return y_list

def NNForward(x,y, alpha, beta):
    a = np.matmul(alpha,x.T)
    z_intermediate = 1 / (1 + np.exp(-a))
    z = np.zeros(len(z_intermediate) + 1)
    z[1:] = z_intermediate
    z[0] = 1
    b = np.matmul(beta,z.T)
    y_hat = np.exp(b)/np.exp(b).sum()
    J = -np.dot(y,np.log(y_hat))
    return x,a,z,b,y_hat,J

def NNBackward(alpha,beta,x,y,a,z,b,y_hat,J):
    dl_by_db = y_hat - y
    dl_by_dz = np.matmul(dl_by_db,beta[:,1:])
    dl_by_dbeta = np.matmul(z.reshape(-1,1),dl_by_db.reshape(1,-1)).T
    dl_by_da = dl_by_dz*(z[1:]*(1-z[1:]))
    dl_by_dalpha = np.matmul(x.reshape(-1,1),dl_by_da.reshape(1,-1))
    return dl_by_dalpha.T,dl_by_dbeta

def initialization(hidden_units,unique_labels,train_data, init_param):
    if(int(init_param) == 2):
        alpha = np.zeros((int(hidden_units),train_data.shape[1]))
        beta = np.zeros((unique_labels,int(hidden_units)+1))
    elif(int(init_param) == 1):
        alpha = np.random.uniform(-0.1, 0.1,size =(int(hidden_units),train_data.shape[1]))
        beta = np.random.uniform(-0.1, 0.1,size=(unique_labels,int(hidden_units)+1))
        alpha[:,0] = 0
        beta[:,0] = 0
    return alpha,beta

def write_file(filename, labels):
        np.savetxt(filename, labels, fmt='%d', delimiter='\t')

def error(output_labels, predicted_labels):
    count = 0
    for index in range(output_labels.shape[0]):
        if(int(float(output_labels[index])) != predicted_labels[index]):
            count = count + 1
    error_rate = count/output_labels.shape[0]
    return error_rate

if __name__ == '__main__':
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    train_out = sys.argv[3]
    validation_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = sys.argv[6]
    hidden_units = sys.argv[7]
    init_flag = sys.argv[8]
    learning_rate = sys.argv[9]

    start_time = time.time()
    train_data, train_data_labels, y_train_label = extract_data(train_input)
    valid_data, valid_data_labels, y_valid_label = extract_data(validation_input)

    g_alpha, g_beta, train_list, valid_list = SGD(train_data, train_data_labels,valid_data, valid_data_labels,int(init_flag),int(hidden_units),int(num_epoch),float(learning_rate),metrics_out)
    
    y_train_list = prediction(train_data,train_data_labels,g_alpha,g_beta)
    write_file(train_out,np.array(y_train_list))
    
    y_valid_list = prediction(valid_data,valid_data_labels,g_alpha,g_beta)
    write_file(validation_out,np.array(y_valid_list))

    train_error_rate = error(y_train_label,np.array(y_train_list))
    valid_error_rate = error(y_valid_label,np.array(y_valid_list))
    error_file = open(metrics_out, 'a')
    error_file.write('error(train): '+str(train_error_rate)+'\n')
    error_file.write('error(validation): '+str(valid_error_rate))
    error_file.close()

    with open('cross_entropy_train.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(list(zip(train_list)))
    with open('cross_entropy_train.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(list(zip(valid_list)))


