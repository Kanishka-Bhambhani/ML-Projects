import numpy as np
import sys

def extract_data(filename):
    word = []
    state = []
    count = 0
    with open(filename) as f:
        data = f.readlines()
    for line in data:
        if(line != '\n'):
            word.append(line.split('\t')[0])
            state.append(line.split('\t')[1].rstrip('\n'))
        else:
            count = count + 1
            if(count == 9):
                word_10 = word
                state_10 = state
            if(count == 99):
                word_100 = word
                state_100 = state
            if(count == 999):
                word_1000 = word
                state_1000 = state
            if(count == 9999):
                word_10000 = word
                state_10000 = state
            word.append('\n')
            state.append('\n')
    return word, state, word_10, state_10, word_100, state_100, word_100, state, word_1000, state_1000, word_10000, state_10000

def extract_data_in_dict(filename):
    a_dictionary = {}
    a_file = open(filename)
    for index, line in enumerate(a_file):
        a_dictionary[line.rstrip('\n')] = index
    return a_dictionary

def init_matrix(state, tags_dict):
    init = np.ones(len(tags_dict))
    init[tags_dict[state[0]]] += 1
    for index,lines in enumerate(state):
        if(lines == '\n'): 
            init[tags_dict[state[index+1]]] += 1
    init = init/np.sum(init)
    return init

def emission_matrix(word, state, words_dict, tags_dict):
    emit = np.ones([len(tags_dict), len(words_dict)])
    for index,line in enumerate(word):
        if(line != '\n'):
            emit[tags_dict[state[index]],words_dict[line]] += 1
    word_sum = emit.sum(axis=1)
    emit = emit/word_sum[:,None]
    return emit

def transition_matrix(state, tags_dict):
    trans = np.ones([len(tags_dict), len(tags_dict)])
    for index in range(len(state)):
        if (index != (len(state) -1)) and (state[index+1] != '\n') and (state[index] != '\n'):
            trans[tags_dict[state[index]],tags_dict[state[index+1]]] += 1
    state_sum = trans.sum(axis=1)
    trans = trans/state_sum[:,None]
    return trans

def write_file(filename, data):
        np.savetxt(filename, data, delimiter=' ') 

if __name__ == '__main__':
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmminit = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
   
    word, state, word_10, state_10, word_100, state_100, word_100, state, word_1000, state_1000, word_10000, state_10000,= extract_data(train_input)
    words_dict = extract_data_in_dict(index_to_word)
    tags_dict = extract_data_in_dict(index_to_tag)
    write_file("hmminit_10.txt",init_matrix(state_10, tags_dict))
    write_file("hmmemit_10.txt",emission_matrix(word_10, state_10, words_dict, tags_dict))
    write_file("hmmtrans_10.txt",transition_matrix(state_10, tags_dict))
    write_file("hmminit_100.txt",init_matrix(state_100, tags_dict))
    write_file("hmmemit_100.txt",emission_matrix(word_100, state_100, words_dict, tags_dict))
    write_file("hmmtrans_100.txt",transition_matrix(state_100, tags_dict))
    write_file("hmminit_1000.txt",init_matrix(state_1000, tags_dict))
    write_file("hmmemit_1000.txt",emission_matrix(word_1000, state_1000, words_dict, tags_dict))
    write_file("hmmtrans_1000.txt",transition_matrix(state_1000, tags_dict))
    write_file("hmminit_10000.txt",init_matrix(state_10000, tags_dict))
    write_file("hmmemit_10000.txt",emission_matrix(word_10000, state_10000, words_dict, tags_dict))
    write_file("hmmtrans_10000.txt",transition_matrix(state_10000, tags_dict))


