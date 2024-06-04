#!/usr/bin/env python
# coding: utf-8

# # Task 1 - Vocabulary

# ## Reading train data

# In[1]:


with open('train', 'r') as file:
   lines_for_vocab = file.readlines()


# In[2]:


for i in range(len(lines_for_vocab)):
    lines_for_vocab[i] = lines_for_vocab[i].split()


# In[3]:


lines_for_vocab = [x for x in lines_for_vocab if x != []]


# ## Taking the words from the train data.

# In[4]:


list_of_words = []
for i in range(0, len(lines_for_vocab)):
    list_of_words.append(lines_for_vocab[i][1])


# ## Finding the frequency of the words using a dictionary

# In[5]:


freq_for_vocab = {}
i=0
for element in list_of_words:
    if element in freq_for_vocab:
        freq_for_vocab[element] += 1
    else:
        freq_for_vocab[element] = 1


# ## Using a threshold value and taking only those equal and above the theshold into a new dictionary.

# In[6]:


threshold_freq = dict((k, v) for k, v in freq_for_vocab.items() if v >= 3)


# ## Those below the threshold are tagged as unknown represented by '\<unk>'

# In[7]:


unknown_token = "<unk>"
new_text_vocab = []
for word in list_of_words:
    if word not in threshold_freq:
        new_text_vocab.append(unknown_token)
    else:
        new_text_vocab.append(word)


# ## FInding the frequency of all the words including '\<unk>'

# In[8]:


prop_freq_vocab = {}
i=0
for element in new_text_vocab:
    if element in prop_freq_vocab:
        prop_freq_vocab[element] += 1
    else:
        prop_freq_vocab[element] = 1


# In[9]:


unknown = {k : v for k, v in prop_freq_vocab.items() if k == '<unk>' }


# In[10]:


del prop_freq_vocab['<unk>']


# ## Sorting the dictionary based on the occurances.

# In[11]:


threshold_sorted_dict = {k : v for k, v in sorted(prop_freq_vocab.items(), key=lambda item: item[1], reverse=True)}


# ## Writing the dictionary into a file 'vocab.txt'

# In[12]:


with open("vocab.txt", "w") as f:
    i=0
    for key, value in unknown.items():
        f.write(str(key)+'\t'+str(i)+'\t'+str(value)+'\n')
        i+=1
    for key, value in threshold_sorted_dict.items():
        f.write(str(key)+'\t'+str(i)+'\t'+str(value)+'\n')
        i=i+1 


# # Task 2 - Model Learning

# In[13]:


# import main_code
with open('train', 'r') as file:
   lines = file.readlines()


# In[14]:


for i in range(len(lines)):
    lines[i] = lines[i].split()


# In[15]:


def update_array(lines):
    updated_lines = []
    for sub_list in lines:
        if not sub_list:
          updated_lines.append(["", "", "START"])
        else:
          updated_lines.append(sub_list)
    return updated_lines


# In[16]:


lines = update_array(lines)


# In[17]:


lst = [sublist[2] for sublist in lines]


# In[18]:


tag_freq = {}
for element in lst:
    if element in tag_freq:
        tag_freq[element] += 1
    else:
        tag_freq[element] = 1


# In[19]:


word_lst = [sub[1] for sub in lines]


# In[20]:


word_freq = {}
for element in word_lst:
    if element in word_freq:
        word_freq[element] += 1
    else:
        word_freq[element] = 1


# In[21]:


senil = lines


# In[22]:


for sub in senil:
    if "\/" in sub[1]:
        senil.remove(sub)


# ## Probabilty Calculation

# In[23]:


transition = {}
emission = {}
# transition_prob = {}
# emission_prob = {}

for count,i in enumerate(senil):
    if i[2] == '.':
        nxt = 'START'
        
    if count == len(senil) - 1:
        break
    _, word, tag = i
    nxt = senil[count+1][2]
    
    key = (nxt, tag)

    if key in transition:
        transition[key] += 1
    else:
        transition[key] = 1
        
    if(key[0] == 'START'):
        del transition[key]
        
        
    key2 = (word, tag)
    
    if key2 in emission:
        emission[key2] += 1
    else:
        emission[key2] = 1


# # Emission Probabilty 

# In[24]:


emission_prob = {}
for key in emission:
    emission_prob[key] = emission[key]/tag_freq[key[1]]


# # Transition Probability

# In[25]:


transition_prob = {}
for key in transition:
    transition_prob[key] = transition[key]/tag_freq[key[1]]


# # Printing the Transition and Emission probabilities into hmm.json

# In[26]:


import json

transition_prob_str = {}
emission_prob_str = {}
for key, value in transition_prob.items():
    transition_prob_str[str(key)] = value

for key, value in emission_prob.items():
    emission_prob_str[str(key)] = value

params = {"TRANSITION": dict(transition_prob_str), "EMISSION": dict(emission_prob_str)}
with open("hmm.json", "w") as file:
    json.dump(params, file)


# # Task 3 - Greedy Decoding

# In[27]:


def greedy_decoding_v3(dev_data, transition_prob, emission_prob):
    result = []
    for i in range(len(dev_data)):
        if len(dev_data[i]) > 0:
            word = dev_data[i][1]
            max_prob = float('-inf')
            prev_tag = result[-1][2] if result else "START"
            best_tag = "NN"
        else:
            prev_tag = "START"
            
        for tag in tag_freq.keys():
            tk = 0
            ek = 0
            if (tag, prev_tag) in transition_prob.keys():
                tk = transition_prob[(tag, prev_tag)]
            if (word, tag) in emission_prob.keys():
                ek = emission_prob[(word, tag)]
            prob = tk * ek
            if prob > max_prob:
                max_prob = prob
                best_tag = tag
        if len(dev_data[i]) > 0:
            result.append([dev_data[i][0], dev_data[i][1], best_tag])
    return result



# ## Reading the dev data

# In[28]:


# import main_code
with open('dev', 'r') as file:
   dev_lines = file.readlines()


# In[29]:


for i in range(len(dev_lines)):
    dev_lines[i] = dev_lines[i].split()


# ## Applying the Greedy Decoding function on the entire dev data

# In[30]:


pos_tags_fulldata = greedy_decoding_v3(dev_lines, transition_prob, emission_prob)


# ## Accuracy for the entire dev data

# In[31]:


correct = 0
j=0
for i in range(len(dev_lines)):
    #print(dev_data[i])
    if len(dev_lines[i]) > 0:
        if pos_tags_fulldata[j][2] == dev_lines[i][2]:
            correct += 1
        j+=1
            
accuracy = correct / len(pos_tags_fulldata)
print("Greedy Decoding Accuracy:", accuracy)


# ## Reading the test data

# In[32]:


with open('test', 'r') as file:
   test_lines = file.readlines()


# In[33]:


for i in range(len(test_lines)):
    test_lines[i] = test_lines[i].split()


# ## Applying the greedy decoding function to the test data and writing the tag predictions for the words in test data to the file greedy.out

# In[34]:


import json

pos_tags_fulldata_test = greedy_decoding_v3(test_lines, transition_prob, emission_prob)
with open("greedy.out", "w") as f:
    for i in range(len(pos_tags_fulldata_test)):
        if i > 0 and pos_tags_fulldata_test[i][0] == '1':
            f.write("\n")
        f.write(pos_tags_fulldata_test[i][0]+'\t'+pos_tags_fulldata_test[i][1]+'\t'+pos_tags_fulldata_test[i][2]+'\n')


# # Task 4 - Viterbi Decoding

# ## A function that takes the data, and transition and emission probabilities and implement the Viterbi Algorithm (Dynamic Programming)

# In[35]:


def viterbiupd(dev_data,transition_prob,emission_prob):
    dp = [{}]
    final_dp = []
    backtrack = [{}]
    index = 0
    count = 0
    for seq in dev_data:
        if len(seq) > 0:
            word = seq[1]
            idx = seq[0]
            if idx == '1':
                for tag in tag_freq.keys():
                    if (tag,'START') in transition_prob.keys():
                        t = transition_prob[(tag,'START')]
                        if (word,tag) in emission_prob.keys():
                            e = emission_prob[(word,tag)]
                            dp[index][tag] = e*t
                            backtrack[index][tag] = 'NN'
                        else:
                            dp[index][tag] = 0.000000001
                    else:
                        dp[index][tag] = 0.000000001
                index+=1
            else:
                dp.append({})
                backtrack.append({})
                for tag in tag_freq.keys():
                    dp[index][tag] = 0
                    for key in tag_freq.keys():
                        t = 0.000000001
                        if (tag,key) in transition_prob.keys():
                            t = transition_prob[(tag,key)]
                        e = 0.000000001
                        if (word,tag) in emission_prob.keys():
                            e = emission_prob[(word,tag)]

    #                     dp[index][tag] = max(dp[index-1][tag]*e*t,dp[index][tag])
                        if dp[index-1][key]*e*t > dp[index][tag]:
                            dp[index][tag] = dp[index-1][key]*e*t
                            backtrack[index][tag] = key
                index+=1

        else:
#             dp = dp[:-1]
            tag_seq = []
            final_tag = max(dp[-1], key=dp[-1].get)
            tag_seq.append(final_tag)
            for i in range(len(dp)-1, 0, -1):
                if final_tag in backtrack[i].keys():
                    final_tag = backtrack[i][final_tag]
                    tag_seq.append(final_tag)
                else:
                    tag_seq.append('.')

            # Reverse the tag sequence to obtain the correct order
            tag_seq = tag_seq[::-1]
            final_dp += tag_seq
            dp = [{}]
            index = 0
            backtrack = [{}]
    
    return final_dp


# In[36]:


dev_lines.append([])


# ## All the tags returned after the Viterbi decoding is stored in a list.

# In[37]:


all_tags = []
all_tags = viterbiupd(dev_lines,transition_prob,emission_prob)


# ## Accuracy of the Viterbi algorithm against the dev data.

# In[38]:


idx=0
cnt=0
# for seq in dev_lines:
for i in range(len(dev_lines)):
    if len(dev_lines[i]) > 0:
        if dev_lines[i][2] == all_tags[idx]:
            cnt+=1
        idx+=1
print("Viterbi Decoding Accuracy:", cnt/idx)


# ## Predicting the tags for the test data.

# In[39]:


test_lines.append([])


# In[40]:


all_test_tags = []
all_test_tags = viterbiupd(test_lines,transition_prob,emission_prob)


# ## Writing the predicted tags along with the test data into a file.

# In[41]:


test_list = []
idx = 0
for i in range(len(test_lines)):
    if len(test_lines[i]) > 0:
        test_list.append([test_lines[i][0],test_lines[i][1],all_test_tags[idx]])
        idx+=1
        
with open("viterbi.out", "w") as f:
    for i in range(len(test_list)):
        if i > 0 and test_list[i][0] == '1':
            f.write("\n")
        f.write(test_list[i][0]+'\t'+test_list[i][1]+'\t'+test_list[i][2]+'\n')

