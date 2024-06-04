#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


with open('train', 'r') as file:
   lines = file.readlines()


# In[3]:


for i in range(len(lines)):
    lines[i] = lines[i].split()


# # TASK 1

# In[4]:


import torch
import torch.nn as nn
import torch.optim as optim

class BLSTM(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, dropout):
        super(BLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size=hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden1tag = nn.Linear(hidden_dim, hidden_dim // 2)
        self.hidden2tag = nn.Linear(hidden_dim // 2, len(tag_to_ix))
        self.dropout = nn.Dropout(p=0.33)
        self.activation = nn.ELU()

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden1tag(lstm_out)
        tag_scores = self.activation(tag_space)
        tag_space1 = self.hidden2tag(tag_scores)
        return tag_space1


# In[5]:


# Step 1: Convert the training data to a list of sentences and tags
sentences = []
tags = []
current_sentence = []
current_tags = []
for line in lines:
    if len(line) == 0:
        sentences.append(current_sentence)
        tags.append(current_tags)
        current_sentence = []
        current_tags = []
    else:
        current_sentence.append(line[1])
        current_tags.append(line[2])
        
# Step 2: Convert the sentences and tags to PyTorch tensors
word_to_ix = {}
tag_to_ix = {"O": 0, "B-MISC": 1, "I-MISC": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6, "B-PER": 7, "I-PER": 8}
for sentence in sentences:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
            
X = []
for sentence in sentences:
    sentence_indices = [word_to_ix[word] for word in sentence]
    X.append(torch.tensor(sentence_indices, dtype=torch.long))


# In[6]:


if '<UNK>' not in word_to_ix:
    word_to_ix['<UNK>'] = len(word_to_ix)


# In[7]:


y = []
for tags_for_sentence in tags:
    tag_indices = [tag_to_ix[tag] for tag in tags_for_sentence]
    y.append(torch.tensor(tag_indices, dtype=torch.long))


# In[8]:


# Define hyperparameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 128
DROPOUT = 0.33
LEARNING_RATE = 0.1
NUM_EPOCHS = 20

# Instantiate the model
model1 = BLSTM(vocab_size=len(word_to_ix), tag_to_ix=tag_to_ix, 
              embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model1.parameters(), lr=LEARNING_RATE)

# Train the model
for epoch in range(NUM_EPOCHS):
    for i in range(len(X)):
        sentence = X[i]
        tags = y[i]
        
        # Clear accumulated gradients
        model1.zero_grad()
        
        # Forward pass
        tag_scores = model1(sentence)
        
        # Calculate loss and perform backpropagation
        loss = loss_function(tag_scores, tags)
        loss.backward()
        optimizer.step()

    # Print epoch and loss
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, NUM_EPOCHS, loss.item()))


# In[9]:


with open('dev', 'r') as file:
   dev_lines = file.readlines()
for i in range(len(dev_lines)):
    dev_lines[i] = dev_lines[i].split()


# In[10]:


# Step 1: Convert the dev_lines data to a list of sentences
dev_sentences = []
current_sentence = []
for line in dev_lines:
    if len(line) == 0:
        dev_sentences.append(current_sentence)
        current_sentence = []
    else:
        current_sentence.append(line[1])
dev_sentences.append(current_sentence)

# Step 2: Convert the dev_sentences to PyTorch tensors
dev_X = []
for sentence in dev_sentences:
    sentence_indices = [word_to_ix.get(word, word_to_ix['<UNK>']) for word in sentence]
    dev_X.append(torch.tensor(sentence_indices, dtype=torch.long))

# Step 3: Pass each dev sentence tensor through the model to get the predicted tag scores
model1.eval()
with torch.no_grad():
    dev_tag_scores = []
    for sentence in dev_X:
        tag_scores = model1(sentence)
        dev_tag_scores.append(tag_scores)

# Step 4: Get the predicted tags for each sentence using the tag_to_ix dictionary
dev_predicted_tags = []
for tag_scores in dev_tag_scores:
    _, predicted_tags = torch.max(tag_scores, dim=1)
    dev_predicted_tags.append([list(tag_to_ix.keys())[i] for i in predicted_tags.tolist()])


# In[11]:


total = 0
correct = 0

for pred_tags, line in zip(dev_predicted_tags, dev_lines):
    if len(line) > 0:
        total += 1
        if pred_tags[-1] == line[-1]:
            correct += 1

accuracy = correct / total
print(accuracy)


# In[12]:


from sklearn.metrics import classification_report

# Get the true tags from dev_lines
dev_true_tags = []
current_tags = []
for line in dev_lines:
    if len(line) == 0:
        dev_true_tags.append(current_tags)
        current_tags = []
    else:
        current_tags.append(line[2])
dev_true_tags.append(current_tags)

# Flatten the predicted tags and true tags lists
dev_predicted_flat = [tag for tags in dev_predicted_tags for tag in tags]
dev_true_flat = [tag for tags in dev_true_tags for tag in tags]

# Print the classification report
print(classification_report(dev_true_flat, dev_predicted_flat))


# In[14]:


from sklearn.metrics import precision_score, recall_score, f1_score

# Extract ground truth tags from dev_lines
dev_y = []
current_sentence = []
for line in dev_lines:
    if len(line) == 0:
        dev_y.append(current_sentence)
        current_sentence = []
    else:
        current_sentence.append(line[2])
dev_y.append(current_sentence)

# Flatten dev_predicted_tags and dev_y
dev_predicted_tags_flat = [tag for sentence_tags in dev_predicted_tags for tag in sentence_tags]
dev_y_flat = [tag for sentence_tags in dev_y for tag in sentence_tags]

# Compute precision, recall, and f1 score
precision = precision_score(dev_y_flat, dev_predicted_tags_flat, average='macro')
recall = recall_score(dev_y_flat, dev_predicted_tags_flat, average='macro')
f1 = f1_score(dev_y_flat, dev_predicted_tags_flat, average='macro')

print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 score: {f1:.3f}")


# In[15]:


with open('dev1.out', 'w') as f:
    i = 0
    for index in range(len(dev_lines)):
        line = dev_lines[index]
        if not line:
            f.write('\n')
        else:
            f.write(f"{line[0]} {line[1]} {dev_predicted_tags_flat[i]}\n")
            i += 1


# In[16]:


with open('test', 'r') as file:
   test_lines = file.readlines()


# In[17]:


for i in range(len(test_lines)):
    test_lines[i] = test_lines[i].split()


# In[18]:


# Step 1: Convert the dev_lines data to a list of sentences
test_sentences = []
curr_sentence = []
for line in test_lines:
    if len(line) == 0:
        test_sentences.append(curr_sentence)
        curr_sentence = []
    else:
        curr_sentence.append(line[1])
test_sentences.append(curr_sentence)

# Step 2: Convert the dev_sentences to PyTorch tensors
test_X = []
for sentence in test_sentences:
    sentence_indices = [word_to_ix.get(word, word_to_ix['<UNK>']) for word in sentence]
    test_X.append(torch.tensor(sentence_indices, dtype=torch.long))

# Step 3: Pass each dev sentence tensor through the model to get the predicted tag scores
model1.eval()
with torch.no_grad():
    test_tag_scores = []
    for sentence in test_X:
        tag_scores = model1(sentence)
        test_tag_scores.append(tag_scores)

# Step 4: Get the predicted tags for each sentence using the tag_to_ix dictionary
test_predicted_tags = []
for tag_scores in test_tag_scores:
    _, predicted_tags = torch.max(tag_scores, dim=1)
    test_predicted_tags.append([list(tag_to_ix.keys())[i] for i in predicted_tags.tolist()])


# In[19]:


test_predicted_tags_flat = [tag for sentence_tags in test_predicted_tags for tag in sentence_tags]


# In[20]:


with open('test1.out', 'w') as f:
    i = 0
    for index in range(len(test_lines)):
        line = test_lines[index]
        if not line:
            f.write('\n')
        else:
            f.write(f"{line[0]} {line[1]} {test_predicted_tags_flat[i]}\n")
            i += 1


# In[21]:


torch.save(model1, 'blstm1.pt')


# # TASK 2

# In[ ]:


import numpy as np

word_to_vec = {}
with open('glove.6B.100d', 'r') as f:
    for line in f:
        parts = line.split()
        word = parts[0]
        vec = np.array(parts[1:], dtype=np.float32)
        word_to_vec[word] = vec


# In[ ]:


if '<PAD>' not in word_to_ix:
    word_to_ix['<PAD>'] = len(word_to_ix)


# In[ ]:


# Define embedding matrix
embedding_dim = 100
num_words = len(word_to_ix)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_to_ix.items():
    if word.lower() in word_to_vec:
        embedding_matrix[i] = word_to_vec[word.lower()]
    else:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))


# In[ ]:


class GloveBLSTM(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim, hidden_dim, dropout, word_embeddings):
        super(GloveBLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = word_embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, num_layers=1, bidirectional=True, dropout=dropout, batch_first=True)
        self.hidden1tag = nn.Linear(hidden_dim, hidden_dim//2)
        self.dropout = nn.Dropout(p=0.33)
        self.activation = nn.ELU()
        self.hidden2tag = nn.Linear(hidden_dim//2, tag_size)

    def forward(self, sentence):
        g_embeddings = []
        for word in sentence:
            if word.lower() in self.word_embeddings:
                g_embeddings.append(self.word_embeddings[word.lower()])
            else:
                g_embeddings.append(np.random.normal(scale=0.6, size=self.embedding_dim))
        g_embeddings = torch.tensor(g_embeddings)
        g_embeddings = g_embeddings.type(torch.float32)
        g_embeddings = g_embeddings.unsqueeze(0)
        lstm_out, _ = self.lstm(g_embeddings)
        lstm_out = lstm_out.squeeze(0)
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden1tag(lstm_out)
        tag_scores = self.activation(tag_space)
        tag_scores_final = self.hidden2tag(tag_scores)
        return tag_scores_final


# In[ ]:


EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 128
DROPOUT = 0.33
NUM_EPOCHS = 20
VOCAB_SIZE = len(word_to_ix)
TAG_SIZE = len(tag_to_ix)

model2 = GloveBLSTM(VOCAB_SIZE, TAG_SIZE, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT, word_to_vec)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model2.parameters(), lr=0.1)

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for i in range(len(sentences)):
        sentence = sentences[i]
        tags = y[i]
        model2.zero_grad()
        outputs = model2(sentence)
        loss = loss_function(outputs, tags)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} loss: {total_loss/len(sentences):.4f}")


# In[ ]:


# Step 1: Convert the dev_lines data to a list of sentences
dev_sentences = []
current_sentence = []
for line in dev_lines:
    if len(line) == 0:
        dev_sentences.append(current_sentence)
        current_sentence = []
    else:
        current_sentence.append(line[1])
dev_sentences.append(current_sentence)

# Step 2: Convert the dev_sentences to PyTorch tensors
dev_X = []
for sentence in dev_sentences:
    sentence_indices = [word_to_ix.get(word, word_to_ix['<UNK>']) for word in sentence]
    dev_X.append(torch.tensor(sentence_indices, dtype=torch.long))

# Step 3: Pass each dev sentence tensor through the model to get the predicted tag scores
model2.eval()
with torch.no_grad():
    dev_tag_scores = []
    for sentence in dev_sentences:
        tag_scores = model2(sentence)
        dev_tag_scores.append(tag_scores)

# Step 4: Get the predicted tags for each sentence using the tag_to_ix dictionary
dev_predicted_tags = []
for tag_scores in dev_tag_scores:
    _, predicted_tags = torch.max(tag_scores, dim=1)
    dev_predicted_tags.append([list(tag_to_ix.keys())[i] for i in predicted_tags.tolist()])


# In[ ]:


total = 0
correct = 0

for pred_tags, line in zip(dev_predicted_tags, dev_lines):
    if len(line) > 0:
        total += 1
        if pred_tags[-1] == line[-1]:
            correct += 1

accuracy = correct / total
print(accuracy)


# In[ ]:


from sklearn.metrics import classification_report

# Get the true tags from dev_lines
dev_true_tags = []
current_tags = []
for line in dev_lines:
    if len(line) == 0:
        dev_true_tags.append(current_tags)
        current_tags = []
    else:
        current_tags.append(line[2])
dev_true_tags.append(current_tags)

# Flatten the predicted tags and true tags lists
dev_predicted_flat = [tag for tags in dev_predicted_tags for tag in tags]
dev_true_flat = [tag for tags in dev_true_tags for tag in tags]

# Print the classification report
print(classification_report(dev_true_flat, dev_predicted_flat))


# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score

# Extract ground truth tags from dev_lines
dev_y = []
current_sentence = []
for line in dev_lines:
    if len(line) == 0:
        dev_y.append(current_sentence)
        current_sentence = []
    else:
        current_sentence.append(line[2])
dev_y.append(current_sentence)

# Flatten dev_predicted_tags and dev_y
dev_predicted_tags_flat = [tag for sentence_tags in dev_predicted_tags for tag in sentence_tags]
dev_y_flat = [tag for sentence_tags in dev_y for tag in sentence_tags]

# Compute precision, recall, and f1 score
precision = precision_score(dev_y_flat, dev_predicted_tags_flat, average='macro')
recall = recall_score(dev_y_flat, dev_predicted_tags_flat, average='macro')
f1 = f1_score(dev_y_flat, dev_predicted_tags_flat, average='macro')

print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 score: {f1:.3f}")


# In[ ]:


with open('dev2.out', 'w') as f:
    i = 0
    for index in range(len(dev_lines)):
        line = dev_lines[index]
        if not line:
            f.write('\n')
        else:
            f.write(f"{line[0]} {line[1]} {dev_predicted_tags_flat[i]}\n")
            i += 1


# In[ ]:


# Step 1: Convert the dev_lines data to a list of sentences
test_sentences = []
curr_sentence = []
for line in test_lines:
    if len(line) == 0:
        test_sentences.append(curr_sentence)
        curr_sentence = []
    else:
        curr_sentence.append(line[1])
test_sentences.append(curr_sentence)

# Step 2: Convert the dev_sentences to PyTorch tensors
test_X = []
for sentence in test_sentences:
    sentence_indices = [word_to_ix.get(word, word_to_ix['<UNK>']) for word in sentence]
    test_X.append(torch.tensor(sentence_indices, dtype=torch.long))

# Step 3: Pass each dev sentence tensor through the model to get the predicted tag scores
model2.eval()
with torch.no_grad():
    test_tag_scores = []
    for sentence in test_sentences:
        tag_scores = model2(sentence)
        test_tag_scores.append(tag_scores)

# Step 4: Get the predicted tags for each sentence using the tag_to_ix dictionary
test_predicted_tags = []
for tag_scores in test_tag_scores:
    _, predicted_tags = torch.max(tag_scores, dim=1)
    test_predicted_tags.append([list(tag_to_ix.keys())[i] for i in predicted_tags.tolist()])


# In[ ]:


test_predicted_tags_flat = [tag for sentence_tags in test_predicted_tags for tag in sentence_tags]


# In[ ]:


with open('test2.out', 'w') as f:
    i = 0
    for index in range(len(test_lines)):
        line = test_lines[index]
        if not line:
            f.write('\n')
        else:
            f.write(f"{line[0]} {line[1]} {test_predicted_tags_flat[i]}\n")
            i += 1


# In[ ]:


torch.save(model2, 'blstm2.pt')


# # -------------------------------------------------------------------------------------------
