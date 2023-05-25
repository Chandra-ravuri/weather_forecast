

# In[1]:


import numpy as np
from hmmlearn import hmm
from matplotlib import cm, pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import pandas as pd
from pandas import DataFrame, Series


# In[2]:


state_seq = []
observation_seq = []
f = open('weather-test1-1000.txt','r')
line = f.readline()
while line:
    line=line.rstrip().split(',')
    
    # encoding: sunny:0, rainy:1, foggy:2
    if line[0] == 'sunny':
        state_seq.append(0)
    elif line[0] == 'rainy':
        state_seq.append(1)
    else:
        state_seq.append(2)
    
    # encoding: yes:0, no:1
    if line[1] == 'yes':
        observation_seq.append(0)
    else:
        observation_seq.append(1)
    line = f.readline()


# In[3]:


# hidden states
z_decode = {0:'sunny',1:'rainy',2:'foggy'}
z_encode = {'sunny':0,'rainy':1,'foggy':2}

# possible observations
# 'yes' for unbrella is observed
x_decode = {0:'yes',1:'no'}
x_encode = {'yes':0,'no':1}

# start probability vector for hidden states is given in the instruction
# P(sunny) = 0.5, P(rainy) = 0.25, P(foggy) = 0.25
start_prob = [0.5, 0.25, 0.25]


# In[4]:


Series(state_seq).value_counts().plot(kind='bar')


# In[5]:


Series(observation_seq).value_counts().plot(kind='bar')


# In[6]:


df = DataFrame({'hidden':state_seq,'observation':observation_seq})


# In[7]:


df.groupby(['hidden','observation'])['observation'].count()


# In[8]:


# x-axis: 1000 days, y-axis: with umbrella(0), without umbrella(1)
num_hidden_states = 3
fig, axs = plt.subplots(num_hidden_states, sharex=True, sharey=True,figsize=(15,5))
colours = cm.rainbow(np.linspace(0, 1, num_hidden_states))
date_range = np.array(range(1,len(state_seq)+1))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = np.array(state_seq) == i
    ax.plot(date_range[mask], np.array(observation_seq)[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i)+': '+z_decode[i])

    ax.grid(True)

plt.show()


# In[9]:


fig = plt.figure(figsize=(5,5)) 
fig1 = fig.add_subplot(111)
fig1.scatter(range(1,len(state_seq)+1), state_seq, s=480, c=observation_seq, alpha=0.1, cmap=plt.cm.rainbow)
fig1.set_xlabel('Days', fontsize=20)
fig1.set_ylabel('Weather', fontsize=20)


# In[13]:


# 1. Construct HMM manually
# Compute hidden state transitional probability matrix


# In[10]:


# E.G. {sunny: {sunny: 11, rainy: 20, foggy: 5}}
state_occurrence_dic = {}
for i in range(1,len(state_seq)):
    if state_seq[i-1] not in state_occurrence_dic:
        state_occurrence_dic[state_seq[i-1]] = {state_seq[i]:1}
    else:
        if state_seq[i] not in state_occurrence_dic[state_seq[i-1]]:
            state_occurrence_dic[state_seq[i-1]][state_seq[i]] = 1
        else:
            state_occurrence_dic[state_seq[i-1]][state_seq[i]] += 1


# In[11]:


state_occurrence_dic


# In[14]:


sunny = np.array(state_occurrence_dic[0].values())/float(sum(state_occurrence_dic[0].values()))
rainy = np.array(state_occurrence_dic[1].values())/float(sum(state_occurrence_dic[1].values()))
foggy = np.array(state_occurrence_dic[2].values())/float(sum(state_occurrence_dic[2].values()))
transition_matrix = np.vstack((sunny,rainy,foggy))
transition_matrix


# In[21]:


# E.G. {sunny: {yes: 11, no: 20}}
ob_occurrence_dic = {}
for i in range(len(state_seq)):
    if state_seq[i] not in ob_occurrence_dic:
        ob_occurrence_dic[state_seq[i]] = {observation_seq[i]:1}
    else:
        if observation_seq[i] not in ob_occurrence_dic[state_seq[i]]:
            ob_occurrence_dic[state_seq[i]][observation_seq[i]] = 1
        else:
            ob_occurrence_dic[state_seq[i]][observation_seq[i]] += 1


# In[22]:


ob_occurrence_dic


# In[23]:


sunny = np.array(ob_occurrence_dic[0].values())/float(sum(ob_occurrence_dic[0].values()))
rainy = np.array(ob_occurrence_dic[1].values())/float(sum(ob_occurrence_dic[1].values()))
foggy = np.array(ob_occurrence_dic[2].values())/float(sum(ob_occurrence_dic[2].values()))
emission_matrix = np.vstack((sunny,rainy,foggy))
emission_matrix


# In[24]:


# construct HMM model using the calculated parameters


# In[15]:


# E.G. {sunny: {yes: 11, no: 20}}
ob_occurrence_dic = {}
for i in range(len(state_seq)):
    if state_seq[i] not in ob_occurrence_dic:
        ob_occurrence_dic[state_seq[i]] = {observation_seq[i]:1}
    else:
        if observation_seq[i] not in ob_occurrence_dic[state_seq[i]]:
            ob_occurrence_dic[state_seq[i]][observation_seq[i]] = 1
        else:
            ob_occurrence_dic[state_seq[i]][observation_seq[i]] += 1


# In[16]:


ob_occurrence_dic


# In[17]:


sunny = np.array(ob_occurrence_dic[0].values())/float(sum(ob_occurrence_dic[0].values()))
rainy = np.array(ob_occurrence_dic[1].values())/float(sum(ob_occurrence_dic[1].values()))
foggy = np.array(ob_occurrence_dic[2].values())/float(sum(ob_occurrence_dic[2].values()))
emission_matrix = np.vstack((sunny,rainy,foggy))
emission_matrix


# In[18]:


# n_components=3 means three hidden states [0, 1, 2]
model_manual = hmm.MultinomialHMM(n_components=3)

# start probabilities for each hidden state
model_manual.startprob_ = start_prob

# transitional matrix for pair-wise hidden state
model_manual.transmat_ = transition_matrix

# emission probability matrix for [hidden state,observation]
model_manual.emissionprob_ = emission_matrix


# In[19]:


# probability of the model
print 'log probability under the model: ',model_manual.score(observation_seq)
print 'probability under the model: ',np.exp(model_manual.score(observation_seq))


# In[20]:


#test_observation = ['no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes']
test_observation = ['no', 'yes', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no']
test_observation_encode = [x_encode[i] for i in test_observation]
test_observation_encode = np.atleast_2d(test_observation_encode).T
test_observation_encode


# In[22]:


log_prob, pred_sequence = model_manual.decode(test_observation_encode, algorithm="viterbi")
for i in zip(test_observation,map(lambda x:z_decode[x],pred_sequence)):
    print("Observation on Umbrella:",i[0], ", Predicted Weather:",i[1])


# In[23]:


all_observation_encode = np.atleast_2d(observation_seq).T
_, pred_sequence_all = model_manual.decode(all_observation_encode, algorithm="viterbi")
# x-axis: 1000 days, y-axis: with umbrella(0), without umbrella(1)
num_hidden_states = 3
fig, axs = plt.subplots(num_hidden_states, sharex=True, sharey=True,figsize=(15,5))
colours = cm.rainbow(np.linspace(0, 1, num_hidden_states))
date_range = np.array(range(1,len(state_seq)+1))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = np.array(pred_sequence_all) == i
    ax.plot(date_range[mask], np.array(observation_seq)[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i)+': '+z_decode[i])

    ax.grid(True)

plt.show()


# In[25]:


sample_observation, sample_hidden = model_manual.sample(10)

# decode sample observation into yes/no
sample_observation = sample_observation.reshape(10)
sample_observation = map(lambda i:x_decode[i],sample_observation)

# decode sample weather into sunny/rainy/foggy
sample_hidden = map(lambda i:z_decode[i],sample_hidden)

for i in range(10):
    print("Observation on Umbrella:",sample_observation[i], ", Weather:",sample_hidden[i])


# In[26]:


# n_components=3 means three hidden states [0, 1, 2]
# n_iter = Maximum number of iterations to perform when fitting using Expectation Maximization
model_fitted = hmm.MultinomialHMM(n_components=3, n_iter=1000)

# since the EM algorithm is a gradient-based optimization method, it will generally get stuck 
#in local optima. You should in general try to run fit with various initializations and select 
# the highest scored model.
model_fitted.fit(observation_seq)


# In[27]:


# probability of the model fitted
print 'log probability under the model: ',model_fitted.score(observation_seq)
print 'probability under the model: ',np.exp(model_fitted.score(observation_seq))


# In[28]:


print 'true transition probability matrix'
print transition_matrix,'\n'

print 'fitted transition probability matrix'
print model_fitted.transmat_


# In[29]:


print 'true emission_matrix:'
print emission_matrix ,'\n'

print 'estimated emission matrix:'
print model_fitted.emissionprob_


# In[30]:


print 'fitted emission probabilities for hidden states'
print model_fitted.startprob_


# In[31]:


# x-axis: 1000 days, y-axis: with umbrella(0), without umbrella(1)
num_hidden_states = 3
fig, axs = plt.subplots(num_hidden_states, sharex=True, sharey=True,figsize=(15,5))
colours = cm.rainbow(np.linspace(0, 1, num_hidden_states))
date_range = np.array(range(1,len(state_seq)+1))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = np.array(pred_sequence_all) == i
    ax.plot(date_range[mask], np.array(observation_seq)[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))

    ax.grid(True)

plt.show()


# In[32]:


z_decode_fitted = z_decode = {0:'sunny',2:'rainy',1:'foggy'}
#test_observation = ['no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes']
test_observation = ['no', 'yes', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no']
test_observation_encode = [x_encode[i] for i in test_observation]
test_observation_encode = np.atleast_2d(test_observation_encode).T
test_observation_encode


# In[33]:


log_prob, pred_sequence = model_manual.decode(test_observation_encode, algorithm="viterbi")
for i in zip(test_observation,map(lambda x:z_decode_fitted[x],pred_sequence)):
    print ("Observation on Umbrella:",i[0], ", Predicted Weather:",i[1])


# In[34]:


sample_observation, sample_hidden = model_fitted.sample(10)

# decode sample observation into yes/no
sample_observation = sample_observation.reshape(10)
sample_observation = map(lambda i:x_decode[i],sample_observation)

# decode sample weather into sunny/rainy/foggy
sample_hidden = map(lambda i:z_decode_fitted[i],sample_hidden)

for i in range(10):
    print ("Observation on Umbrella:",sample_observation[i], ", Weather:",sample_hidden[i])


# In[ ]:




