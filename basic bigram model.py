import torch
import matplotlib.pyplot as plt
words = open('names.txt', 'r').read().splitlines() # read text file of names
'''for word in words[:10]:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        print(ch1, ch2)
    print("\n")''' # how bigram model reads text to learn
chars = "".join(words)
stoi = {s:i+1 for i, s in enumerate(sorted(set(chars)))} # create string to integer dictonary for indexing
stoi['.'] = 0 # adding start and end token
itos = {i:s for s, i in stoi.items()} # create integer to string dixtonary for later use
table = torch.ones((27, 27), dtype=torch.int32) # model smoothing, creating table of 27 x 27 which is alphabet + start/end token. reason for tables of ones to make bigram combinations not in text possible
for word in words: # iterating through all words to collect combination counts
    chs = ['.'] + list(word) + ['.'] # adding start/end token
    for ch1, ch2 in zip(chs, chs[1:]):
        index1 = stoi[ch1] # getting index in dictionary of first character
        index2 = stoi[ch2] # getting index in dictionary of second character
        table[index1, index2] += 1 # incrementing combination
prob = table.float() # copying table to new table to calculate probabilities and converting to float to work with decimals
for i in range(27):
    for j in range(27):
        prob[i, j] /= table.sum(axis = 1)[i] # calculate probability given row
plt.figure(figsize=(16,16))
plt.imshow(table, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va ="bottom", color="gray")
        plt.text(j, i, f"{prob[i, j].item():.3f}", ha="center", va ="top", color="gray")
plt.axis('off') # displaying table of probabilities
# To generate text given bigram model:
g = torch.Generator().manual_seed(2147483647) 
for i in range(20):
    word = ""
    index = 0
    while True:
        index = torch.multinomial(prob[index], num_samples=1, replacement = True, generator=g).item() # generating index of character through a "random" generator given probabilities
        if index == 0:
            break # breaking loop if end token is reached
        word += itos[index] # adding character of index to word
    print(word)
# to check likelihood of certain words (optimally 0):
checks = ["amr"]
likelihood = 0.0
n = 0
for word in checks:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        index1 = stoi[ch1]
        index2 = stoi[ch2]
        likelihood += torch.log(prob[index1, index2]).item()
        n += 1
likelihood = -likelihood 
print(likelihood/n)
# to check how good model is usiing likelihood loss function (optimally 0):
likelihood = 0.0
n = 0
for word in words:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        index1 = stoi[ch1]
        index2 = stoi[ch2]
        likelihood += torch.log(prob[index1, index2]).item() # using log to scale cost function so liklihood is close to 0 when there are high amounts of approximately 100% predictions in certain characters and close to 0% prediction i=for other characters so model is sure of what to do next
        n += 1
likelihood = -likelihood # using negative log liklihood to minimize cost function
print(likelihood/n)

