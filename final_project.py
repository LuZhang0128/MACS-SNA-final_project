# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 08:01:41 2021

@author: Haohan
"""

from build_user_subreddit_history import read_json_list
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from networkx.algorithms import bipartite
import networkx as nx
import random
from community import community_louvain
import matplotlib.cm as cm
from datetime import datetime
from datetime import date
import datetime as dt
import seaborn as sns
import pandas as pd
from scipy import stats

spacy.prefer_gpu()
nlp = spacy.load('en_core_web_trf')

# select variables that we want related to the posts: content, time, author, title, subreddit
def get_contents(sub):
    
    result_dic = {}
    
    filename = "new_data/{1}/{0}/{1}_jsonlists.gz".format(sub, 'posts')
    
    i = 0
    
    for dic in read_json_list(filename):
            result = {}
            if 'selftext' in dic:
                id = dic['id']
                text = dic['selftext']
                subreddit = dic['subreddit']
                created_time = dic['created_utc']
                author = dic['author']
                title = dic['title']
                
                result['id'] = id
                
                result['author'] = author
                
                if text != '[removed]' and text != '':
                    result['text'] = text
                else:
                    result['text'] = title
                
                result['subreddit'] = subreddit
                result['time'] = created_time
                #if author != '[deleted]':
                  #  author_sub_timestamps_dic[author].append((subreddit, created_time))
                  #  sub_author_timestamps_dic[subreddit].append((author, created_time))
                result_dic[i] = result
                i += 1
    
    return result_dic

print('1')
AllWomen = get_contents('AllWomen')
print('2')
asiantwoX = get_contents('asiantwoX')
print('3')
AskFeminists = get_contents('AskFeminists')
print('4')
blackladies = get_contents('blackladies')
print('5')
careerwomen = get_contents('careerwomen')
print('6')
Feminism = get_contents('Feminism')
print('7')
FeMRADebates = get_contents('FeMRADebates')
print('8')
NotHowGirlsWork = get_contents('NotHowGirlsWork')
print('9')
SexPositive = get_contents('SexPositive')
print('10')
transgender = get_contents('transgender')

# use spacy to tokenize the contents
def get_token(dic):
    for i in dic.keys():
        dic[i]['text'] = nlp(dic[i]['text'])

print('1')
get_token(AllWomen)
print('2')
get_token(asiantwoX)
print('3')
get_token(AskFeminists)
print('4')
get_token(blackladies)
print('5')
get_token(careerwomen)
print('6')
get_token(Feminism)
print('7')
get_token(FeMRADebates)
print('8')
get_token(NotHowGirlsWork)
print('9')
get_token(SexPositive)
print('10')
get_token(transgender)



# we only need the proper nouns, nouns, and adjectives
def get_noun(dic):
    for i in dic.keys():
        result = []
        
        for token in dic[i]['text']:
            if token.pos_ == 'PROPN' or token.pos_ == 'NOUN' or token.pos_ == 'ADJ':
                result.append(token.lemma_)
        
        dic[i]['text'] = result

    
print('1')
get_noun(AllWomen)
print('2')
get_noun(asiantwoX)
print('3')
get_noun(AskFeminists)
print('4')
get_noun(blackladies)
print('5')
get_noun(careerwomen)
print('6')
get_noun(Feminism)
print('7')
get_noun(FeMRADebates)
print('8')
get_noun(NotHowGirlsWork)
print('9')
get_noun(SexPositive)
print('10')
get_noun(transgender)    





# combine all the lemmas to a string for each subreddit, because the sklearn tf-idf function only takes this format
def combine_lemmas(dic):
    result = ''
    if dic.keys() is None:
        return result
    for i in dic.keys():
        for lemma in dic[i]['text']:
            result += lemma + ' '
    
    return result

allwomen_str = combine_lemmas(AllWomen)
asiantwox_str = combine_lemmas(asiantwoX)
askfeminists_str = combine_lemmas(AskFeminists)
blackladies_str = combine_lemmas(blackladies)
careerwomen_str = combine_lemmas(careerwomen)
feminism_str = combine_lemmas(Feminism)
femradebates_str = combine_lemmas(FeMRADebates)
nothowgirlswork_str = combine_lemmas(NotHowGirlsWork)
sexpositive_str = combine_lemmas(SexPositive)
transgender_str = combine_lemmas(transgender)




# prepare input for sklearn tf-idf
corpus = [allwomen_str, asiantwox_str, askfeminists_str,
          blackladies_str, careerwomen_str, feminism_str,
          femradebates_str, nothowgirlswork_str, sexpositive_str,
          transgender_str]    

vectorizer = TfidfVectorizer()
result = vectorizer.fit_transform(corpus)  
len(vectorizer.get_feature_names_out())
    
print(result.shape)




# plot sample network, select 0.001 of the network
subreddits = ['r/AllWomen', 'r/asiantwoX', 'r/AskFeminists', 'r/blackladies',
                  'r/careerwomen', 'r/Feminism', 'r/FeMRADebates', 'r/NotHowGirlsWork',
                  'r/SexPositive', 'r/transgender']

    
names = vectorizer.get_feature_names_out()

result_array = result.toarray()


g = nx.Graph()

for row in range(len(result_array)):
    for column in range(len(result_array[row])):
        
        weight =  result_array[row][column]
        if weight > 0:
            
            if random.random() < 0.001:
                g.add_node(f'{subreddits[row]}', bipartite = 0)
                g.add_node(f'{names[column]}', bipartite = 1)
                g.add_edge(f'{subreddits[row]}', f'{names[column]}', weight = weight)


# formating bipartite network plot
top = [node for node in g.nodes() if g.nodes(data = True)[node]['bipartite']==0]
bottom = [node for node in g.nodes() if g.nodes(data = True)[node]['bipartite']==1]

pos = dict()
pos.update( (n, (1, 8*i)) for i, n in enumerate(top) ) # put nodes from X at x=1
pos.update( (n, (2, i)) for i, n in enumerate(bottom) ) # put nodes from Y at x=2

labels = nx.get_edge_attributes(g,'weight')

edges = g.edges()
weights = [70 * g[u][v]['weight'] for u,v in edges]

f = plt.figure()
nx.draw(g, ax = f.add_subplot(), pos = pos, node_size = 10, width = weights, 
        font_size = 5, with_labels = True)

f.savefig("plot.png", dpi=1000)
plt.show()




# get the whole network
g = nx.Graph()

for row in range(len(result_array)):
    for column in range(len(result_array[row])):
        
        weight =  result_array[row][column]
        if weight > 0:
            
                g.add_node(f'{subreddits[row]}', bipartite = 0)
                g.add_node(f'{names[column]}', bipartite = 1)
                g.add_edge(f'{subreddits[row]}', f'{names[column]}', weight = weight)


# get the one model network from the whole network with only the subreddits as nodes
G = bipartite.projected_graph(g,['r/AllWomen', 'r/asiantwoX', 'r/AskFeminists', 'r/blackladies',
                  'r/careerwomen', 'r/Feminism', 'r/FeMRADebates', 'r/NotHowGirlsWork',
                  'r/SexPositive', 'r/transgender'])


nx.draw(G, with_labels = True)



# add weight to the network based on tf-idf
weights = []
for edge in G.edges():
    print(edge)
    
    ind1 = subreddits.index(edge[0])
    ind2 = subreddits.index(edge[1])
    
    weight = 0
    
    for w1 in result_array[ind1]:
        if w1 != 0:
            for w2 in result_array[ind2]:
                if  w2 != 0:
                    weight += w1 + w2
                    
    weights.append(weight)
    

print(weights)

w = []

for i in weights:
    w.append(i/100000)

i = 0
for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = w[i]
    i += 1


# draw network

partition = community_louvain.best_partition(G)

f = plt.figure()

pos = nx.spring_layout(G)

cmap = cm.get_cmap('viridis', max(partition.values()) + 1)


nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=100, cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
f.savefig("plot.png", dpi=1000)
plt.show()
    

# get the author usernames and time of the first comment from the comment dataset
def get_author(sub):
    
    result_dic = {}
    
    filename = "data/{1}/{0}/{1}_jsonlists.gz".format(sub, 'comments')
    
    result = {}
    for dic in read_json_list(filename):
            
            if dic['author'] != '[deleted]':
               
                author = dic['author']
                
                created_time = dic['created_utc']
              
                
                
                if author not in result.keys():
                    result[author] = created_time
 
    return result
    
    
AllWomen_com = get_author('AllWomen')
print('2')
asiantwoX_com = get_author('asiantwoX')
print('3')
AskFeminists_com = get_author('AskFeminists')
print('4')
blackladies_com = get_author('blackladies')
print('5')
careerwomen_com = get_author('careerwomen')
print('6')
Feminism_com = get_author('Feminism')
print('7')
FeMRADebates_com = get_author('FeMRADebates')
print('8')
NotHowGirlsWork_com = get_author('NotHowGirlsWork')
print('9')
SexPositive_com = get_author('SexPositive')
print('10')
transgender_com = get_author('transgender')
    

# we want to check from 2020/12/4 to 2021/12/4
start_date = date(2020, 12, 4)
end_date = date(2021, 12, 4)
delta = dt.timedelta(days=15)

# get the posts before time
def before(dic, time):
    result = {}
    for key in dic.keys():
        if datetime.fromtimestamp(dic[key]['time']).date() <= time:
            result[key] = dic[key]
            
    return result

import numpy as np

betweenness_centrality = []
matrix = []
network = []

# get network for each 15 days and calculate cultural betweenness
while start_date <= end_date:
    print(start_date)
    s1 = before(AllWomen,start_date)
    s2 = before(asiantwoX,start_date)
    s3 = before(AskFeminists,start_date)
    s4 = before(blackladies,start_date)
    s5 = before(careerwomen,start_date)
    s6 = before(Feminism,start_date)
    s7 = before(FeMRADebates,start_date)
    s8 = before(NotHowGirlsWork,start_date) 
    s9 = before(SexPositive,start_date)
    s10 = before(transgender,start_date)
    
    s1_str = combine_lemmas(s1)
    s2_str = combine_lemmas(s2)
    s3_str = combine_lemmas(s3)
    s4_str = combine_lemmas(s4)
    s5_str = combine_lemmas(s5)
    s6_str = combine_lemmas(s6)
    s7_str = combine_lemmas(s7)
    s8_str = combine_lemmas(s8)
    s9_str = combine_lemmas(s9)
    s10_str = combine_lemmas(s10)
    
    corpus = [s1_str, s2_str, s3_str,
              s4_str, s5_str, s6_str,
              s7_str, s8_str, s9_str,
              s10_str]
    
    score = vectorizer.fit_transform(corpus)
    
    score_array = score.toarray()
    
    subreddits = ['r/AllWomen', 'r/asiantwoX', 'r/AskFeminists', 'r/blackladies',
                      'r/careerwomen', 'r/Feminism', 'r/FeMRADebates', 'r/NotHowGirlsWork',
                      'r/SexPositive', 'r/transgender']
    
    g = nx.Graph()

    for row in range(len(score_array)):
        for column in range(len(score_array[row])):
            
            weight =  score_array[row][column]
            if weight > 0:
                
                    g.add_node(f'{subreddits[row]}', bipartite = 0)
                    g.add_node(f'{names[column]}', bipartite = 1)
                    g.add_edge(f'{subreddits[row]}', f'{names[column]}', weight = weight)
                    
    
    top = [node for node in g.nodes() if g.nodes(data = True)[node]['bipartite']==0]
    
    G = bipartite.projected_graph(g,top)
    
    network.append(G)
    
    weights = []
    for edge in G.edges():
        
        ind1 = subreddits.index(edge[0])
        ind2 = subreddits.index(edge[1])
        
        weight = 0
        
        for w1 in score_array[ind1]:
            if w1 != 0:
                for w2 in score_array[ind2]:
                    if  w2 != 0:
                        weight += w1 + w2
                        
        weights.append(weight)
        
        
    i = 0
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = weights[i]
        i += 1
    
    # calculate cultural betweenness
    num_nodes = G.number_of_nodes()
    
    dj_path_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            dj_path_matrix[i,j] = nx.dijkstra_path_length(G, list(G.nodes())[i], list(G.nodes())[j])

    
    L = np.sum(dj_path_matrix, axis=0)/(num_nodes - 1)
    
    print(L)
    
    betweenness_centrality.append(L)
    matrix.append(dj_path_matrix)
    
    start_date += delta


print(g.number_of_nodes())

# replace null with 0 in betweenness_centrality matrix
c = betweenness_centrality.copy()
c


for i in range(len(c)):
    if len(c[i]) == 9:
        c[i] = np.insert(c[i],0,0)

print(c)


# get the number of users for each 15 days
def get_author_num(dic):
    start_date = date(2020, 12, 4)
    end_date = date(2021, 12, 4)
    delta = dt.timedelta(days=15)
    
    result = []
    
    while start_date <= end_date:
        count = 0
        
        for key in dic.keys():
            if datetime.fromtimestamp(dic[key]).date() <= start_date:
                count+=1        
        
        result.append(count)
        
        start_date += delta
        
    return result





print('1')
AllWomen_author_num = get_author_num(AllWomen_com)
print('2')
asiantwoX_author_num = get_author_num(asiantwoX_com) 
print('3')
AskFeminists_author_num = get_author_num(AskFeminists_com)
print('4')
blackladies_author_num = get_author_num(blackladies_com)
print('5')
careerwomen_author_num = get_author_num(careerwomen_com)
print('6')
Feminism_author_num = get_author_num(Feminism_com)
print('7')
FeMRADebates_author_num = get_author_num(FeMRADebates_com)
print('8')
NotHowGirlsWork_author_num = get_author_num(NotHowGirlsWork_com)
print('9')
SexPositive_author_num = get_author_num(SexPositive_com)
print('10')
transgender_author_num = get_author_num(transgender_com)


# get ordered cultural btweenness for each subreddit

AllWomen_btw = []
print('2')
asiantwoX_btw = []
print('3')
AskFeminists_btw = []
print('4')
blackladies_btw = []
print('5')
careerwomen_btw = []
print('6')
Feminism_btw = []
print('7')
FeMRADebates_btw = []
print('8')
NotHowGirlsWork_btw = []
print('9')
SexPositive_btw = []
print('10')
transgender_btw = []


for row in range(len(c)):
    AllWomen_btw.append(c[row][0])
    print('2')
    asiantwoX_btw.append(c[row][1])
    print('3')
    AskFeminists_btw.append(c[row][2])
    print('4')
    blackladies_btw.append(c[row][3])
    print('5')
    careerwomen_btw.append(c[row][4])
    print('6')
    Feminism_btw.append(c[row][5])
    print('7')
    FeMRADebates_btw.append(c[row][6])
    print('8')
    NotHowGirlsWork_btw.append(c[row][7])
    print('9')
    SexPositive_btw.append(c[row][8])
    print('10')
    transgender_btw.append(c[row][9])



# plot linear regression and calculate slope and intercept

def get_plot(x, y, t):


    labels = [float(i) for i in y]
    features = [float(i) for i in x]
    con = pd.DataFrame(labels, features).reset_index()

    con = con.rename(columns={"index": "cultural betweenness", 0: "number of new commenters"})
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(features,labels)
    
    slope = ('{:.3g}'.format(slope))
    intercept = ("%.3f" % intercept)



    f = sns.lmplot(x="cultural betweenness", y='number of new commenters', data=con).set(title=t)
    
    plt.legend(loc = 'upper left', labels = ['data points', f'number of new commenters = {slope}*cultural betweenness + {intercept}',
                                             '0.95 Confience Interval'])
    
    
    f.savefig(f'{t}.png', dpi=1000)

get_plot(AllWomen_btw, AllWomen_author_num, 'AllWomen')

get_plot(asiantwoX_btw, asiantwoX_author_num, 'asiantwoX')

get_plot(AskFeminists_btw, AskFeminists_author_num, 'AskFeminists')

get_plot(blackladies_btw, blackladies_author_num, 'blackladies')

get_plot(careerwomen_btw, careerwomen_author_num, 'careerwomen')

get_plot(Feminism_btw, Feminism_author_num, 'Feminism')

get_plot(FeMRADebates_btw, FeMRADebates_author_num, 'FeMRADebates')

get_plot(NotHowGirlsWork_btw, NotHowGirlsWork_author_num, 'NotHowGirlsWork')

get_plot(SexPositive_btw, SexPositive_author_num, 'SexPositive')

get_plot(transgender_btw, transgender_author_num, 'transgender')


























