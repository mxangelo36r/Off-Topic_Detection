# Off-Topic_Detection
Utilised word2vec, GloVe and LDA to visualise conversations and see how topics shift within a dialogue

### Aims
Investigated the effectiveness of word embedding tools for off-topic detection in conversations 

### Method
1) Gathered 5 conversational corpora containing a different number of topics (5, 7, 9, 11 and 13). Each set contains a unique conversational topic to be trained on using GloVe, word2vec
and LDA
2) Adopted a similarity measurement approach and calculated the topic relevancies for each conversation in the corpora using cosine similarity values determined from word embeddings, and 
document-topic & topic-word probabilities gathered from LDA
3) Compared topic relevancy values against each other to determine when the conversation shifted in topic

### Topic Relevancy
- Calculated values which shows how related a conversation is to the topics determined by LDA
- Larger values indicates that a conversation is highly related to a topic and smaller values represents a conversation that is slightly related to a topic
- Determined the cosine similarity of every LDA-derived topic words with all the word items in the corpus to extract the semantic information, and using the probalistic values gathered
from LDA, the topic relevancies of each conversation can be calculated using the following formulas:

#### Probability Weighted Sum of every LDA-derived Topic Word and Word Item
<p align="center">$S(w_j,t_i) = \sum P(w_n|t_i) \cdot cos(w_j|w_n)$</p> 

#### Probability Weighted Sum of every Word Item and Topic
<p align="center">$S(w_j,d_m) = \sum P(t_i|d_m) \cdot S(w_j|w_n)$</p> 

#### Total Relevancy
<p align="center">$S_m = \sum S(w_j|d_m)$</p> 

### Results 
 - Findings show that some conversations have high topic relevancy values, indicating that a topic is highly related and is more dominant than the other topics. Dominance is compared 
against the whole corpus to quantitatively measure off-topic detection. Some conversations have varied topic relevancy distributions showing that conversations are a mix of topics.
- Differences in values between GloVe and word2vec are based on their cosine values (semantic information) given by their respective word embeddings
- Topic relevancy distributions are influenced by document-topic and topic-word probabilities determined by LDA
