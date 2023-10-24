% Importing pre-processed train data
data = importdata("data_train[9].txt");

% Importing GloVe pre-trained word embedding
filename = "/Users/miguelrosales/Documents/UEA/Computer Science/Dissertation/Paper/MATLAB/GloVe/glove.6B.300d";
if exist(filename + '.mat', 'file') ~= 2
    emb = readWordEmbedding(filename + '.txt');
    save(filename + '.mat', 'emb', '-v7.3');
else
    load(filename + '.mat')
end

% Pre-Processing the Data
preProData = tokenizedDocument(data);
preProData = addPartOfSpeechDetails(preProData);
preProData = removeStopWords(preProData);
preProData = erasePunctuation(preProData);
preProData = lower(preProData);
preProData = removeShortWords(preProData, 2);

% Writing the pre-processed data set to a text file
filename = "preProTrain[9].txt";
writeTextDocument(preProData,filename)

% Extracting file text from filename
words = extractFileText(filename);

% Loading pre-trained word embedding 
% emb = fastTextWordEmbedding;

% Joining words from data set with pre-trained embedding
nouns = words;
nouns = split(nouns);
nouns(~ismember(nouns,emb.Vocabulary)) = [];
vec = word2vec(emb,nouns);

% Visualising data set with pre-trained word embedding
% Resources: https://uk.mathworks.com/help/textanalytics/ug/visualize-word-embedding-using-text-scatter-plot.html?searchHighlight=3d%20text%20scatter%20plot&s_tid=srchtitle_3d%20text%20scatter%20plot_1
rng('default'); % for reproducibility
xy = tsne(vec);
figure
textscatter(xy,nouns)
title('GloVe Embeddings (9 Topics)')
set(gca,'clipping','off')
axis off

% Visualising train data via 3D text scatter plot
XYZ = tsne(vec,'NumDimensions',3);
figure 
ts = textscatter3(XYZ,nouns);
title("3-D GloVe Embeddings (9 Topics)")

% Normalising vectors
range = [-1,1];
norm = normalize(vec,"range",range);

% Cosine Similarity of word2vec
co_sim = cosineSimilarity(norm);
figure
heatmap(co_sim);
xlabel("Word Vectors in the Corpus")
ylabel("Word Vectors in the Corpus")
title("Cosine Similarities of Word Vectors (9 Topics) via GloVe")

% Importing pre-processed train data
data = importdata("data_train[9].txt");

% Pre-Processing the Data
preProData = tokenizedDocument(data);
preProData = addPartOfSpeechDetails(preProData);
preProData = removeStopWords(preProData);
preProData = erasePunctuation(preProData);
preProData = lower(preProData);
preProData = removeShortWords(preProData, 2);

% Create a bag-of-words model
bag = bagOfWords(preProData);

% Removing empty utterances
bag = removeEmptyDocuments(bag);

% Fitting the LDA model
rng("default")
numTopics = 9;
mdl = fitlda(bag,numTopics,Verbose=0);

% Visualising Corpus Topic Probabilities (Bar Chart)
topicProbabilities = mdl.CorpusTopicProbabilities;
figure
bar(topicProbabilities(1,:))
title("Corpus Topic Probabilities (9 Topics)")
xlabel("Topics")
ylabel("Probability")

% Visualising Topic Word Clouds
figure
t = tiledlayout("flow");
title(t,"LDA Topics (9 Topics)")

for i = 1:numTopics
    nexttile
    wordcloud(mdl,i);
    title("Topic " + i)
end

% Visualising Topic Mixtures for each conversation
topicDocumentProbabilities = mdl.DocumentTopicProbabilities;
figure
barh(topicDocumentProbabilities,'stacked')
xlim([0 1])
title("Conversation Topic Mixtures (9 Topics)")
xlabel("Topic Probability")
ylabel("Conversation")
legend("Topic " + string(1:numTopics),'Location','northeastoutside')
 
% Visualising Topic Probabilities
for i = 1:numTopics
    top = topkwords(mdl,3,i);
    topWords(i) = join(top.Word,", ");
end
 
figure
bar(topicDocumentProbabilities(1,:))
 
xlabel("Topic")
xticklabels(topWords);
ylabel("Probability")
title("Conversation Topic Probabilities (9 Topics)")
 
% Obtain the Document-Term Matrix
dtm = full(bag.Counts);

% Obtaining the feature words 
FW = topkwords(bag);

% Defining variables for LDA
topicWordProbabilities = mdl.TopicWordProbabilities;

% Creating Feature Word (FW) Array

no_FW = 3;

featureWords = cell(numTopics, 1);

for n = 1:numTopics
    featureWords{n} = topkwords(mdl,3,n);
end

featureWordsTbl = vertcat(featureWords{:});

% Extract the column data as a column vector
columnData = featureWordsTbl.Word;

% Convert the column vector into a row vector using the reshape function
rowVector = reshape(columnData.', 1, []);

% Loading pre-trained word embedding 
% emb = fastTextWordEmbedding;

% Joining feature words with pre-trained embedding
words = rowVector;
words(~ismember(words,emb.Vocabulary)) = [];
vectors = word2vec(emb,words);

% Normalising vectors
range = [-1,1];
topicNorm = normalize(vectors,"range",range);

% Cosine similarities between feature words and every word in the data
% (label y-axis)
topicSimilarities = cosineSimilarity(topicNorm,norm);
figure
heatmap(topicSimilarities);
xlabel("Words in Data")
ylabel("Feature Words determined by LDA")
title("Feature Word Cosine Similarities of Word Vectors (9 Topics) via GloVe")

% Calculating the Probability Weighted Sum of each Feature Word

% (i) Count number of occurences of each feature word in each conversation
% conversation = preProData.tokenizedDocument;

numFeatureWords = size(featureWordsTbl.Word, 1);
wordCount = cell(numFeatureWords, numTopics);

wordCountsTable = table('Size', [numFeatureWords, numTopics], 'VariableTypes', repmat({'double'}, 1, numTopics), 'RowNames', rowVector);

noCount = numel(wordCountsTable);
numTopicVariables = (noCount/numTopics);

% Creating Arrays
fwOccurences = zeros(noCount, 1);
wordCountArray = zeros(numTopicVariables,numTopics);

wCount = 0;


for n = 1:numTopics
    % Iterating through every conversation in the data set 
    conversations = preProData(n,1);
    for w = rowVector
        % Finding the number of occurences of every feature word for each
        % topic through the sum of the documents and saving it to an array.
        occurences = context(conversations, w);
        count = sum(occurences.Document);
        wCount = wCount +1;
        fwOccurences(wCount, 1) = count;
       
    end
end

% Adding fw occurences to the table to read the occurences of each feature
% word for each topic

wCount2 = 0;

for t = 1:numTopics
    for x = 1:numTopicVariables
        wCount2 = wCount2 + 1;
        wordCountArray(x,t) = fwOccurences(wCount2, 1);
        disp(wCount2)
    end
end

%  Visualising FW Occurence Table

wordCountsTable = array2table(wordCountArray);
wordCountsTable.Properties.VariableNames = cellstr(string(1:numTopics));
wordCountsTable.Properties.RowNames = rowVector;

figure;
bar(wordCountArray, 'stacked');
xlabel('Feature Words');
ylabel('Occurences');
title('Feature Word Occurences per Topic (9 Topics)');
set(gca, 'XTickLabel', rowVector);
legend('Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5', 'Topic 6', 'Topic 7', 'Topic 8', 'Topic 9');


% ii) Select FW and their Word-Topic Probabilities

numWords = numel(mdl.Vocabulary);

% Making a labelled table with the same values as mdl.TopicWordProbabilities
wordProbabilities = array2table(mdl.TopicWordProbabilities);
wordProbabilities.Properties.VariableNames = cellstr(string(1:numTopics));
wordProbabilities.Properties.RowNames = mdl.Vocabulary;

% Selecting feature word values from the table 

fwProbabilities = wordProbabilities(rowVector, :);

% iii) Multiply every probability for each fw with every cosine similarity
% for that fw for each topic 

% Creating a cell for the probability weighted sum of each fw under the
% k-th topic

fwProbArray = table2array(fwProbabilities);
probWeightSum = zeros(numFeatureWords, numTopics);
allProbWeightSum = cell(numFeatureWords, numTopics);


for fw = 1:numFeatureWords
    % Selecting the cosine similarities for the fw and every term
    sim = topicSimilarities(fw, :);
    for n = 1:numTopics
        % Selecting each fw probabilitiy under every topic
        value = fwProbArray(fw, n);
        % Multiplying each selection to get the probability weighted sum
        probWeightSum = sim * value;
        % Storing it in a cell
        allProbWeightSum(fw, n) = {probWeightSum};
    end
end

% iv) Probability Weighted Sum of All Topics

% Creating cell array of document (conversation) topic probabilities * by
% probability weighted sum

docProbSum = cell(numTopics, numTopics);
docProbCell = zeros(numFeatureWords, numel(nouns));

k = 0;
for n = 1:numTopics
    sumWeight = allProbWeightSum(:, n);
    sumWeightArray = cell2mat(sumWeight);
    for i = 1:numTopics
        docProb = mdl.DocumentTopicProbabilities(i, n);
        docValue = docProb * sumWeightArray;
        docProbSum(n, i) = {docValue};
    end
end

% v) Calculating Total Relevancy

% Creating cell array for total topic relevancy

topicRelevancy = zeros(numTopics, numTopics);

for n = 1:numTopics
    for i = 1:numTopics
        doCell = docProbSum{n, i};
        sumProb = sum(doCell(:));
        topicRelevancy(n, i) = sumProb;
    end
end

figure
heatmap(topicRelevancy);
xlabel("Conversations")
ylabel("Topics")
title("Topic Relevancy of each Conversation (9 Topics) - GloVe")

