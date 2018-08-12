
# coding: utf-8

# ## Summary
# #### Steps
# 1. All the data and categories are copied to their respective arrays(test and training data are already split in corpus).
# 2. A set of features are selected on the training data.
#     - Replace digits with character ' NN '
#     - Removed Punctuations
#     - Filtered Close class words from the documents using POS tag.
# 3. Feature Vector is generated from the extracted features in the previous step using CountVectorizer
# 4. Vectorize the category list using Multi Lable Binarizer
# 5. Extrinsic Evaluation of multiple classifiers from Scikit-learn algorithm cheat-sheet and those taught in class.
#     - Selected the one with the best score.
# 6. Training one classifier per class on training set.
# 7. Predict tags from the trained clasifiers on Test set.
# 8. Performance analysis on the test set.
#     - Accuracy, Precision, Recall and F1-score for combined classifier and per class(tag).
#     - Plot confusion matrix for any class user enters.
# 9. Pipeline - Predict tag for any data user enters
#     - Copy the data in the string inp_text
# 10. Conclusion and Outlook into future work.
# 
# #### Note  
# Cross validaton is skipped because
# - Only k-fold cross validation is supported in sklearn for multi value classification problems which split the training set randomly
#     - This can cause the problem of sampling errors that we are trying to avoid
# - The second approach "stratified kfold" which preserves the percentage of samples for each class, does not support multivalue problems.

# ## Importing required modules and packages

# In[ ]:


import re
import string

try:
    import sklearn
except ModuleNotFoundError:
    get_ipython().system('pip install scikit-learn   ')

try:
    import nltk
except ModuleNotFoundError:
    get_ipython().system('pip install nltk')

try:
    import numpy as np
except ModuleNotFoundError:
    get_ipython().system('pip install numpy')

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    get_ipython().system('pip install matplotlib')


# In[ ]:


## This code downloads the required packages.
## You can run `nltk.download('all')` to download everything.
nltk_packages = [
    ("reuters", "corpora/reuters.zip"),
]

for pid, fid in nltk_packages:
    try:
        nltk.data.find(fid)
    except LookupError:
        nltk.download(pid)


# ## Setting up corpus

# In[ ]:


from nltk.corpus import reuters


# ## Copying Train and Test data

# In[ ]:


train_documents     = [(reuters.raw(i)) for i in reuters.fileids() if i.startswith('training/')]
train_categories    = [(reuters.categories(i)) for i in reuters.fileids() if i.startswith('training/')]
test_documents      = [(reuters.raw(i)) for i in reuters.fileids() if i.startswith('test/')]
test_categories     = [(reuters.categories(i)) for i in reuters.fileids() if i.startswith('test/')]


# In[ ]:


all_categories      = sorted(list(set(reuters.categories())))
print("Data Extracted from corpus !!!")


# ## Feature Extraction
# #### All the numbers and dates are replace by 'NN'
# - As different values of dates or numbers will not be a relevant feature in our problem
# - Replacing all with same character might contribute to some tags.
# 
# #### Remove the puntuations.
# - Add no semantical meaning to a sentence.
# 
# #### Filter out close class words.
# - Close class words are related to syntactics of a sentence, rather than the semantics.
# - Hence, Open class words(Nouns, Verbs and Adjectives) are selected as features for the classifier.
# 
# ###### Note -  Tried upvoting(copying title twice) but didn't had a positive effect on F1-score.

# In[ ]:


#--------------------------------------------------------------------------------------#
# Using regular expression to capture all the digits(decimal, float, dates, years)
# Encoding all the ocurances of the found instances with ' NN '
for i in range (len(train_documents)):
    number = re.findall(r'[0-9]+[,/]?[\.0-9]*', train_documents[i])
    num_cnt = len(number) 
    if(num_cnt > 0):
        for j in range (num_cnt):
            big_regex = re.compile('|'.join(map(re.escape, number)))
            train_documents[i] = big_regex.sub(" NN ", train_documents[i])


# In[ ]:


#--------------------------------------------------------------------------------------#
from nltk.tokenize import word_tokenize
#--------------------------------------------------------------------------------------#
puntuation_list = set(string.punctuation)
tokens_filtered = []
for i in range (len(train_documents)):
    doc_tokens = word_tokenize(train_documents[i])
    #----------------------------------------------------------------------------------#
    rem_puntuations = [t for t in doc_tokens if t not in puntuation_list]
    #----------------------------------------------------------------------------------#
    add_pos = nltk.pos_tag(rem_puntuations, tagset='universal')
    #----------------------------------------------------------------------------------#
    ## Nouns also capture the character NN we generated to represent digits.
    collect_nouns = [(t, pos) for t, pos in add_pos if pos.startswith('NOUN')]
    collect_verbs = [(t, pos) for t, pos in add_pos if pos.startswith('VERB')]
    collect_adjectives = [(t, pos) for t, pos in add_pos if pos.startswith('ADJ')]
    #----------------------------------------------------------------------------------#
    tokens_filtered.append(collect_nouns + collect_verbs + collect_adjectives)


# In[ ]:


#--------------------------------------------------------------------------------------#
# Filtering out the POS tags from the list of selected features
word_features = []
for i in range (len(train_documents)):
    doc_features = ' '.join([w for w, _ in tokens_filtered[i]])
    word_features.append(doc_features)
print("Features Extracted !!!")


# ## Applying Multi Label Binarizer on categories 
# - It returns a 2D array, each row representing a document
# - It generates an array of all the unique categories 
# - Assigns a value '1' to a particular class if it belongs to a document, otherwise '0'

# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train_categories)
Y_test  = mlb.fit_transform(test_categories)


# ## CountVectorizer on data
# - Similar to Multi Label Binarizer, CountVectorizer returns a 2D array, each row representing a document
# - It generates a vocabulary of all the unique words using fit()
# - Then assign the count of each word in particular document, i.e. assigning importance of that word in the document
# 
# ###### Note - TfidVectorizer is advantageous to filter out the common words in most of the documents, but we have already filtered out those words in feature extraction so it makes sense to use the count value of the words that appear in the document 
# - CountVectorizer gave better accuracy, than TfidVectorizer

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(
                analyzer='word',
                ngram_range=(1, 1),
                stop_words='english',                        
                max_df=1.0,
                min_df=1,
                max_features=None)
#--------------------------------------------------------------------------------------#
# Generating the vocabulary
count_vect.fit(word_features + test_documents)
#--------------------------------------------------------------------------------------#
# Transforming the features to vectors from the vocabulary
X_train = count_vect.transform(word_features)
X_test  = count_vect.transform(test_documents)
#--------------------------------------------------------------------------------------#


# ## Selecting the classifier
# - As explained in the lecture, with reasonable amount of data, SVM, Logistic Regression and Decision trees are better. 
# - It is proven here as Naive Bayes had the worst score amoung SGDC, Decision tree and SVM. 
# - Decision tree and Linear SVC(SVM) were similar but Linear SVC are very fast in comparison to Decision trees.
# 
# ## Training the classifier
# - OrderedDict -  remember the order that items were inserted.
# - A classifier is trained for each category and then inserted in an OrderedDict arranged in alphabetic order of classes.

# In[ ]:


from sklearn.svm import LinearSVC
from collections import OrderedDict

clfs = OrderedDict()
for i, category in enumerate(all_categories):
    clf = LinearSVC()
    #--------------------------------------------------------------------------------------#
    # We train each classifier individually
    # So used each column in Y_train vector
    y_train_clf = [yt[i] for yt in Y_train]
    #--------------------------------------------------------------------------------------#
    # fit() will train the model with the training data
    clf.fit(X_train, y_train_clf)

    clfs[category] = clf


# ## Classifier Evaluation

# In[ ]:


Y_pred = np.zeros((len(Y_test), len(all_categories)))
for i, (cat, clf) in enumerate(clfs.items()):
    Y_pred[:, i] = clf.predict(X_test)


# ## Performance Measurements
# - Reuters data set is very skewed. 
#     - There are 90 categoties and 10788 documents and for equal distribution there should have been 120 docs/category, but less than 10% categories satisify the condition.
#     - The most common category contributes 36.7% of the corpus.
#     
# - If we want to measure how good our classifier is for a particular class then "Macro averaging" is the way to go
#     - As it gives equal importance to all the classes.
# 
# - We are interested in the number of documents our classifier gets correctly, "Micro averaging" does that precisely
#     - combine the confusion matrix of all the classes and then calculate the scores.
#     
# ###### Hence Micro averaging is selected.

# In[ ]:


from sklearn import metrics

print("Accuracy : {:.2f}%".format(metrics.accuracy_score(Y_test, Y_pred)*100))
print("Precision: {:.2f}%".format(metrics.precision_score(Y_test, Y_pred, average='micro')*100))
print("Recall   : {:.2f}%".format(metrics.recall_score(Y_test, Y_pred, average='micro')*100))
print("F1-Score : {:.2f}%".format(metrics.f1_score(Y_test, Y_pred, average='micro')*100))


# In[ ]:


print(metrics.classification_report(y_true=Y_test, y_pred=Y_pred, target_names=mlb.classes_))


# ## Confusion matrix for per class
# - Enter name of any class fom the list below for the confusion matrix of the class.
# 
# #### List of classes
#     acq              alum          barley        bop           carcass       castor-oil    cocoa     
#     coconut          coconut-oil   coffee        copper        copra-cake    corn          cotton        
#     cotton-oil       cpi           cpu           crude         dfl           dlr           dmk      
#     earn             fuel          gas           gnp           gold          grain         groundnut    
#     groundnut-oil    heat          hog           housing       income        instal-debt   interest
#     ipi              iron-steel    jet           jobs          l-cattle      lead          money-supply
#     lei              lin-oil       livestock     lumber        meal-feed     money-fx      naphtha           
#     nat-gas          nickel        nkr           nzdlr         oat           oilseed       orange        
#     palladium        palm-oil      palmkernel    pet-chem      platinum      potato        propane       
#     rand             rape-oil      rapeseed      reserves      retail        rice          rubber         
#     rye              ship          silver        sorghum       soy-meal      soy-oil       soybean      
#     strategic-metal  sugar         sun-meal      sun-oil       sunseed       tea           tin           
#     trade            veg-oil       wheat         wpi           yen           zinc 
# 
# ##### Format of confusion matrix
#     Class Name
#     [[ True Neg   False Pos]
#      [ False Neg  True Pos ]]

# In[ ]:


cat_inp = input("Enter the name of any class : ")
if cat_inp in (all_categories):
    for i, cat in enumerate(all_categories):
        if(cat_inp == cat):
            print("")
            print("{}".format(cat))
            m = metrics.confusion_matrix(y_true=Y_test[:,i], y_pred=Y_pred[:,i])
            print(m)
else:
     print(" Enter a valid category name")


# In[ ]:


plt.clf()
plt.imshow(m, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Not ' + cat_inp, cat_inp]
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(m[i][j]))
plt.show()


# ## Pipeline
# - Enter the text to be tagged, some examples are given to just copy and try.

# In[ ]:


#--------------------------------------------------------------------------------------#
# Correct or partly correct tags
    #Palm kernel oil is an edible plant oil derived from the kernel of the oil palm Elaeis guineensis.[1] It should not be confused with the other two edible oils derived from palm fruits: palm oil, extracted from the pulp of the oil palm fruit, and coconut oil, extracted from the kernel of the coconut.
    #Natural gas is a naturally occurring hydrocarbon gas mixture consisting primarily of methane, but commonly including varying amounts of other higher alkanes, and sometimes a small percentage of carbon dioxide, nitrogen, hydrogen sulfide, or helium.[2] It is formed when layers of decomposing plant and animal matter are exposed to intense heat and pressure under the surface of the Earth over millions of years. The energy that the plants originally obtained from the sun is stored in the form of chemical bonds in the gas.
    #Copper rebounded on Friday from a fresh 11-month low, as some investors regard the recent sharp losses as exaggerated and believe the metal has hit bottom. Copper, widely viewed as a bellwether for the global economy, has been battered by escalating trade tensions that resulted in the United States imposing tariffs on $34 billion of Chinese imports and Beijing quickly retaliating. The recent downtrend - which has seen copper shed 14 percent since touching a 4-1/2 year peak of $7,348 in early June - was fuelled by computer-driven speculators and long liquidation by Chinese hedge funds, said Gianclaudio Torlizzi, Partner at consultancy T-Commodity in Milan. Many people are wondering if the long-term bull market is over. I don't think it's over but copper has to hold above $6,200, which is the watershed level for the long-term uptrend, he said. It doesn't make any sense to have such a gloomy sentiment on metals demand. This is a good opportunity to go long again, Torlizzi added, saying he had already taken a long position
    #Zinc is a bluish-white, lustrous, diamagnetic metal,[12] though most common commercial grades of the metal have a dull finish.[13] It is somewhat less dense than iron and has a hexagonal crystal structure, with a distorted form of hexagonal close packing, in which each atom has six nearest neighbors (at 265.9 pm) in its own plane and six others at a greater distance of 290.6 pm.[14] The metal is hard and brittle at most temperatures but becomes malleable between 100 and 150 °C.[12][13] Above 210 °C, the metal becomes brittle again and can be pulverized by beating.[15] Zinc is a fair conductor of electricity.[12] For a metal, zinc has relatively low melting (419.5 °C) and boiling points (907 °C).[16] The melting point is the lowest of all the d-block metals aside from mercury and cadmium; for this, among other reasons, zinc, cadmium, and mercury are often not considered to be transition metals like the rest of the d-block metals are.
#--------------------------------------------------------------------------------------#
# Missed Tags
    # Aluminium-based alums have a number of common chemical properties. They are soluble in water, have a sweetish taste, react acid to litmus, and crystallize in regular octahedra. In alums each metal ion is surrounded by six water molecules. When heated, they liquefy, and if the heating is continued, the water of crystallization is driven off, the salt froths and swells, and at last an amorphous powder remains.[3] They are astringent and acidic.
#--------------------------------------------------------------------------------------#
# Input sentence
inp_text = input("Enter the text you want to classify")


# In[ ]:


pipeline = []
pipeline.append(inp_text)
#--------------------------------------------------------------------------------------#
# Extract features
example_features = count_vect.transform(pipeline)
#--------------------------------------------------------------------------------------#
# Do prediction
example_preds = [clf.predict(example_features)[0] for clf in clfs.values()]
#--------------------------------------------------------------------------------------#
# Convert predictions back to labels
mlb.inverse_transform(np.array([example_preds]))


# ## Conclusion
# - The approach is pretty balanced as the precision and recall are pretty high and almost similar.
# 
# ### Outlook into future work
# - Out of 90 classes there are still 20 which has a precision and recall value of 0.00
# - To improve further following can be explored
#     - Lemmatize the selected features.
#     - Calculate similarity(thisaurous and distributional) to find similar features and collapse then into one word.
#     - Use dense vectors instad of sparse vectors
#     - Maybe, training the classfier as OneVsRest,as multiple clasifier support that in sklearn.
