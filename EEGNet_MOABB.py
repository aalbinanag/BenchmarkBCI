"""
==================================
EEGNet applied in MOABB framework
==================================
"""
# Sklearn imports
import numpy as np
from scipy.fftpack import dst
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from sklearn.model_selection import train_test_split

# MOABB imports
import moabb
from moabb.datasets import BNCI2014001, utils
from moabb.paradigms import MotorImagery

# EEGNet-specific imports
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# Classifiers comparison export
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import Covariances
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# tools for plotting
from matplotlib import pyplot as plt
import pandas as pd

# Intra - subject classification function:

def intra_subject_classification(epochs, labels, clf1, clf2):
    # Convert the str values in labels to integers
    labels = np.where(labels == 'right_hand', 1, labels)
    labels = np.where(labels == 'left_hand', 2, labels)
    labels = np.where(labels == 'tongue', 3, labels)
    labels = np.where(labels == 'feet', 4, labels)
    labels = labels.astype(int)

    X = epochs
    y = labels

    kernels, chans, samples = 1, 22, 257 # Obtained from epochs.shape

        ############################# EEGNet portion ##################################
    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
    # model configurations may do better, but this is a good starting point)
    model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples, 
                dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                dropoutType = 'Dropout')

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])

    # count number of parameters in the model
    numParams    = model.count_params()    

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=0,
                                save_best_only=True)

    ###############################################################################
    # if the classification task was imbalanced (significantly more trials in one
    # class versus the others) you can assign a weight to each class during 
    # optimization to balance it out. This data is approximately balanced so we 
    # don't need to do this, but is shown here for illustration/completeness. 
    ###############################################################################

    # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
    # the weights all to be 1
    class_weights = {0:1, 1:1, 2:1, 3:1}
    acc_intra_lst = []
    acc2_lst = []
    acc3_lst = []
    kf = KFold(n_splits=4)
    for train_index, test_index in kf.split(y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]

        # convert labels to one-hot encodings.
        Y_train      = np_utils.to_categorical(Y_train-1)
        Y_test       = np_utils.to_categorical(Y_test-1)

        # convert data to NHWC (trials, channels, samples, kernels) format. Data 
        # contains 22 channels and 1001 time-points. Set the number of kernels to 1.
        X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

        ################################################################################
        # fit the model. Due to very small sample sizes this can get
        # pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
        # Riemannian geometry classification (below)
        ################################################################################

        fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 500, 
                                verbose = 0, validation_split=1/3,
                                callbacks=[checkpointer], class_weight = class_weights)

        # load optimal weights
        model.load_weights('/tmp/checkpoint.h5')

        ###############################################################################
        # can alternatively used the weights provided in the repo. If so it should get
        # you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
        # system.
        ###############################################################################

        # WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5 
        # model.load_weights(WEIGHTS_PATH)

        ###############################################################################
        # make prediction on test set.
        ###############################################################################

        probs       = model.predict(X_test)
        preds       = probs.argmax(axis = -1)  
        acc         = np.mean(preds == Y_test.argmax(axis=-1))
        acc_intra_lst.append(acc)

        # Test other classifiers
        
        preds_rg     = np.zeros(len(Y_test))
        # reshape back to (trials, channels, samples)
        X_train      = X_train.reshape(X_train.shape[0], chans, samples)
        X_test       = X_test.reshape(X_test.shape[0], chans, samples)

        clf1.fit(X_train, Y_train.argmax(axis = -1))
        preds_rg     = clf1.predict(X_test)

        acc2         = np.mean(preds_rg == Y_test.argmax(axis = -1))
        acc2_lst.append(acc2)

        preds_rg     = np.zeros(len(Y_test))

        clf2.fit(X_train, Y_train.argmax(axis = -1))
        preds_rg     = clf2.predict(X_test)

        acc3         = np.mean(preds_rg == Y_test.argmax(axis = -1))
        acc3_lst.append(acc3)


    acc2_mean = sum(acc2_lst)/len(acc2_lst)
    acc3_mean = sum(acc3_lst)/len(acc3_lst)
    
    acc_intra_mean = sum(acc_intra_lst)/len(acc_intra_lst)
    return acc_intra_mean, acc2_mean, acc3_mean

def inter_subject_classification(epochs, labels, clf1, clf2):
    labels = np.where(labels == 'right_hand', 1, labels)
    labels = np.where(labels == 'left_hand', 2, labels)
    labels = np.where(labels == 'tongue', 3, labels)
    labels = np.where(labels == 'feet', 4, labels)
    labels =labels.astype(int)

    X = epochs
    y = labels

    kernels, chans, samples = 1, 22, 257 # Obtained from epochs.shape

    ############################# EEGNet portion ##################################
    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
    # model configurations may do better, but this is a good starting point)
    model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples, 
                dropoutRate = 0.25, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                dropoutType = 'Dropout')

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])

    # count number of parameters in the model
    numParams    = model.count_params()    

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=0,
                                save_best_only=True)

    ###############################################################################
    # if the classification task was imbalanced (significantly more trials in one
    # class versus the others) you can assign a weight to each class during 
    # optimization to balance it out. This data is approximately balanced so we 
    # don't need to do this, but is shown here for illustration/completeness. 
    ###############################################################################

    # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
    # the weights all to be 1
    class_weights = {0:1, 1:1, 2:1, 3:1}
    acc_inter_lst = []
    acc2_lst = []
    acc3_lst = []
    kf = KFold(n_splits=9)
    for train_index, test_index in kf.split(y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]

        # convert labels to one-hot encodings.
        Y_train      = np_utils.to_categorical(Y_train-1)
        Y_test       = np_utils.to_categorical(Y_test-1)

        # convert data to NHWC (trials, channels, samples, kernels) format. Data 
        # contains 22 channels and 1001 time-points. Set the number of kernels to 1.
        X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)


        ################################################################################
        # fit the model. Due to very small sample sizes this can get
        # pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
        # Riemannian geometry classification (below)
        ################################################################################

        fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 500, 
                                verbose = 0, validation_split=1/3,
                                callbacks=[checkpointer], class_weight = class_weights)

        # load optimal weights
        model.load_weights('/tmp/checkpoint.h5')

        ###############################################################################
        # can alternatively used the weights provided in the repo. If so it should get
        # you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
        # system.
        ###############################################################################

        # WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5 
        # model.load_weights(WEIGHTS_PATH)

        ###############################################################################
        # make prediction on test set.
        ###############################################################################

        probs       = model.predict(X_test)
        preds       = probs.argmax(axis = -1)  
        acc         = np.mean(preds == Y_test.argmax(axis=-1))
        acc_inter_lst.append(acc)

        # Test other classifiers
        
        preds_rg     = np.zeros(len(Y_test))
        # reshape back to (trials, channels, samples)
        X_train      = X_train.reshape(X_train.shape[0], chans, samples)
        X_test       = X_test.reshape(X_test.shape[0], chans, samples)

        clf1.fit(X_train, Y_train.argmax(axis = -1))
        preds_rg     = clf1.predict(X_test)

        acc2         = np.mean(preds_rg == Y_test.argmax(axis = -1))
        acc2_lst.append(acc2)

        preds_rg     = np.zeros(len(Y_test))

        clf2.fit(X_train, Y_train.argmax(axis = -1))
        preds_rg     = clf2.predict(X_test)

        acc3         = np.mean(preds_rg == Y_test.argmax(axis = -1))
        acc3_lst.append(acc3)
    print(acc_inter_lst)
    acc2_mean = sum(acc2_lst)/len(acc2_lst)
    acc3_mean = sum(acc3_lst)/len(acc3_lst)
    
    acc_inter_mean = sum(acc_inter_lst)/len(acc_inter_lst)
    return acc_inter_mean, acc2_mean, acc3_mean

##############################################################################
# Import dataset

dataset = BNCI2014001()
subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Dataset description: 
# MI
# 9 subjects
# 4 MI tasks
# 2 sessions/subject on different days
# 6 runs/session
# 48 trials(12 * 4 tasks)/run

##############################################################################
# Import paradigm:
    # Select number of classes
    # Filter data
    # Select time window for epochs
    # Resample the frequency

paradigm = MotorImagery(n_classes=4, fmin=4, fmax=40, tmin=0.5, tmax=2.5, resample= 128.0)

##############################################################################
# Assemble classifiers to compare with EEGNet

# CSP + LDA
lda = LDA()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Use scikit-learn Pipeline
clf1 = Pipeline([('CSP', csp), ('LDA', lda)])

# Riemannian classifier
# Assemble feature extractor 
cov = Covariances(estimator='scm')                                                      
ts = TangentSpace()                                                                     
ss = StandardScaler()                                                                   

# Assemble a classifier
rf = RandomForestClassifier()                                                           

# Use scikit-learn Pipeline
clf2 = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('rf', rf)])

##############################################################################
# Intra-subject classification

acc_intra_lst = []
acc2_lst = []
acc3_lst = []


for subj in subject_list:
    epochs, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[subj])
    epochs_T, epochs_E = np.split(epochs, 2)
    labels_T, labels_E = np.split(labels, 2)
    acc_intra, acc2, acc3 = intra_subject_classification(epochs_T, labels_T, clf1, clf2)
    acc_intra_lst.append(acc_intra)
    acc2_lst.append(acc2)
    acc3_lst.append(acc3)

acc_intra_mean = sum(acc_intra_lst)/len(acc_intra_lst)
acc2_intra_mean = sum(acc2_lst)/len(acc2_lst)
acc3_intra_mean = sum(acc3_lst)/len(acc3_lst)

print(acc_intra_mean)
print(acc2_intra_mean)
print(acc3_intra_mean)

# Plot table with results
fig, ax = plt.subplots(1,1)
data=[[round(acc_intra_mean, 6)],
      [round(acc2_intra_mean, 6)],
      [round(acc3_intra_mean, 6)]]
column_labels=["Within-Subject Classification"]
df=pd.DataFrame(data,columns=column_labels)
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df.values,colLabels=df.columns,rowLabels=["EEGNet-8,2 ","CSP + LDA","Riemannian"],loc="center")
plt.show()

##############################################################################
# Inter-subject classification

epochs_lst = []
labels_lst = []

for subj in subject_list:    
    epochs, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[subj])
    epochs_T, epochs_E = np.split(epochs, 2)
    labels_T, labels_E = np.split(labels, 2)    
    epochs_T.tolist()
    labels_T.tolist()
    epochs_lst = [*epochs_lst, *epochs_T]
    labels_lst = [*labels_lst, *labels_T]
epochs_lst = np.array(epochs_lst)
labels_lst = np.array(labels_lst)

acc_inter, acc2, acc3 = inter_subject_classification(epochs_lst, labels_lst, clf1, clf2)

print(acc_inter)
print(acc2)
print(acc3)

# Plot table with results
fig, ax = plt.subplots(1,1)
data=[[round(acc_inter, 6)],
      [round(acc2, 6)],
      [round(acc3, 6)]]
column_labels=["Cross-Subject Classification"]
df=pd.DataFrame(data,columns=column_labels)
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df.values,colLabels=df.columns,rowLabels=["EEGNet-8,2 ","CSP + LDA","Riemannian"],loc="center")
plt.show()