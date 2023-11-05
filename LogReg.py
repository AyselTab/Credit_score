#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


data = pd.read_csv(r"credit_score.csv")
data.head()


# In[3]:


pd.set_option('display.max_columns', 30)

data.describe(include='all')


# In[4]:


data.dtypes


# In[5]:


data.CreditScore.value_counts()


# In[6]:


data['score'] = np.where(data['CreditScore']!='Poor',0,1)
data


# In[7]:


data.score.value_counts()


# In[8]:


data.columns


# In[9]:


for i in data[['ID', 'CustomerID', 'Name', 'TypeofLoan', 'SSN']]:
    data.drop(i,axis=1,inplace=True)


# In[10]:


data.columns


# In[11]:


data.isnull().sum()


# In[12]:


for i in data[['MonthlyInhandSalary','NumofDelayedPayment','ChangedCreditLimit','NumCreditInquiries','Amountinvestedmonthly','MonthlyBalance','Occupation']]:
    if data[i].dtype == 'object':
        data[i].fillna(data[i].mode()[0], inplace=True)
    else:
        data[i].fillna(data[i].mean(), inplace=True)


# In[13]:


data.isnull().sum()


# In[14]:


corr = data.corr()['score']

high_corr_feats = corr[abs(corr) > 0.01].index.tolist()

high_corr_feats


# In[15]:


data.dtypes


# In[16]:


data = data[['MonthlyInhandSalary','Delayfromduedate','ChangedCreditLimit','OutstandingDebt','CreditUtilizationRatio','Amountinvestedmonthly',
'MonthlyBalance','Occupation','Month','PaymentofMinAmount','PaymentBehaviour','score']]

data


# In[17]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = data[['Delayfromduedate','ChangedCreditLimit','OutstandingDebt','Amountinvestedmonthly',
'MonthlyBalance']]

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]

vif["Features"] = variables.columns

vif


# In[18]:


data = data[['Delayfromduedate','ChangedCreditLimit','OutstandingDebt','Amountinvestedmonthly',
'MonthlyBalance','Occupation','Month','PaymentofMinAmount','PaymentBehaviour','score']]

data


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt

for i in data[['Delayfromduedate','ChangedCreditLimit','OutstandingDebt','Amountinvestedmonthly','MonthlyBalance']]:
    sns.boxplot(data=data,x=data[i])
    plt.show()


# In[20]:


q1=data.quantile(0.25)
q3=data.quantile(0.75)
IQR=q3-q1

Lower = q1-1.5*IQR
Upper = q3+1.5*IQR


# In[21]:


for i in data[['Delayfromduedate','ChangedCreditLimit','OutstandingDebt','Amountinvestedmonthly','MonthlyBalance']]:
    data[i] = np.where(data[i] > Upper[i],Upper[i],data[i])
    data[i] = np.where(data[i] < Lower[i],Lower[i],data[i])


# In[22]:


data.columns


# In[23]:


ranges = [-np.inf, data['Delayfromduedate'].quantile(0.25), data['Delayfromduedate'].quantile(0.5), data['Delayfromduedate'].quantile(0.75), np.inf]  # np.inf for infinity
data['Delayfromduedate_category'] = pd.cut(data['Delayfromduedate'], bins=ranges)

grouped = data.groupby(['Delayfromduedate_category', 'score'])['score'].count().unstack().reset_index()

grouped['positive_prop'] = grouped[0] / grouped[0].sum()
grouped['negative_prop'] = grouped[1] / grouped[1].sum()

grouped['woe'] = np.log(grouped['positive_prop'] / grouped['negative_prop'])
    
grouped.rename(columns={'woe':'Delayfromduedate_woe'}, inplace=True)
data = data.merge(grouped[['Delayfromduedate_category','Delayfromduedate_woe']], how='left', on='Delayfromduedate_category')

# --------------------------------------------------------------------------------------------------------------
ranges = [-np.inf, data['ChangedCreditLimit'].quantile(0.25), data['ChangedCreditLimit'].quantile(0.5), data['ChangedCreditLimit'].quantile(0.75), np.inf]  # np.inf for infinity
data['ChangedCreditLimit_category'] = pd.cut(data['ChangedCreditLimit'], bins=ranges)

grouped = data.groupby(['ChangedCreditLimit_category', 'score'])['score'].count().unstack().reset_index()

grouped['positive_prop'] = grouped[0] / grouped[0].sum()
grouped['negative_prop'] = grouped[1] / grouped[1].sum()
    
grouped['woe'] = np.log(grouped['positive_prop'] / grouped['negative_prop'])
    
grouped.rename(columns={'woe':'ChangedCreditLimit_woe'}, inplace=True)
data = data.merge(grouped[['ChangedCreditLimit_category','ChangedCreditLimit_woe']], how='left', on='ChangedCreditLimit_category')
# --------------------------------------------------------------------------------------------------------------
ranges = [-np.inf, data['OutstandingDebt'].quantile(0.25), data['OutstandingDebt'].quantile(0.5), data['OutstandingDebt'].quantile(0.75), np.inf]  # np.inf for infinity
data['OutstandingDebt_category'] = pd.cut(data['OutstandingDebt'], bins=ranges)
    
grouped = data.groupby(['OutstandingDebt_category', 'score'])['score'].count().unstack().reset_index()


grouped['positive_prop'] = grouped[0] / grouped[0].sum()
grouped['negative_prop'] = grouped[1] / grouped[1].sum()

grouped['woe'] = np.log(grouped['positive_prop'] / grouped['negative_prop'])
    
grouped.rename(columns={'woe':'OutstandingDebt_woe'}, inplace=True)
data = data.merge(grouped[['OutstandingDebt_category','OutstandingDebt_woe']], how='left', on='OutstandingDebt_category')
# --------------------------------------------------------------------------------------------------------------
ranges = [-np.inf, data['Amountinvestedmonthly'].quantile(0.25), data['Amountinvestedmonthly'].quantile(0.5), data['Amountinvestedmonthly'].quantile(0.75), np.inf]  # np.inf for infinity
data['Amountinvestedmonthly_category'] = pd.cut(data['Amountinvestedmonthly'], bins=ranges)

grouped = data.groupby(['Amountinvestedmonthly_category', 'score'])['score'].count().unstack().reset_index()


grouped['positive_prop'] = grouped[0] / grouped[0].sum()
grouped['negative_prop'] = grouped[1] / grouped[1].sum()

grouped['woe'] = np.log(grouped['positive_prop'] / grouped['negative_prop'])
    
grouped.rename(columns={'woe':'Amountinvestedmonthly_woe'}, inplace=True)
data = data.merge(grouped[['Amountinvestedmonthly_category','Amountinvestedmonthly_woe']], how='left', on='Amountinvestedmonthly_category')
# --------------------------------------------------------------------------------------------------------------
ranges = [-np.inf, data['MonthlyBalance'].quantile(0.25), data['MonthlyBalance'].quantile(0.5), data['MonthlyBalance'].quantile(0.75), np.inf]  # np.inf for infinity
data['MonthlyBalance_category'] = pd.cut(data['MonthlyBalance'], bins=ranges)
    

grouped = data.groupby(['MonthlyBalance_category', 'score'])['score'].count().unstack().reset_index()


grouped['positive_prop'] = grouped[0] / grouped[0].sum()
grouped['negative_prop'] = grouped[1] / grouped[1].sum()
    
grouped['woe'] = np.log(grouped['positive_prop'] / grouped['negative_prop'])
    
grouped.rename(columns={'woe':'MonthlyBalance_woe'}, inplace=True)
data = data.merge(grouped[['MonthlyBalance_category','MonthlyBalance_woe']], how='left', on='MonthlyBalance_category')

data


# In[24]:


data.ChangedCreditLimit_category.value_counts()


# In[25]:


grouped = data.groupby(['Occupation', 'score'])['score'].count().unstack().reset_index()


grouped['positive_prop'] = grouped[0] / grouped[0].sum()
grouped['negative_prop'] = grouped[1] / grouped[1].sum()

grouped['woe'] = np.log(grouped['positive_prop'] / grouped['negative_prop'])
    
grouped.rename(columns={'woe':'Occupation_woe'}, inplace=True)
data = data.merge(grouped[['Occupation','Occupation_woe']], how='left', on='Occupation')
#-------------------------------------------------------------------------------------------------------------
grouped = data.groupby(['Month', 'score'])['score'].count().unstack().reset_index()

grouped['positive_prop'] = grouped[0] / grouped[0].sum()
grouped['negative_prop'] = grouped[1] / grouped[1].sum()

grouped['woe'] = np.log(grouped['positive_prop'] / grouped['negative_prop'])
    
grouped.rename(columns={'woe':'Month_woe'}, inplace=True)
data = data.merge(grouped[['Month','Month_woe']], how='left', on='Month')
#---------------------------------------------------------------------------------------------------------------
grouped = data.groupby(['PaymentofMinAmount', 'score'])['score'].count().unstack().reset_index()

grouped['positive_prop'] = grouped[0] / grouped[0].sum()
grouped['negative_prop'] = grouped[1] / grouped[1].sum()
    
grouped['woe'] = np.log(grouped['positive_prop'] / grouped['negative_prop'])
    
grouped.rename(columns={'woe':'PaymentofMinAmount_woe'}, inplace=True)
data = data.merge(grouped[['PaymentofMinAmount','PaymentofMinAmount_woe']], how='left', on='PaymentofMinAmount')

#--------------------------------------------------------------------------------------------------------------------
grouped = data.groupby(['PaymentBehaviour', 'score'])['score'].count().unstack().reset_index()


grouped['positive_prop'] = grouped[0] / grouped[0].sum()
grouped['negative_prop'] = grouped[1] / grouped[1].sum()

grouped['woe'] = np.log(grouped['positive_prop'] / grouped['negative_prop'])
    
grouped.rename(columns={'woe':'PaymentBehaviour_woe'}, inplace=True)
data = data.merge(grouped[['PaymentBehaviour','PaymentBehaviour_woe']], how='left', on='PaymentBehaviour')

data


# In[26]:


data.columns


# In[27]:


data_fin = data[['Delayfromduedate','Delayfromduedate_woe', 'ChangedCreditLimit','ChangedCreditLimit_woe', 'OutstandingDebt','OutstandingDebt_woe',
       'Amountinvestedmonthly','Amountinvestedmonthly_woe', 'MonthlyBalance','MonthlyBalance_woe',
       'Occupation','Occupation_woe', 'Month','Month_woe', 'PaymentofMinAmount','PaymentofMinAmount_woe',
       'PaymentBehaviour','PaymentBehaviour_woe', 'score']]
data_fin


# In[28]:


inputs = data[['Delayfromduedate_woe', 'ChangedCreditLimit_woe', 'OutstandingDebt_woe',
       'Amountinvestedmonthly_woe', 'MonthlyBalance_woe',
       'Occupation_woe', 'Month_woe', 'PaymentofMinAmount_woe',
       'PaymentBehaviour_woe']]
output = data['score']


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(inputs, output, test_size=0.3, random_state=42)


# In[30]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def evaluate(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    roc_prob = roc_auc_score(y_test, y_prob)
    
    gini_prob = roc_prob*2-1
    
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print('Model Performance')

    print('Gini prob is', gini_prob*100)
    print(report)
    print(confusion_matrix)


# In[31]:


clf = LogisticRegression()
clf.fit(X_train, y_train)


# In[32]:


result = evaluate(clf, X_test, y_test)


# In[33]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

y_prob = clf.predict_proba(X_test)[:,1]

roc_auc = roc_auc_score(y_test, y_prob)
gini = (2*roc_auc_score(y_test, y_prob))-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Auc = %0.2f)' % roc_auc)
plt.plot(fpr, tpr, label='(Gini = %0.2f)' % gini)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='lower right')
plt.show()


# In[34]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc

variables = []
train_gini_scores = []
test_gini_scores = []


for i in X_train.columns:
    X_train_single_var = X_train[[i]]
    X_test_single_var = X_test[[i]]

    clf.fit(X_train_single_var, y_train)
    y_pred_train_single_var = clf.predict_proba(X_train_single_var)[:, 1]

    train_roc = roc_auc_score(y_train, y_pred_train_single_var)

    train_gini = 2 * train_roc - 1

    y_pred_test_single_var = clf.predict_proba(X_test_single_var)[:, 1]

    test_roc = roc_auc_score(y_test, y_pred_test_single_var)

    test_gini = 2 * test_roc - 1

    variables.append(i)
    train_gini_scores.append(train_gini)
    test_gini_scores.append(test_gini)

results_df = pd.DataFrame({
    'Variable': variables,
    'Train Gini': train_gini_scores,
    'Test Gini': test_gini_scores
})

results_df_sorted = results_df.sort_values(by='Test Gini', ascending=False)

pd.options.display.float_format = '{:.4%}'.format


results_df_sorted


# In[36]:


import pandas as pd

test_data = pd.read_excel(r'test_data_LR.xlsx')

test_data


# In[37]:


data


# In[38]:


data.Delayfromduedate_category.value_counts()


# In[39]:


ranges = [-np.inf, data['Delayfromduedate'].quantile(0.25), data['Delayfromduedate'].quantile(0.5), data['Delayfromduedate'].quantile(0.75), np.inf]  # np.inf for infinity
test_data['Delayfromduedate_category'] = pd.cut(test_data['Delayfromduedate'], bins=ranges)

test_data


# In[40]:


test_data = test_data.merge(data[['Delayfromduedate_category','Delayfromduedate_woe']].drop_duplicates(subset='Delayfromduedate_category'), how='left', on='Delayfromduedate_category')

test_data


# In[41]:


test_data.isnull().sum()


# In[42]:


ranges = [-np.inf, data['ChangedCreditLimit'].quantile(0.25), data['ChangedCreditLimit'].quantile(0.5), data['ChangedCreditLimit'].quantile(0.75), np.inf]  # np.inf for infinity
test_data['ChangedCreditLimit_category'] = pd.cut(test_data['ChangedCreditLimit'], bins=ranges)

ranges = [-np.inf, data['OutstandingDebt'].quantile(0.25), data['OutstandingDebt'].quantile(0.5), data['OutstandingDebt'].quantile(0.75), np.inf]  # np.inf for infinity
test_data['OutstandingDebt_category'] = pd.cut(test_data['OutstandingDebt'], bins=ranges)

ranges = [-np.inf, data['Amountinvestedmonthly'].quantile(0.25), data['Amountinvestedmonthly'].quantile(0.5), data['Amountinvestedmonthly'].quantile(0.75), np.inf]  # np.inf for infinity
test_data['Amountinvestedmonthly_category'] = pd.cut(test_data['Amountinvestedmonthly'], bins=ranges)

ranges = [-np.inf, data['MonthlyBalance'].quantile(0.25), data['MonthlyBalance'].quantile(0.5), data['MonthlyBalance'].quantile(0.75), np.inf]  # np.inf for infinity
test_data['MonthlyBalance_category'] = pd.cut(test_data['MonthlyBalance'], bins=ranges)

test_data


# In[43]:


test_data = test_data.merge(data[['ChangedCreditLimit_category','ChangedCreditLimit_woe']].drop_duplicates(subset='ChangedCreditLimit_category'), how='left', on='ChangedCreditLimit_category')
test_data = test_data.merge(data[['OutstandingDebt_category','OutstandingDebt_woe']].drop_duplicates(subset='OutstandingDebt_category'), how='left', on='OutstandingDebt_category')
test_data = test_data.merge(data[['Amountinvestedmonthly_category','Amountinvestedmonthly_woe']].drop_duplicates(subset='Amountinvestedmonthly_category'), how='left', on='Amountinvestedmonthly_category')
test_data = test_data.merge(data[['MonthlyBalance_category','MonthlyBalance_woe']].drop_duplicates(subset='MonthlyBalance_category'), how='left', on='MonthlyBalance_category')
test_data = test_data.merge(data[['Occupation','Occupation_woe']].drop_duplicates(subset='Occupation'), how='left', on='Occupation')
test_data = test_data.merge(data[['Month','Month_woe']].drop_duplicates(subset='Month'), how='left', on='Month')
test_data = test_data.merge(data[['PaymentofMinAmount','PaymentofMinAmount_woe']].drop_duplicates(subset='PaymentofMinAmount'), how='left', on='PaymentofMinAmount')
test_data = test_data.merge(data[['PaymentBehaviour','PaymentBehaviour_woe']].drop_duplicates(subset='PaymentBehaviour'), how='left', on='PaymentBehaviour')


# In[44]:


test_data


# In[45]:


test_data.isnull().sum()


# In[46]:


test_data.columns


# In[57]:


test_data_woe = test_data[['CustomerID','Delayfromduedate_woe', 'ChangedCreditLimit_woe', 'OutstandingDebt_woe',
       'Amountinvestedmonthly_woe', 'MonthlyBalance_woe', 'Occupation_woe',
       'Month_woe', 'PaymentofMinAmount_woe', 'PaymentBehaviour_woe']]

test_data_woe


# In[58]:


inputs.columns


# In[59]:


test_data_woe.columns


# In[60]:


prob = clf.predict_proba(test_data_woe.iloc[:,1:])[:,1]


# In[ ]:


test_data_woe['PD'] = prob

test_data_woe


# In[ ]:




