#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IPOs Initial Return Prediction
#RIN: 661967126
#Jiawen Zhang
#In order to find all IPOs from 2000-01-01 -- 2019-12-31 in U.S. as well as its initial return,
#download all stock symbols in
#1. NASDAQ Stock Exchange
#2. New York Stock Exchange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import warnings
from matplotlib.pyplot import figure
from gapminder import gapminder # data set
warnings.filterwarnings('ignore')
#PartA: Data preparation
#1 closing price from yfinance
#from yahoo finance
# r_data = pd.read_csv('~/DeskTop/usticker.csv')
# r_data[:5]
# r_data.iloc[2833,0]
# tickers = []
# for i in range(0,8889):
#     ticker = r_data.iloc[i,0]
#     tickers.append(ticker)
# # plz donnot run this part of code since it may take more than 25 minutes
# # I've saved this data to csv and import it again to this project
# ipo_price = []
# ipo_close = []
# ipo_ticker = []
# for stock in tickers:
#     tickerData = yf.Ticker(stock)
#     data = tickerData.history(period = '1d', start = '1950-01-01',
#                               end = '2020-04-12')
#     if len(data) > 0:
#         if data.index[1] > datetime.datetime.strptime('2000-01-01', '%Y-%m-%d'):
#             ipo_price.append(data.iloc[0,0])
#             ipo_close.append(data.iloc[0,3])
#             ipo_ticker.append(stock)
# ipo = pd.DataFrame(ipo_price,columns = ['op_price'], index = ipo_ticker)
# ipo['cl_price'] = ipo_close
# ipo.to_csv('~/DeskTop/ipo.csv')
op_cl1 = pd.read_csv('~/DeskTop/ipo.csv')
op_cl1.iloc[:,0]
op_cl = op_cl1.copy()
op_cl.index = op_cl1.iloc[:,0]
op_cl
op_cl = op_cl1.copy()
op_cl.index = op_cl1.iloc[:,0]
op_cl
op_cl['Initial_Return'] = (op_cl['ipo_close'] - op_cl['ipo_price'])/(op_cl['ipo_price'])



#2 firm specific data
#from Gurufocus and hoovers
guru = pd.read_csv('~/DeskTop/guru.csv')
guru
offering = pd.read_csv('~/DeskTop/offersharing.csv')
offering


#join above two dataframes on column "symbol" to get all features in 
#1 Firm specific characteristics
#2 IPO offering price, IPO first-day closing price & number of shares offered
firm_ipo  = guru.set_index('Symbol').join(op_cl.set_index('Symbol'))
firm_ipo  = firm_ipo.reset_index().merge(offering, left_on='Symbol',right_on='Symbol').set_index('Symbol')
firm_ipo


# In[ ]:


#3 market factor at the point of IPO issueing
#from yahoo finance
#download data of S&P500 (2010-02-16 -- 2020-03-13)
#we need one-day-ahead(Mkt1), one-week-rolling(Mkt7) & one-month-rolling(Mkt30) market return factor
mkt_fct = pd.read_csv('~/DeskTop/marketfactor.csv')
mkt_fct
all_data = firm_ipo.reset_index().merge(mkt_fct, left_on='IPO Date',right_on='IPO Date').set_index('Symbol')
all_data
all_data.to_csv('~/DeskTop/with_symbol.csv')


# In[73]:


#4 data cleaning
# (1) check the percentage of missing value
## drop some useless string columns directly by excel
# also modifiy the initial return: initial return = ((first-day closing price / IPO offering price) - 1)*100
alldt = pd.read_csv('~/DeskTop/alldt.csv')
def missing_table(df):
    """Display the number of missing values and percentage in each features"""
    missing_series = df.isnull().sum()
    missing_percentage = df.isnull().sum()/len(df)*100
    missing_df = pd.concat([missing_series,missing_percentage],axis=1)
    missing_df.columns = ['missing_values','missing_percentage(%)']
    missing_df.sort_values(by = 'missing_percentage(%)', ascending=False, inplace = True)
    return missing_df
missing_df = missing_table(alldt)
missing_df.head(15)  
missing_df = missing_df.head(7)
missing_df["concat"] = "0"
missing_df
for i in range(0, 7):
    missing_df["concat"][i] = missing_df.index[i] + """
    """  " Count:" + str(missing_df["missing_values"][i])
# missing_df["concat"] = missing_df.index + str(missing_df["missing_values"])
missing_df


# In[74]:


plt.figure(figsize=(20,10))
ax = plt.subplot(111, polar=True)
plt.axis('off')
upperLimit = 160
lowerLimit = 0
labelPadding = 4
max = missing_df['missing_values'].max()
slope = (max - lowerLimit) / max
heights = slope * missing_df.missing_values + lowerLimit
width = 2*np.pi / len(missing_df.index)
indexes = list(range(1, len(missing_df.index)+1))
angles = [element * width for element in indexes]
bars = ax.bar(
    x=angles, 
    height=heights, 
    width=width, 
    bottom=lowerLimit,
    linewidth=2, 
    edgecolor="white",
    color="#fc8c03",
)
for bar, angle, height, label in zip(bars,angles, heights, missing_df["concat"]):
    rotation = np.rad2deg(angle)

    # Flip some labels upside down
    alignment = ""
    if angle >= np.pi/2 and angle < 3*np.pi/2:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"

    # Finally add the labels
    ax.text(
        x=angle, 
        y=lowerLimit + bar.get_height() + labelPadding, 
        s=label, 
        ha=alignment, 
        va='center', 
        rotation=rotation, 
        rotation_mode="anchor") 
plt.savefig('circular.png')


# In[56]:


# remove the 10 features holding the largest perventage of missing_value
missing_df.head(8).index
missing_df.head(8).to_csv('~/DeskTop/missing.csv')
alldt['Initial_Return'].describe()


# In[77]:


alldt


# In[81]:



# # data
# data = gapminder.loc[gapminder.year == 2007]

# # use the scatterplot function to build the bubble map
# sns.scatterplot(data= alldt, x=alldt.index, y="Initial_Return", size="pop", legend=False, sizes=(20, 2000))

# # show the graph
# plt.show()
sns.regplot(x=alldt.index, y=alldt["Initial_Return"])


# In[89]:


df_clear = alldt.drop(alldt[alldt['Initial_Return'] > 2].index)
df_clear.boxplot(column=["Initial_Return"], notch=True)


# In[102]:


#delete some outlier for our target prediction: initial return
# (2) check the correlation between features and initial returns
outlier_dlted = pd.read_csv('~/DeskTop/alldt_delete.csv')
# the percentage of missing value is not very huge 
# conduct some data visualization to check the potential relationship
outlier_dlted.to_csv('~/DeskTop/adata.csv')
temp1 = pd.read_csv('~/DeskTop/haha.csv')
temp1 = temp1.drop(temp1[temp1['Initial_Return'] > 2].index)
# df_clear.boxplot(column=["Initial_Return"], notch=True)
temp1
len(temp1)
tempx = temp1.drop(['Initial_Return'], axis = 1)
tempy = temp1['Initial_Return']


# In[103]:


temp1


# In[142]:


corr = temp1.corr().iloc[:,-1]


Correlation = pd.DataFrame({'Correlation * 100':temp1.corr().iloc[:,-1]*100},
                         index = temp1.columns)
Correlation = Correlation.sort_values(by = 'Correlation * 100',axis = 0,ascending = False)
Correlation

top = Correlation.iloc[1:16, :]

top


plt.bar(top.index, top["Correlation * 100"], color = "#fc8c03")
 
# Add title and axis names
# plt.xlabel('Feature')
plt.ylabel('Correlation value * 100')
 
# Create names on the x axis
plt.xticks(np.arange(15), top.index, rotation=90)
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
# Show graph
plt.show()

draw = temp1.drop(temp1[temp1['Initial_Return'] > 0.2].index)

plt.subplot(221)
plt.plot( 'Initial_Return', 'Mkt1', data=draw, marker='o', alpha=0.4)
plt.title("A subplot with 2 lines")
# second line
plt.subplot(222)
plt.plot( 'Initial_Return','Mkt7', data=draw, linestyle='none', marker='o', color="orange", alpha=0.3)

# third line
plt.subplot(223)
plt.plot( 'Initial_Return','Mkt30', data=draw, linestyle='none', marker='o', color="red", alpha=0.3)

# Show the graph
figure(figsize=(16, 12), dpi=80)
plt.show()

missing_fn = missing_table(temp1)
missing_fn


X_train, X_test, y_train, y_test = train_test_split(tempx, tempy, test_size = 0.25, random_state=1)
xgb = XGBRegressor(max_depth=20, 
                    learning_rate=0.1, 
                    n_estimators=100, 
                    random_state=1).fit(X_train, y_train)
predict_xgb = xgb.predict(X_test)
mse_xgb =mean_squared_error(y_test,predict_xgb)
print('mse for xgboostregression is',mse_xgb)


#Crossvalidation
kfold = KFold(n_splits = 10, random_state=0)
cross_xgb = cross_val_score(xgb,tempx,tempy,cv=kfold,scoring='neg_mean_squared_error')
cross_xgb.mean()
#2 Random Forest
#We need to fill NAN value at first
missing_df.head(10)
#the columns with missing values are :
#%Below Historical High EV-to-EBITDA [column 4] , % Above Historical Low EV-to-Revenue #[column 2], % Below Historical High EV-to-Revenue [column 5]
#%Below Historical High EV-to-EBIT [column 3], % Above Historical Low EV-to-EBITDA #[column 1], Debt-to-EBITDA [column 31], Cash-to-Debt [column 22], % Above Historical #Low EV-to-EBIT [column 0]
#mpute NAN with Median, why? The std of these columns are huge, mean is not the #objective value.


temp_rf = temp1.copy()
temp_rfx = temp_rf.drop(['Initial_Return'], axis = 1)
temp_rfy = temp_rf['Initial_Return']
temp_rfx.iloc[:,0] = temp_rfx.iloc[:,0].replace(np.nan,temp_rfx.iloc[:,0].median())
temp_rfx.iloc[:,4] = temp_rfx.iloc[:,4].replace(np.nan,temp_rfx.iloc[:,4].median())
temp_rfx.iloc[:,2] = temp_rfx.iloc[:,2].replace(np.nan,temp_rfx.iloc[:,2].median())
temp_rfx.iloc[:,5] = temp_rfx.iloc[:,5].replace(np.nan,temp_rfx.iloc[:,5].median())
temp_rfx.iloc[:,3] = temp_rfx.iloc[:,3].replace(np.nan,temp_rfx.iloc[:,3].median())
temp_rfx.iloc[:,1] = temp_rfx.iloc[:,1].replace(np.nan,temp_rfx.iloc[:,1].median())
temp_rfx.iloc[:,31] = temp_rfx.iloc[:,31].replace(np.nan,temp_rfx.iloc[:,31].median())
temp_rfx.iloc[:,22] = temp_rfx.iloc[:,22].replace(np.nan,temp_rfx.iloc[:,22].median())
X_trainrf, X_testrf, y_trainrf, y_testrf = train_test_split(temp_rfx, temp_rfy, test_size = 0.25, random_state=1)
m_rf = RandomForestRegressor(max_depth=5,random_state=1).fit(X_trainrf,y_trainrf)
predict_rf = m_rf.predict(X_testrf)
mse_rf = mean_squared_error(y_testrf,predict_rf)


print('mse for random forest regression is',mse_rf)


#Crossvalidation
cross_rf = cross_val_score(m_rf,temp_rfx,temp_rfy,cv=kfold,scoring='neg_mean_squared_error')
cross_rf.mean()


#3 Linear Regression
#Also build a linear regression to check whether the improvement exist
temp_lry = temp_rfy
temp_lrx = temp_rfx
X_trainlr, X_testlr, y_trainlr, y_testlr = train_test_split(temp_lrx, temp_lry, test_size = 0.25, random_state=1)
m_lr = LinearRegression().fit(X_trainlr, y_trainlr)
predict_lr = m_lr.predict(X_testlr)
mse_lr = mean_squared_error(y_testlr,predict_lr)
print('mse for linear regression is',mse_lr)

#Crossvalidation
cross_lr = cross_val_score(m_lr,temp_lrx,temp_lry,cv=kfold,scoring='neg_mean_squared_error')
cross_lr.mean()

#4 Ridge model
ridge = Ridge()
parameters = {'alpha': [1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,2,3,4,5,10,15,20]}
pa = GridSearchCV(ridge, parameters, scoring = 'neg_mean_absolute_error',cv=20)
m_r = pa.fit(X_trainlr, y_trainlr)
predict_r = m_r.predict(X_testlr)
mse_r = mean_squared_error(y_testlr,predict_r)
print('mse for ridge model is',mse_r)
#Crossvalidation
cross_r = cross_val_score(m_r,temp_lrx,temp_lry,cv=kfold,scoring='neg_mean_squared_error')
cross_r.mean()

#5 XGBoost without missing value
xgb2 = XGBRegressor(max_depth=20, 
                    learning_rate=0.1, 
                    n_estimators=100, 
                    random_state=1).fit(X_trainrf, y_trainrf)
predict_xgb2 = xgb2.predict(X_testrf)
mse_xgb2 =mean_squared_error(y_testrf,predict_xgb2)
print('mse for xgboostregression is',mse_xgb2)
#Crossvalidation
cross_xgb2 = cross_val_score(xgb2,X_trainrf,y_trainrf,cv=kfold,scoring='neg_mean_squared_error')
cross_xgb2.mean()

print('mse for xgboostregression is',mse_xgb2)

#Crossvalidation
cross_xgb2 = cross_val_score(xgb2,X_trainrf,y_trainrf,cv=kfold,scoring='neg_mean_squared_error')
cross_xgb2.mean()
param_test = {'bootstrap': [True, False],
                 'max_depth': [10, 20, 30, 40, 50],
                 'max_features': list(range(5,36,5)),
                 'n_estimators': list(range(10,101,10))}
model = RandomForestRegressor()
kfold =  KFold(n_splits = 10, random_state=0)
rf_opt = GridSearchCV(estimator = model, 
                        param_grid = param_test, scoring= 'neg_mean_squared_error',verbose = 2,
                        n_jobs= -1,iid=False, cv = kfold)
rf_opt.fit(X_trainrf, y_trainrf)
rf_opt.best_params_
#PartD: Final random forest model
m_final = RandomForestRegressor(max_depth=50,max_features = 20, n_estimators = 30, bootstrap = 'True', random_state=1).fit(X_trainrf,y_trainrf)
predict_final = m_final.predict(X_testrf)
mse_final = mean_squared_error(y_testrf,predict_final)
print('mse for random forest regression after hyperparameter tuning is',mse_final)


rf_opt.best_params_

Importance = pd.DataFrame({'Importance':m_final.feature_importances_*100},
                         index = temp_rfx.columns)
imp_data = Importance.sort_values(by = 'Importance',axis = 0,ascending = False)
imp_data
imp_data20 = imp_data.head(15)
fig = imp_data20.plot(kind = 'barh',color = 'orange')
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
plt.grid()
imp_data20
firm_ipo.to_csv('~/DeskTop/bfir.csv')
imp_data.to_csv('~/DeskTop/importance.csv')