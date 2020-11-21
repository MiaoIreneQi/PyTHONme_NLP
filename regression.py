#sp500, compound_script, compound_tweet
import statsmodels.formula.api as sm
analysis = pd.merge(sp_500, sentiment_combined, on = 'Date')
analysis.rename(columns = {'S&P 500 index':'SP_500_index','percentage change' : 'percentage_change'}, inplace = True)
result = sm.ols(formula="percentage_change ~ compound_script + compound_tweet", data=analysis).fit()
print(result.params)
print(result.summary())

#VIX 
vix = pd.read_excel('VIX.xlsx')
analysis_sp500 = pd.read_excel('analysis.xlsx')


vix.rename(columns = {'date' : 'Date'}, inplace = True)
analysis_all = pd.merge(vix, analysis_sp500, on = 'Date')

analysis_all.insert(loc=0, column = 'absolute_compound_script', value = analysis_all['compound_script'])
analysis_all['absolute_compound_script'] = analysis_all['compound_script'].abs()

analysis_all.insert(loc=0, column = 'absolute_compound_tweet', value = analysis_all['compound_tweet'])
analysis_all['absolute_compound_tweet'] = analysis_all['compound_tweet'].abs()


result = sm.ols(formula="index ~ compound_script + compound_tweet", data=analysis_all).fit()
result = sm.ols(formula="index ~ absolute_compound_script + absolute_compound_tweet", data=analysis_all).fit()

import matplotlib.pyplot as plt
y_2 = analysis_all['percentage_change']
y_3 = analysis_all['index']
x_1 = analysis_all['absolute_compound_script']
x_2 = analysis_all['absolute_compound_tweet']
x_3 = analysis_all['compound_script']
x_4 = analysis_all['compound_tweet']
plt.scatter(x_2, y_3)

plt.scatter(x_4, y_2)

plt.scatter(x_2, y_2)


#sp500 pc - absolute 

result = sm.ols(formula="percentage_change ~ absolute_compound_script + absolute_compound_tweet", data=analysis_all).fit()
result = sm.ols(formula="percentage_change ~ absolute_compound_script + absolute_compound_tweet", data=analysis_all).fit()
result = sm.ols(formula="percentage_change ~ compound_script + compound_tweet", data=analysis_all).fit()


print(result.summary())


#industry index
data_new = pd.read_excel('data_new.xlsx')
data_new.rename(columns = {'索引' : 'index', 'Dow_Jones_工业' : 'DJ_industrial', '美国:道琼斯公用事业平均指数' : 'DJ_public', '美国:威尔希尔美国房地产投资信托市场总指数': 'WS_housing', '美国:能源产业ETF波动率指数' : 'Energy_ETFVIX'}, inplace = True)
data_new.index.name = 'index'

data_new.Date = pd.to_datetime(data_new.Date)
analysis_all = pd.merge(analysis_all, data_new, on = 'Date')

#DJ_industrial, compound_script, compound_tweet
result = sm.ols(formula="pc_DJ_industrial ~ compound_script + compound_tweet", data=analysis_all).fit()
print(result.summary())

y_4 = analysis_all['pc_DJ_industrial']
plt.scatter(x_4,y_4)

#DJ_Industrial , absolute_compound_script, absolute_compound_tweet
result = sm.ols(formula="pc_DJ_industrial ~ absolute_compound_script + absolute_compound_tweet", data=analysis_all).fit()
print(result.summary())


plt.scatter(x_2,y_4)

#DJ_public , compound_script, compound_tweet
result = sm.ols(formula="pc_Dj_public ~ compound_script + compound_tweet", data=analysis_all).fit()
print(result.summary())

y_5 = analysis_all['pc_Dj_public']
plt.scatter(x_4,y_5)

#DJ_public , absolute_compound_script, absolute_compound_tweet
result = sm.ols(formula="pc_Dj_public ~ absolute_compound_script + absolute_compound_tweet", data=analysis_all).fit()
print(result.summary())


plt.scatter(x_2,y_5)

#WS_housing , compound_script, compound_tweet
result = sm.ols(formula="pc_WS_housing ~ compound_script + compound_tweet", data=analysis_all).fit()
print(result.summary())

y_6 = analysis_all['pc_WS_housing']
plt.scatter(x_4,y_6)

#WS_housing , absolute_compound_script, absolute_compound_tweet
result = sm.ols(formula="pc_WS_housing ~ absolute_compound_script + absolute_compound_tweet", data=analysis_all).fit()
print(result.summary())


plt.scatter(x_2,y_6)



#Energy_ETFVIX , compound_script, compound_tweet
analysis_all.rename(columns = {'美国:能源产业ETF波动率指数' : 'Energy_ETFVIX'}, inplace = True)

result = sm.ols(formula="Energy_ETFVIX ~ compound_script + compound_tweet", data=analysis_all).fit()
print(result.summary())

y_7 = analysis_all['Energy_ETFVIX']
plt.scatter(x_4,y_7)

#nergy_ETFVIX , absolute_compound_script, absolute_compound_tweet
result = sm.ols(formula="Energy_ETFVIX ~ absolute_compound_script + absolute_compound_tweet", data=analysis_all).fit()
print(result.summary())

plt.scatter(x_2,y_7)
