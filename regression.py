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
y_3 = analysis_all['index']
x_1 = analysis_all['absolute_compound_script']
x_2 = analysis_all['absolute_compound_tweet']
plt.scatter(x_2, y_3)

#sp500 pc - absolute 

result = sm.ols(formula="percentage_change ~ absolute_compound_script + absolute_compound_tweet", data=analysis_all).fit()

print(result.summary())
