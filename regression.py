import statsmodels.formula.api as sm
analysis = pd.merge(sp_500, sentiment_combined, on = 'Date')
analysis.rename(columns = {'S&P 500 index':'SP_500_index','percentage change' : 'percentage_change'}, inplace = True)
result = sm.ols(formula="percentage_change ~ compound_script + compound_tweet", data=analysis).fit()
print(result.params)
print(result.summary())
