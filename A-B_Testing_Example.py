#A-B testing for Click Through Ratio (CTR) based on presumed enhancement.
#Scenario: A website conducted an AB test for an enhancement. 
#Does the enhancement impact CTR?
#Null Hypothesis: The new enhancement of the website does not influence CTR (𝜇ₐ = 𝜇ᵦ).
#Alternative Hypothesis: The new enhancement of the website does influence CTR (𝜇ₐ ≠ 𝜇ᵦ).
#p value of 0.05 as significance level.
#data set courtesy of https://www.kaggle.com/datasets/sergylog/split-experiment-results
#Note: Column name "Views" changed to "Impressions" for clarity purposes of CTR calculation.
import pandas as pd
import numpy as np
from scipy import stats as st
from scipy.stats import t 
import matplotlib.pyplot as plt


url = ('https://raw.githubusercontent.com/A-Bin1/Statistical-Samples/main/ab_website_result_set.csv')
website_ab_df = pd.read_csv(url)

website_ab_df.head()
#    user_id    group  Impressions  Clicks
# 0        1  control            3       0
# 1        2  control            2       0
# 2        3  control            2       0
# 3        4  control            2       0
# 4        5  control            5       0

#filter data function
def filter_data(df, groupName, colName):
   df_col = df.loc[df['group'] == str(groupName), str(colName)]
   return df_col


#CTR Column:
website_ab_df['CTR'] = website_ab_df['Clicks']/website_ab_df['Impressions']

website_ab_df.head(6)
#    user_id    group  Impressions  Clicks       CTR
# 0        1  control            3       0  0.000000
# 1        2  control            2       0  0.000000
# 2        3  control            2       0  0.000000
# 3        4  control            2       0  0.000000
# 4        5  control            5       0  0.000000
# 5        6  control            3       1  0.333333

#calculate CTR (click-through ratio): 
#CTR = Total number of clicks / Total number of impressions
#formal results in function:
def CTR_summary(group, df):
    clicks = df.loc[df['group'] == str(group), 'Clicks'].sum()
    impressions = df.loc[df['group'] == str(group), 'Impressions'].sum()
    print("total clicks for group " + str(group)+ ": " + str(clicks)) 
    print("total clicks for group " + str(group)+ ": " + str(impressions))
    print('CTR:'+ str(clicks/impressions))
    return clicks/impressions

#strictly metric calculation
def CTR(group, df):
    clicks = df.loc[df['group'] == str(group), 'Clicks'].sum()
    impressions = df.loc[df['group'] == str(group), 'Impressions'].sum()
    return clicks/impressions

#CTR for Control Group (A):
CTR_A_sum = CTR_summary('control', website_ab_df)
# total clicks for group control: 11008
# total clicks for group control: 226101
# CTR:0.048686206606781926

#CTR for Test Group (B):
CTR_B_sum = CTR_summary('test', website_ab_df)
# total clicks for group test: 14045
# total clicks for group test: 224936
# CTR:0.06243998292847743

diff = CTR_B_sum-CTR_A_sum
print(diff)
#0.013753776321695506
# appx 1.38% increase from control group A CTR.
#Determine if the population means have a normal distribution with a Histogram

#CTR per group
CTR_A = filter_data(website_ab_df, 'control', 'CTR')
CTR_B = filter_data(website_ab_df, 'test', 'CTR')

plt.hist(CTR_A, edgecolor='black', bins=5)
plt.title('CTR_A Distribution Shape', weight='bold', size=5)
plt.show()

plt.hist(CTR_B, edgecolor='black', bins=5)
plt.title('CTR_B Distribution Shape', weight='bold', size=5)
plt.show()

#Means for both groups appear skewed (not normally distributed). However, our sample sizes are
#45,000 per group, indicating sufficiently large samples. Due to the Central Limit Theorem,
#we can assume these groups are normal based on the sampling distribution of the mean 
#fitting a normal distribution.

#Perform a two sample t test (independent t test) for both groups (Control and Test Groups)

#function for two tailed t-test
def two_tailed_ttest(data, group1, group2, alpha):
    alpha = float(alpha)
    n1 = group1.size #sample size A
    n2 = group2.size #sample size B
    m1 = CTR('control', data) #meanA
    m2 = CTR('test', data) #meanB
    v1 = np.var(group1, ddof=1) #varianceA
    v2 = np.var(group2, ddof=1) #varianceB
    delta = m1-m2 #mean delta
    degf = group1.count() + group2.count() - 2 #degrees of freedom
    pooled_sd = np.sqrt(((n1-1)*(v1**2) + (n2-1)*(v2**2))/degf) #pooled standard dev
    tstat = delta / pooled_sd
    p = 2 * st.t.cdf(-abs(tstat), degf)
    # Confidence Interval upper and lower bounds: CI% = 1-alpha
    lower = delta - st.t.ppf(1-alpha,degf)*pooled_sd 
    upper = delta + st.t.ppf(1-alpha,degf)*pooled_sd
    return pd.DataFrame(np.array([tstat,degf,p,delta,lower,upper]).reshape(1,-1),
                         columns=['T-Statistic','DegFreedom','Two-tailed P Value','Difference in Mean','Lower','Upper']) 

two_tailed_ttest(website_ab_df,CTR_A, CTR_B, 0.05)
#    T-Statistic  DegFreedom  Two-tailed P Value  Difference in Mean     Lower     Upper
# 0    -0.654639     89998.0            0.512702           -0.013754 -0.048312  0.020804

#P-value > alpha and therefore we fail to reject the Null Hypothesis.
#the enhancement cannot be proven to impact the CTR.

