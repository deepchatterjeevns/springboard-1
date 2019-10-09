#!/usr/bin/env python
# coding: utf-8

# # Inferential Statistics Ib - Frequentism

# ## Learning objectives

# Welcome to the second Frequentist inference mini-project! Over the course of working on this mini-project and the previous frequentist mini-project, you'll learn the fundamental concepts associated with frequentist inference. The following list includes the topics you will become familiar with as you work through these two mini-projects:
# * the _z_-statistic
# * the _t_-statistic
# * the difference and relationship between the two
# * the Central Limit Theorem, its assumptions and consequences
# * how to estimate the population mean and standard deviation from a sample
# * the concept of a sampling distribution of a test statistic, particularly for the mean
# * how to combine these concepts to calculate confidence intervals and p-values
# * how those confidence intervals and p-values allow you to perform hypothesis (or A/B) tests

# ## Prerequisites

# * what a random variable is
# * what a probability density function (pdf) is
# * what the cumulative density function is
# * a high-level sense of what the Normal distribution
# 
# If these concepts are new to you, please take a few moments to Google these topics in order to get a sense of what they are and how you might use them.
# 
# These two notebooks were designed to bridge the gap between having a basic understanding of probability and random variables and being able to apply these concepts in Python. This second frequentist inference mini-project focuses on a real-world application of this type of inference to give you further practice using these concepts. 

# In the previous notebook, we used only data from a known normal distribution. You'll now tackle real data, rather than simulated data, and answer some relevant real-world business problems using the data.

# ## Hospital medical charges

# Imagine that a hospital has hired you as their data analyst. An administrator is working on the hospital's business operations plan and needs you to help them answer some business questions. This mini-project, as well as the bootstrap and Bayesian inference mini-projects also found in this unit are designed to illustrate how each of the inferential statistics methods have their uses for different use cases. In this assignment notebook, you're going to use frequentist statistical inference on a data sample to answer the questions:
# * has the hospital's revenue stream fallen below a key threshold?
# * are patients with insurance really charged different amounts than those without?
# Answering that last question with a frequentist approach makes some assumptions, or requires some knowledge, about the two groups. In the next mini-project, you'll use bootstrapping to test that assumption. And in the final mini-project of the unit, you're going to create a model for simulating _individual_ charges (not a sampling distribution) that the hospital can use to model a range of scenarios.

# We are going to use some data on medical charges obtained from [Kaggle](https://www.kaggle.com/easonlai/sample-insurance-claim-prediction-dataset). For the purposes of this exercise, assume the observations are the result of random sampling from our one hospital. Recall in the previous assignment, we introduced the Central Limit Theorem (CLT), and how it tells us that the distributions of sample statistics approach a normal distribution as $n$ increases. The amazing thing about this is that it applies to the sampling distributions of statistics that have been calculated from even highly non-normal distributions of data. Remember, also, that hypothesis testing is very much based on making inferences about such sample statistics. You're going to rely heavily on the CLT to apply frequentist (parametric) tests to answer the questions in this notebook.

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from numpy.random import seed
medical = pd.read_csv('insurance2.csv')


# In[7]:


medical.shape


# In[8]:


medical.head(10)


# __Q:__ Plot the histogram of charges and calculate the mean and standard deviation. Comment on the appropriateness of these statistics for the data.

# __A:__

# In[9]:


_ = plt.hist(medical.charges, bins = 50)
_ = plt.xlabel('charges')
_ = plt.ylabel('charge amount in USD')
_ = plt.title('Medical Charges')


# In[10]:


charge_mean = np.mean(medical['charges'])
charge_stdev = np.std(medical['charges'], ddof = 1)
print('The mean is' + ' ' + str(charge_mean), 'with a standard deviation of' + ' '+ str(charge_stdev))


# __Q:__ The administrator is concerned that the actual average charge has fallen below 12000, threatening the hospital's operational model. On the assumption that these data represent a random sample of charges, how would you justify that these data allow you to answer that question? And what would be the most appropriate frequentist test, of the ones discussed so far, to apply?

# __A:__ The sample size n = 1338 allows for an unbiased estimate of the true mean of the population distribution. Our estimate won't be exact but will be close. The most appropriate test to apply is the one tailed z-statistic. 

# __Q:__ Given the nature of the administrator's concern, what is the appropriate confidence interval in this case? A one-sided or two-sided interval? Calculate the critical value and the relevant 95% confidence interval for the mean and comment on whether the administrator should be concerned?

# __A:__ Our distribution shown from the histogram above suggest that a one tailed confidence interval is the most appropriate:

# In[18]:


critical_value = t.ppf(.05,1337)
margin_of_error = (critical_value * charge_stdev)/np.sqrt(1338)
margin_of_error


# In[20]:


confidence_interval = [charge_mean + margin_of_error, charge_mean - margin_of_error]
confidence_interval


# The administrator should not be concerned since the chances of average charge is very unlikely to be less than $12,000. It is less than a 5% probability. 

# The administrator then wants to know whether people with insurance really are charged a different amount to those without.
# 
# __Q:__ State the null and alternative hypothesis here. Use the _t_-test for the difference between means where the pooled standard deviation of the two groups is given by
# \begin{equation}
# s_p = \sqrt{\frac{(n_0 - 1)s^2_0 + (n_1 - 1)s^2_1}{n_0 + n_1 - 2}}
# \end{equation}
# 
# and the *t* test statistic is then given by
# 
# \begin{equation}
# t = \frac{\bar{x}_0 - \bar{x}_1}{s_p \sqrt{1/n_0 + 1/n_1}}.
# \end{equation}
# 
# What assumption about the variances of the two groups are we making here?

# __A:__ 
# 
# Null: Patients without insurance are charged the same as those without insurance.
# 
# Alternative: Patients with insurance are charged differently than those without insurance.
# 
# We are assuming that the data is homogenous and the variances of the two groups are equal.

# __Q:__ Perform this hypothesis test both manually, using the above formulae, and then using the appropriate function from [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-tests) (hint, you're looking for a function to perform a _t_-test on two independent samples). For the manual approach, calculate the value of the test statistic and then its probability (the p-value). Verify you get the same results from both.

# __A:__ 

# In[51]:


from scipy import stats
stats.ttest_ind(insured_charges, uninsured_charges)


# In[52]:


insured = medical.query('insuranceclaim == 1')
insured_charges = insured['charges']
insured_mean = np.mean(insured_charges)
insured_std = np.std(insured_charges)

uninsured = medical.query('insuranceclaim == 0')
uninsured_charges = uninsured['charges']
uninsured_mean = np.mean(uninsured_charges)
uninsured_std = np.std(uninsured_charges)


# In[53]:


standard_dev = np.sqrt((((len(insured_charges)-1) *insured_std **2) + ((len(uninsured_charges)-1) * uninsured_std **2))/(len(insured_charges)+len(uninsured_charges)-2))
standard_dev


# In[54]:


t = (insured_mean - uninsured_mean)/(standard_dev*np.sqrt((1/len(insured_charges))+(1/len(uninsured_charges))))
t


# Congratulations! Hopefully you got the exact same numerical results. This shows that you correctly calculated the numbers by hand. Secondly, you used the correct function and saw that it's much easier to use. All you need to do pass your data to it.

# __Q:__ In the above calculations, we assumed the sample variances were equal. We may well suspect they are not (we'll explore this in another assignment). The calculation becomes a little more complicated to do by hand in this case, but we now know of a helpful function. Check the documentation for the function to tell it not to assume equal variances and perform the test again.

# __A:__

# In[55]:


stats.ttest_ind(insured_charges, uninsured_charges, equal_var = False)


# __Q:__ Conceptual question: look through the documentation for statistical test functions in scipy.stats. You'll see the above _t_-test for a sample, but can you see an equivalent one for performing a *z*-test from a sample? Comment on your answer.

# __A:__ I do not see any for conducting a z-test. This is likely because it is relatively simple to perform. 

# ## Learning outcomes

# Having completed this project notebook, you now have good hands-on experience:
# * using the central limit theorem to help you apply frequentist techniques to answer questions that pertain to very non-normally distributed data from the real world
# * performing inference using such data to answer business questions
# * forming a hypothesis and framing the null and alternative hypotheses
# * testing this using a _t_-test
