import pandas
import numpy as np

#Our case:
#We have a mental health at work survey. Now we need to produce some valuable input for our supervisors
#First we clean the data. Next we need to figure out what's interesting for us
#Without ML we can only do so much (no real predicition)
#Still we can attempt to provide input.
#Hypothesis: Employees in the USA with history of mental illness in the family are unlikely to
#discuss the issue with their employer.


_genders = {
    'm': 'm',
    'f':'f'
}

_bools = {
    'Yes': 1,
    'No': 0
}


#let's load and list the first five records of our data frame
survey_dirty = pandas.read_csv("survey.csv")

print(survey_dirty.head(5))

#and now print our columns
print(survey_dirty.columns)
print(survey_dirty.describe())



#All righty now it'd time to clean this data frame a little
#cleaning up columns
survey_dirty.columns = survey_dirty.columns.str.strip().str.lower()

####first of genders
#list all the genders
genders = survey_dirty.gender.unique()
print(genders)
#oh my!
survey_dirty.gender = survey_dirty.gender.str.lower().str.strip()
survey_dirty.gender = survey_dirty.gender.str.replace('^male|maile|make|man|mail|cis male|malr|msle$', 'm', case=False)
survey_dirty.gender = survey_dirty.gender.str.replace('^female|femail|woman$', 'f', case=False)
survey_dirty.gender = survey_dirty.gender.apply(lambda x: _genders.get(x, 'other'))
#lets check
print(survey_dirty.gender.unique())
print(survey_dirty.gender.value_counts())

#hmm, the printing is kinda funky
print(survey_dirty.head(5))

#a touch of panache
with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    print(survey_dirty.head(5))

#We need to check the country
print(survey_dirty.country.value_counts())
#We are only interested in USA
survey_dirty = survey_dirty.loc[lambda df: df.country == 'United States']
#In tech companies
survey_dirty = survey_dirty.loc[lambda df: df.tech_company == 'Yes']
#With family history
survey_dirty = survey_dirty.loc[lambda df: df.family_history == 'Yes']

survey_dirty = survey_dirty.loc[:, ['age', 'gender', 'treatment', 'work_interfere', 'no_employees', 'mental_health_interview']]

survey_dirty.reset_index(inplace=True, drop=True)

with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    print(survey_dirty.head(5))

#at this point we have 265 cases which we can start working with
#let's check a few things
#1. count by age
#2. avg age
#3. median age
#4. The percentage of men with difficulties who do not treat at all


#Ok, so now let's try to visualize something
# How about percent of people with problems by company size
# And another one: Number of people willing to talk by company size