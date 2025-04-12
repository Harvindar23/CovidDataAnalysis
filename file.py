import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_original=pd.read_csv('Provisional_COVID-19_Deaths_by_Sex_and_Age.csv')

# creating a copy of original dataset
df=df_original.copy()
print("Shape of the dataSet" ,df.shape)
print("\n\n ----------------------------------- \n\n")
print(df.info())


print("To know number of missing values in each  column\n", df.isnull().sum(), end='\n\n')


#to make column names as python naming convention so , it become  more convinent to work
df.columns = df.columns.str.lower().str.replace(' ', '_')
print(df.info(),end='\n\n') # to see the names of columns


#starting with data cleaning
print(df['group'].value_counts(),end='\n\n')
# df.shape

df_numeric=df.select_dtypes(include=np.number)
l=df_numeric.columns[2:]
print(l)
print("To prove the ambiguity/repetation of the same data: -> ")
print( df.groupby('group')[l].sum() ,end='\n\n')


#removing the duplicacy/ repeated data
df=df[df['group'] == 'By Month']
print(df.isnull().sum() )
print('\n\n')
#by performing the above step the missing values in the 'year' and 'month' are also removed

#handling the missing values in rest of columns
rows=df[~df['footnote'].isnull()].index
df.loc[rows, l]=df.loc[rows,l].fillna(5)
print(df.isnull().sum()) #none of the column has missing values other than footnote

#the missing values of footnote column is not handled , because it will disturb the originality of the data

#changing the datatype of each column for saving memory usage
category=df.select_dtypes(exclude=np.number).columns
print(category,end='\n\n')
for i in category[0:3]:
    df[i] = pd.to_datetime(df[i], format="%m/%d/%Y", errors='coerce')
df['month']=df['start_date'].dt.month_name()
category=list(category[3:])
category.extend(['year','month'])
df[category]=df[category].astype('category')
print(df.info(),end='\n\n')

#univariate analysis on numerical columns
def univariate_numerical(df, x):
  print(df[x].describe())
  print(f'Skewness in {x} ',df[x].skew())
  print('\n\n')
  df[f'log_{x}']=np.log(df[x]+1)
  fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 1 row, 2 columns
  sns.kdeplot(data=df[x], ax=axes[0])
  sns.kdeplot(data=df[f'log_{x}'], ax=axes[1])
  plt.show()
  print('\n\n')
  print(f'\n skewness  before and after transformation in the column {x}',df[x].skew(), "  ", df[f'log_{x}'].skew(), end='\n\n')
  q1=df[x].quantile(0.25)
  q3=df[x].quantile(0.75)
  iqr=q3-q1
  upper_value= q3+1.5*iqr
  lower_value= q1-1.5*iqr
  print(f'outliers count for {x}: ' ,df[(df[x]>upper_value )| (df[x]<lower_value)].shape)
  df.drop(columns=[f'log_{x}'], inplace=True)


print("Univariate On Numerical columns")
l=df.select_dtypes(include=np.number)
for i in l:
  print(i,end='\n\n\n')
  univariate_numerical(df, i)
  print('----------------------------------------------------------------------------\n\n')



#univariate analysis on categorical columns
def univariate_categorical(df, x):
  print(df[x].value_counts(), '\n\n')
  plt.figure(figsize=(15,6))
  sns.countplot(data=df, x=x)
  plt.title(f'Count Plot of {x}')
  if(i=='state' or i=='age_group' or x=='month'):
    plt.xticks(rotation='vertical')
  # plt.xlabel(i, rotation='vertical')
  plt.show()


print("Univariate On Categorical columns")
for i in category:
  print(i,end='\n\n\n')
  univariate_categorical(df, i)
  print('\n\n')


#multivariate analysis
df_numeric=df[['covid-19_deaths','pneumonia_deaths', 'influenza_deaths']]
corr=df_numeric.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Between COVID-19 and Respiratory Deaths")
plt.show()


sns.pairplot(df[['covid-19_deaths', 'pneumonia_deaths', 'influenza_deaths']], diag_kind='kde')
plt.show()


#objective 1->          COVID-19 Deaths by Age Group & Sex
df_agg=df.groupby(['age_group', 'sex'])['covid-19_deaths'].sum()
df_agg=df_agg.reset_index()
print(df_agg)



g=sns.catplot(kind='bar',data=df_agg, col='sex', y='covid-19_deaths', x='age_group',col_wrap=3)
for ax in g.axes.flat:
    ax.set_yscale('log')

for ax in g.axes.flat:
    ax.tick_params(axis='x', labelrotation=80)
plt.show()


#objective 2->    Sex-based Mortality Analysis

df_sex = df.groupby('sex')['total_deaths'].sum().reset_index()
print(df_sex)
sns.barplot(data=df_sex, x='sex', y='total_deaths')
for i, row in df_sex.iterrows():
    plt.text(
        x=i,
        y=row['total_deaths'],
        s=f"{int(row['total_deaths']):,}",  # Correct comma format
        ha='center',
        va='bottom',
        fontsize=9
    )
plt.title('Sex Based morality analysis')

plt.show()

#objective 3->    Comparison of COVID-19, Pneumonia, and Influenza Death

df_agg=df[['covid-19_deaths','pneumonia_deaths','influenza_deaths']].sum()
# print(df_agg)
df_agg2=df.groupby('year')
df_agg2=df_agg2[['covid-19_deaths','pneumonia_deaths','influenza_deaths']].sum()
df_agg2.reset_index(inplace=True)
# print(df_agg2)
df_melted = df_agg2.melt(id_vars='year',
                         value_vars=['covid-19_deaths', 'pneumonia_deaths', 'influenza_deaths'],
                         var_name='Cause',
                         value_name='Deaths')
# print(df_melted)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#comparing for over all  3 years
axes[0].pie(df_agg.values, labels=df_agg.index, autopct='%1.1f%%')
axes[0].set_title('cause wise Overall Death Distribution')

#year wise comparision
sns.barplot(data=df_melted, x='year', y='Deaths', hue='Cause', ax=axes[1])
axes[1].set_title('Year-wise Cause of Death')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Number of Deaths')

plt.tight_layout()
plt.show()


#objective 4->   Identify the Month with Peak COVID-19 Mortality for Each State

df_ym=df[['year','month','state','covid-19_deaths']]
df_ym=df_ym.groupby(['state','year','month'])
df_ym=df_ym['covid-19_deaths'].sum().reset_index()

df_ym=df_ym.sort_values(by='covid-19_deaths',ascending=False)
df_ym=df_ym.drop_duplicates('state')
df_ym['month-year']=df_ym['month'].astype('str') + ' - ' + df_ym['year'].astype('str')
print(df_ym)

plt.figure(figsize=(15, 14))
sns.barplot(data=df_ym, y='state', x='covid-19_deaths',hue='month-year')

plt.title('Peak COVID-19 Mortality Month for Each State')
plt.xlabel('COVID-19 Deaths')
plt.ylabel('State')
plt.tight_layout()
plt.show()


#objective 5->   COVID Deaths as % of Total Deaths
val=( df['covid-19_deaths'].sum() / df['total_deaths'].sum() )*100
print("COVID Deaths as % of Total Deaths",val, '%')

# year wise COVID Deaths as % of Total Deaths 
df_year=df.groupby('year')
df_year_sum=df_year[['covid-19_deaths','total_deaths']].sum()
df_year_sum['covid%']=df_year_sum['covid-19_deaths']/df_year_sum['total_deaths']*100
print('df_year_sum')
plt.figure(figsize=(10,5))
sns.barplot(data=df_year_sum, x='year', y='covid%')
plt.title('COVID-19 Deaths as % of Total Deaths by Year')
plt.ylabel('Percentage (%)')
plt.xlabel('year')
plt.show()

#objective 6->  Monthly COVID-19 Death Trends
df_ym=df.groupby(['year','month'])
df_ym=df_ym['covid-19_deaths'].sum().reset_index()
df_ym['month_year'] = df_ym['month'].str[:3] + ' ' + df_ym['year'].astype(str)
print(df_ym)
plt.figure(figsize=(14,6))
plt.plot(df_ym['month_year'], df_ym['covid-19_deaths'], marker='o')

plt.xticks(rotation=45, ha='right')
plt.title('Monthly COVID-19 Death Trends')
plt.xlabel('Month-Year')
plt.ylabel('COVID-19 Deaths')
plt.tight_layout()
plt.show()
