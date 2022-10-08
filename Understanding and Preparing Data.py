import pandas as pd
import seaborn as sns
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.neighbors import LocalOutlierFactor


diamonds = sns.load_dataset("diamonds")
pd.set_option("display.max.columns",None)
pd.set_option("display.width",500)

df1 = diamonds.copy()
def simple_control(dataframe):
    """
    A small function to understand the data

    Parameters
    ----------
    dataframe : DataFrame
        Type the name of your dataframe
    -------

    """
    print("----------shape--------")
    print(dataframe.shape)
    print("----------dtypes--------")
    print(dataframe.dtypes)
    print("----------Head--------")
    print(dataframe.head())
    print("----------describe--------")
    print(dataframe.describe())
    print("-----------isnull------------")
    print(dataframe.isnull().sum())
simple_control(df1)

def cat_control(dataFrame):
    """
    Shows the number of null values
    """
    cat_df = [i for i in dataFrame.columns if dataFrame[i].dtypes in ["category","bool","object"]]
    for col in cat_df:
        print(dataFrame[col].unique())
        print(dataFrame[col].value_counts())
        print(dataFrame[col].value_counts().count())
        print("---------------------------")
    return cat_df
cat_control(df1)

# We determined the degrees of ordinal variables
df1["color"] = df1["color"].astype(CategoricalDtype(categories=["J","I","H","G","F","E","D"],ordered=True))
df1["clarity"]=df1["clarity"].astype(CategoricalDtype(categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'],ordered=True))

def deep_cat_control(dataframe):
    """
    We use it to try to figure out if variables actually match their types
    """
    addcat = [z for z in dataframe.columns if dataframe[z].nunique() < 10 and dataframe[z].dtypes in ["int64", "float"]]
    addnum = [z for z in dataframe.columns if dataframe[z].nunique() > 15 and dataframe[z].dtypes in ["category", "object"]]
    categorical_var = cat_control(df1) + addcat
    categorical_var = [z for z in cat_control(df1) if z not in addnum]
    numerical_var = [z for z in dataframe.columns if dataframe[z].dtypes in ["int64", "float"]]
    numerical_var = [z for z in numerical_var if z not in addcat]

    print("Veriables: {}".format(dataframe.shape[1]))
    print("Categorical_Var: {}-{}".format(len(categorical_var),categorical_var))
    print("Numerical_Var: {}-{}".format(len(numerical_var),numerical_var))
    print("Added to numbers: {} \nAdded to categorical: {}".format(addnum,addcat))
    return categorical_var , numerical_var
categorical_var,numerical_var=deep_cat_control(df1)


def Nan(dataframe):
    """
    İf u want to quick check for Nan Values u can use this
    """
    msno.bar(dataframe)
    msno.matrix(dataframe)
    msno.heatmap(dataframe)
    plt.show(block = True)
Nan(df1)

# There are multiple methods to find outliers in a single variable,let's start

def outlier_standard_deviation(dataframe,col):
    upper = dataframe[col].mean() + 3*dataframe[col].std()
    lower =dataframe[col].mean() - 3*dataframe[col].std()
    return dataframe[(dataframe[col]< lower)|(dataframe[col]>upper)]
outlier_standard_deviation(df1,"price")

def outlier_Z_Score(dataframe,col):
    dataframe["Zscore"] = (dataframe[col] - dataframe[col].mean())/ dataframe[col].std()
    return df1[(df1.Zscore<-2.5) | (df1.Zscore>2.5)]
outlier_Z_Score(df1,"price")

def outlier_IQR(dataframe,col):
    Q1 = dataframe[col].quantile(0.25)
    Q3 = dataframe[col].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5 * IQR
    return dataframe[(dataframe[col] < lower) | (dataframe[col] > upper)]
outlier_IQR(df1,"price")

# Now let's find outliers in multivariables
def outlierLocal(dataframe):
    df = [i for i in df1.columns if df1[i].dtypes in ["float","int64"]]
    df = dataframe[df]
    cll = LocalOutlierFactor()
    pred = cll.fit_predict(df)
    dfscore = cll.negative_outlier_factor_
    anomaly = df[pred == -1]
    return anomaly
outlierLocal(df1)


def goal_cat(dataframe):
    for col in dataframe.columns:
        if dataframe[col].dtypes in ["category","bool","object"]:
            print(dataframe.groupby(col)["price"].agg(["mean","max","min"]))
            print("---------------------------------")
        else:
            pass
goal_cat(df1)


def vis_pie(dataframe):
    for col in dataframe.columns:
        if dataframe[col].dtypes in ["category","bool","object"]:
            yazı = dataframe[col].value_counts().index
            tura =dataframe[col].value_counts().values
            plt.figure(figsize=(8, 8))
            plt.pie(tura,labels=yazı,autopct='%1.1f%%')
            plt.show(block=True)
        else:
            pass
vis_pie(df1)

def vis(dataframe):
    for col in dataframe.columns:
        if dataframe[col].dtypes in ["category","bool","object"]:
            sns.countplot(x=col,data=dataframe)
            plt.show(block=True)

        else:
            plt.figure(figsize=(8,8))
            plt.hist(dataframe[col],bins=100,color="red",rwidth=1)
            plt.xlabel(col)
            plt.show(block=True)
vis(df1)

def corr(dataframe):
    corr_matrix = df1.corr()
    print(round(corr_matrix, 5))
    sns.heatmap(corr_matrix);
    plt.show(block=True)

corr(df1)


#OTHERS

plt.figure(figsize=(15,10))
sns.barplot(x="color", y="price",hue = "cut",data=df1)
plt.xlabel('Color/Cut')
plt.ylabel('Price')
plt.show(block=True)
plt.pause(5)



plt.figure(figsize=(15,10))
sns.barplot(x="cut", y="price",hue="clarity",data=df1)
plt.xlabel('Cut/Cla')
plt.ylabel('Price')
plt.show(block=True)
plt.pause(5)



