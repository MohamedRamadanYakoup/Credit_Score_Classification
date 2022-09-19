#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# - Over the years, the company has collected basic bank details and gathered a lot of credit-related information. The management wants to build an intelligent system to segregate the people into credit score brackets to reduce the manual efforts.

# # Data Description

# - Data has 2 Files Train Data and Test Data. Train data has 28 Columns and Test data has 27 Columns
# * Columns:-
#     * **ID**: Represents a unique identification of an entry
#     * **Customer ID**: Represents a unique identification of a person
#     * **Month**: Represents the month of the year
#     * **Name**: Represents the name of a person
#     * **Age**: Represents the age of the person
#     * **SSN**: Represents the social security number of a person
#     * **Occupation**: Represents the occupation of the person
#     * **Annual_Income**: Represents the annual income of the person
#     * **Monthly_Inhand_Salary**: Represents the monthly base salary of a person
#     * **Num_Bank_Accounts**: Represents the number of bank accounts a person holds
#     * **Num_Credit_Card**: Represents the number of other credit cards held by a person
#     * **Interest_Rate**: Represents the interest rate on credit card
#     * **Num_of_Loan**: Represents the number of loans taken from the bank
#     * **Type_of_Loan**: Represents the types of loan taken by a person
#     * **Delay_from_due_date**: Represents the average number of days delayed from the payment date
#     * **Num_of_Delayed_Payment**: Represents the average number of payments delayed by a person
#     * **Changed_Credit_Limit**: Represents the percentage change in credit card limit
#     * **Num_Credit_Inquiries**: Represents the number of credit card inquiries
#     * **Credit_Mix**: Represents the classification of the mix of credits
#     * **Outstanding_Debt**: Represents the remaining debt to be paid (in USD)
#     * **Credit_Utilization_Ratio**: Represents the utilization ratio of credit card
#     * **Credit_History_Age**: Represents the age of credit history of the person
#     * **Payment_of_Min_Amount**: Represents whether only the minimum amount was paid by the person
#     * **Total_EMI_per_month**: Represents the Equated Monthly Installments payments (in USD)
#     * **Amount_invested_monthly**: Represents the monthly amount invested by the customer (in USD)
#     * **Payment_Behaviour**: Represents the payment behavior of the customer (in USD)
#     * **Monthly_Balance**: Represents the monthly balance amount of the customer (in USD)
#     * **Credit_Score**: Represents the bracket of credit score (Poor, Standard, Good)

# # Importing Libraries

# In[253]:


import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


# In[254]:


df = pd.read_csv("train.csv", sep=",", encoding="utf-8")
test = pd.read_csv("test.csv", sep=",", encoding="utf-8")


# In[255]:


df.head()


# In[256]:


test.head()


# In[257]:


df.info()


# # Data Cleaning & Preprocessing

# In[258]:


imp = KNNImputer(n_neighbors=2)


def filling_na(df, column, type_=None):
    """
    This fucntion for filling null values to work with the data properly
    Parameters:
    df: DataFrame to fill the na with
    column: column which will fill the value in it
    type_: type of data needed be filled

    """
    np.random.seed(7)
    if type_ == "num":
        # filling_list = df[column].dropna()
        # df[column] = df[column].fillna(
        #   pd.Series(np.random.choice(filling_list, size=len(df.index)))
        # )
        df[column] = imp.fit_transform(df[column].values.reshape(-1, 1))

    else:
        filling_list = df[column].dropna().unique()
        df[column] = df[column].fillna(
            pd.Series(np.random.choice(filling_list, size=len(df.index)))
        )
    return df[column]


# In[259]:


df.describe().T


# In[260]:


df.describe(include="O").T


# In[261]:


df["Amount_invested_monthly"] = df["Amount_invested_monthly"].replace(
    "__10000__", 10000.00
)
df["Amount_invested_monthly"] = df["Amount_invested_monthly"].astype("float64")
df["Amount_invested_monthly"].dtype

test["Amount_invested_monthly"] = test["Amount_invested_monthly"].replace(
    "__10000__", 10000.00
)
test["Amount_invested_monthly"] = test["Amount_invested_monthly"].astype("float64")
test["Amount_invested_monthly"].dtype


# In[262]:


df["Monthly_Balance"] = df["Monthly_Balance"].replace(
    "__-333333333333333333333333333__", 0
)
df["Monthly_Balance"] = df["Monthly_Balance"].astype("float64")
df["Monthly_Balance"].dtype

test["Monthly_Balance"] = test["Monthly_Balance"].replace(
    "__-333333333333333333333333333__", 0
)
test["Monthly_Balance"] = test["Monthly_Balance"].astype("float64")
test["Monthly_Balance"].dtype


# In[263]:


df["Num_of_Delayed_Payment"] = df["Num_of_Delayed_Payment"].str.replace(
    r"_$", "", regex=True
)
df["Num_of_Delayed_Payment"] = df["Num_of_Delayed_Payment"].astype("float64")
df["Num_of_Delayed_Payment"].dtype

test["Num_of_Delayed_Payment"] = test["Num_of_Delayed_Payment"].str.replace(
    r"_$", "", regex=True
)
test["Num_of_Delayed_Payment"] = test["Num_of_Delayed_Payment"].astype("float64")
test["Num_of_Delayed_Payment"].dtype


# In[264]:


df["Annual_Income"] = df["Annual_Income"].str.replace(r"_$", "", regex=True)
df["Annual_Income"] = df["Annual_Income"].astype("float64")
df["Annual_Income"].dtype

test["Annual_Income"] = test["Annual_Income"].str.replace(r"_$", "", regex=True)
test["Annual_Income"] = test["Annual_Income"].astype("float64")
test["Annual_Income"].dtype


# In[265]:


df["Age"] = df["Age"].str.replace(r"_$", "", regex=True)
df["Age"] = df["Age"].astype("int64")
df["Age"].dtype

test["Age"] = test["Age"].str.replace(r"_$", "", regex=True)
test["Age"] = test["Age"].astype("int64")
test["Age"].dtype


# In[266]:


df["Outstanding_Debt"] = df["Outstanding_Debt"].str.replace(r"_$", "", regex=True)
df["Outstanding_Debt"] = df["Outstanding_Debt"].astype("float64")
df["Outstanding_Debt"].dtype

test["Outstanding_Debt"] = test["Outstanding_Debt"].str.replace(r"_$", "", regex=True)
test["Outstanding_Debt"] = test["Outstanding_Debt"].astype("float64")
test["Outstanding_Debt"].dtype


# In[267]:


df["Occupation"] = df["Occupation"].replace("_______", np.nan)

test["Occupation"] = test["Occupation"].replace("_______", np.nan)


# In[268]:


df["Credit_History_Age_#Year"] = df["Credit_History_Age"].str.split(" ", expand=True)[0]
df["Credit_History_Age_#Month"] = df["Credit_History_Age"].str.split(" ", expand=True)[
    3
]

test["Credit_History_Age_#Year"] = test["Credit_History_Age"].str.split(
    " ", expand=True
)[0]
test["Credit_History_Age_#Month"] = test["Credit_History_Age"].str.split(
    " ", expand=True
)[3]


# In[269]:


df["Payment_Behaviour"] = df["Payment_Behaviour"].replace(
    "!@9#%8", "Medium_spent_Medium_value_payments"
)

test["Payment_Behaviour"] = test["Payment_Behaviour"].replace(
    "!@9#%8", "Medium_spent_Medium_value_payments"
)


# In[270]:


df["Num_of_Loan"] = df["Num_of_Loan"].str.replace(r"_$", "", regex=True)
df["Num_of_Loan"] = df["Num_of_Loan"].astype("int64")
df["Num_of_Loan"].dtype


test["Num_of_Loan"] = test["Num_of_Loan"].str.replace(r"_$", "", regex=True)
test["Num_of_Loan"] = test["Num_of_Loan"].astype("int64")
test["Num_of_Loan"].dtype


# In[271]:


df["Credit_Mix"] = df["Credit_Mix"].replace("_", "Don't Have")

test["Credit_Mix"] = test["Credit_Mix"].replace("_", "Don't Have")


# In[272]:


df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"].replace("_", 0)
df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"].astype("float64")

test["Changed_Credit_Limit"] = test["Changed_Credit_Limit"].replace("_", 0)
test["Changed_Credit_Limit"] = test["Changed_Credit_Limit"].astype("float64")


# In[273]:


df["Interest_Rate"] = df["Interest_Rate"].astype("float64")
df["Interest_Rate"] = df["Interest_Rate"] / 100

test["Interest_Rate"] = test["Interest_Rate"].astype("float64")
test["Interest_Rate"] = test["Interest_Rate"] / 100


# In[274]:


df.Age.replace(-500, np.median(df.Age), inplace=True)
for i in df.Age.values:
    if i > 118:
        df.Age.replace(i, np.median(df.Age), inplace=True)

test.Age.replace(-500, np.median(test.Age), inplace=True)
for i in test.Age.values:
    if i > 118:
        test.Age.replace(i, np.median(test.Age), inplace=True)


# In[275]:


df.Num_of_Loan.replace(-100, np.median(df.Num_of_Loan), inplace=True)
for i in df.Num_of_Loan.values:
    if i > 10:
        df.Num_of_Loan.replace(i, np.median(df.Num_of_Loan), inplace=True)

test.Num_of_Loan.replace(-100, np.median(test.Num_of_Loan), inplace=True)
for i in test.Num_of_Loan.values:
    if i > 10:
        test.Num_of_Loan.replace(i, np.median(test.Num_of_Loan), inplace=True)


# In[276]:


for i in df.Interest_Rate:
    if i > 20:
        df.Interest_Rate.replace(i, np.median(df.Interest_Rate), inplace=True)

for i in test.Interest_Rate:
    if i > 20:
        test.Interest_Rate.replace(i, np.median(test.Interest_Rate), inplace=True)


# In[277]:


for i in df.Num_Bank_Accounts:
    if i > 100:
        df.Num_Bank_Accounts.replace(i, np.median(df.Num_Bank_Accounts), inplace=True)


for i in test.Num_Bank_Accounts:
    if i > 100:
        test.Num_Bank_Accounts.replace(
            i, np.median(test.Num_Bank_Accounts), inplace=True
        )


# In[278]:


for i in df.Num_Credit_Card:
    if i > 50:
        df.Num_Credit_Card.replace(i, np.median(df.Num_Credit_Card), inplace=True)


for i in test.Num_Credit_Card:
    if i > 50:
        test.Num_Credit_Card.replace(i, np.median(test.Num_Credit_Card), inplace=True)


# In[279]:


df["Monthly_Inhand_Salary"] = filling_na(df, "Monthly_Inhand_Salary", "num")
df["Num_Credit_Inquiries"] = filling_na(df, "Num_Credit_Inquiries", "num")
df["Amount_invested_monthly"] = filling_na(df, "Amount_invested_monthly", "num")
df["Num_of_Delayed_Payment"] = filling_na(df, "Num_of_Delayed_Payment", "num")
df["Monthly_Balance"] = filling_na(df, "Monthly_Balance", "num")
df["Credit_History_Age_#Year"] = filling_na(df, "Credit_History_Age_#Year", "num")
df["Credit_History_Age_#Month"] = filling_na(df, "Credit_History_Age_#Month", "num")
df["Type_of_Loan"] = filling_na(df, "Type_of_Loan")
df["Credit_History_Age"] = filling_na(df, "Credit_History_Age")
df["Occupation"] = filling_na(df, "Occupation")


test["Monthly_Inhand_Salary"] = filling_na(test, "Monthly_Inhand_Salary", "num")
test["Num_Credit_Inquiries"] = filling_na(test, "Num_Credit_Inquiries", "num")
test["Amount_invested_monthly"] = filling_na(test, "Amount_invested_monthly", "num")
test["Num_of_Delayed_Payment"] = filling_na(test, "Num_of_Delayed_Payment", "num")
test["Monthly_Balance"] = filling_na(test, "Monthly_Balance", "num")
test["Credit_History_Age_#Year"] = filling_na(test, "Credit_History_Age_#Year", "num")
test["Credit_History_Age_#Month"] = filling_na(test, "Credit_History_Age_#Month", "num")
test["Type_of_Loan"] = filling_na(test, "Type_of_Loan")
test["Credit_History_Age"] = filling_na(test, "Credit_History_Age")
test["Occupation"] = filling_na(test, "Occupation")


# In[280]:


df["Credit_History_Age_#Year"] = df["Credit_History_Age_#Year"].astype("int64")
df["Credit_History_Age_#Month"] = df["Credit_History_Age_#Month"].astype("int64")
df["Credit_History_Age_#Month"] = round(df["Credit_History_Age_#Month"] / 12, 2)
df["Credit_History_Age_In_Years"] = (
    df["Credit_History_Age_#Year"] + df["Credit_History_Age_#Month"]
)


test["Credit_History_Age_#Year"] = test["Credit_History_Age_#Year"].astype("int64")
test["Credit_History_Age_#Month"] = test["Credit_History_Age_#Month"].astype("int64")
test["Credit_History_Age_#Month"] = round(test["Credit_History_Age_#Month"] / 12, 2)
test["Credit_History_Age_In_Years"] = (
    test["Credit_History_Age_#Year"] + test["Credit_History_Age_#Month"]
)


# In[281]:


df.drop_duplicates(subset="ID", inplace=True)
df.drop(
    [
        "Name",
        "Credit_History_Age",
        "Credit_History_Age_#Year",
        "Credit_History_Age_#Month",
        "ID",
        "Customer_ID",
        "SSN",
    ],
    axis=1,
    inplace=True,
)


test.drop_duplicates(subset="ID", inplace=True)
test.drop(
    [
        "Name",
        "Credit_History_Age",
        "Credit_History_Age_#Year",
        "Credit_History_Age_#Month",
        "ID",
        "Customer_ID",
        "SSN",
    ],
    axis=1,
    inplace=True,
)


# In[282]:


df.Type_of_Loan = df.Type_of_Loan.str.replace("and", "")
df.Type_of_Loan = df.Type_of_Loan.str.replace(" ", "")

test.Type_of_Loan = test.Type_of_Loan.str.replace("and", "")
test.Type_of_Loan = test.Type_of_Loan.str.replace(" ", "")

cat_values = []
loan_cat = df.Type_of_Loan.unique()
for i in loan_cat:
    for j in i.split(","):
        cat_values.append(j)

loan_types = set([x.strip(" ") for x in set(cat_values)])
loan_types = list(loan_types)
loan_types


# In[283]:


df.describe().T


# In[284]:


df.describe(include="O").T


# # Exploratory Data Analysis

# In[285]:


plt.figure(figsize=(10, 7))
sns.countplot(data=df, x="Credit_Score")
plt.title("Customers Credit Scores", size=27, fontweight="bold")
plt.xlabel("Credit Score", size=27, fontweight="bold")
plt.ylabel("Count", size=27, fontweight="bold")
plt.show()


# **Comment:**
# <br>
# * Most people fill in the standard category

# In[286]:


plt.figure(figsize=(10, 7))
sns.lineplot(data=df, x="Occupation", y="Annual_Income", hue="Credit_Score")
plt.xticks(rotation=45)
plt.title("Annual Income Salary for Customers Occupation", size=27, fontweight="bold")
plt.xlabel("Occupation", size=27, fontweight="bold")
plt.ylabel("Annual Income", size=27, fontweight="bold")
plt.show()


# **Comment:**
# <br>
# * The Annual Income of the Cutomers doesn't affect on the credit score as we see that the variance on the annual income and the people can still have a good credit score whether the cutomer has a 100000 USD or 250000 USD Annually

# In[287]:


plt.figure(figsize=(10, 7))
sns.lineplot(data=df, x="Month", y="Monthly_Inhand_Salary", hue="Credit_Score")
plt.title("Annual Income Salary for Customers Occupation", size=27, fontweight="bold")
plt.xlabel("Month", size=27, fontweight="bold")
plt.ylabel("Monthly Inhand Salary", size=27, fontweight="bold")
plt.show()


# **Comment:**
# <br>
# * People who has a high inhand monthly salary have a good credit score and who has a low inhand salary has a low credit score

# In[288]:


plt.figure(figsize=(10, 7))
sns.lineplot(data=df, x="Occupation", y="Credit_Utilization_Ratio", hue="Credit_Score")
plt.xticks(rotation=45)
plt.title("Credit Card Usage Ratio According to Occupation", size=27, fontweight="bold")
plt.xlabel("Occupation", size=27, fontweight="bold")
plt.ylabel("Credit Card Utiliztion Ratio", size=27, fontweight="bold")
plt.show()


# **Comment:**
# <br>
# * More the People use the credit card it makes the credit score much better

# In[289]:


plt.figure(figsize=(10, 7))
sns.lineplot(
    data=df, x="Payment_Behaviour", y="Amount_invested_monthly", hue="Credit_Score"
)
plt.xticks(rotation=45)
plt.title(
    "Payment Behaviour of The Customer and The Amounts They Invest",
    size=27,
    fontweight="bold",
)
plt.xlabel("Payment Behaviour", size=27, fontweight="bold")
plt.ylabel("Amount Invested Monthly", size=27, fontweight="bold")
plt.show()


# **Comment:**
# <br>
# * Most People who invest between **700 to 800 USD** of their money have a good Credit Score and most people who have a standard credit score invest between **600 to 700 USD** per Month

# In[290]:


plt.figure(figsize=(10, 7))
sns.lineplot(data=df, x="Payment_Behaviour", y="Outstanding_Debt")
plt.xticks(rotation=45)
plt.title(
    "Payment Behaviour of The Customer and Their Debt", size=27, fontweight="bold"
)
plt.xlabel("Payment Behaviour", size=27, fontweight="bold")
plt.ylabel("Outstanding Debt", size=27, fontweight="bold")
plt.show()


# **Comment:**
# <br>
# * People who don't use the credit card so much but also pay small portion of the credit card has the majority on the outstanding debt **(Low_spent_Small_value_payments)** and the Category after that which has the 2nd most outstanding debt the people who **(Medium_spent_Medium_value_payments)**.
# <br>
# * The people who have the least outstanding debt are **Hight_spent_High_value_payments**.

# In[291]:


plt.figure(figsize=(10, 7))
sns.countplot(data=df, x="Credit_Mix", hue="Credit_Score")
# plt.xticks(rotation=45)
plt.title("Credit Mix", size=27, fontweight="bold")
plt.xlabel("Credit Mix Categories", size=27, fontweight="bold")
plt.ylabel("Count", size=27, fontweight="bold")
plt.show()


# **Comment:**
# <bar>
# * People who don't have a credit mix most of them has a Standard Credit score and the 2nd most category has a bad credit Score.
# <bar>
# * People who have a good credit mix most of them have a good credit score and the 2nd most category has a standard credit score.
# <bar>
# * People who have astandard mix most of them has a standard credit score and the 2nd most category have a bad credit score.
# <bar>
# * People who have a bad credit mix most of the has a bad credit score and the 2nd most category have a standard credit score.

# In[292]:


plt.figure(figsize=(10, 7))
sns.countplot(data=df, x="Payment_of_Min_Amount", hue="Credit_Score")
plt.title("Credit Score for Payment of Minimum Amounts", size=27, fontweight="bold")
plt.xlabel("Payment of Minimum Amounts", size=27, fontweight="bold")
plt.ylabel("Count", size=27, fontweight="bold")
plt.show()


# **Comment:**
# <br>
# * Customers who pay the minimum amounts has a poor credit score which but the people who don't pay the minimum amounts has a good credit score more than the others which mean that there are a lot of people who stay in debt for a long time as they don't pay the all amounts and they pay part of it which made an insterest on them.

# In[293]:


plt.figure(figsize=(10, 7))
sns.lineplot(
    data=df, x="Delay_from_due_date", y="Monthly_Inhand_Salary", hue="Credit_Score"
)
plt.title(
    "Delay of Payment According to Monthly Inhand Salary", size=27, fontweight="bold"
)
plt.xlabel("Delay from Due Date", size=27, fontweight="bold")
plt.ylabel("Monthly Inhand Salary", size=27, fontweight="bold")
plt.show()


# **Comment:**
# <br>
# * More the Customer has less Monthly inhand Salary more he where Delayed from Due Date but at the same time, There are peole who delayed from the due date but also have a good credit score.

# In[294]:


df["Age_Group"] = pd.cut(
    df.Age,
    bins=[14, 25, 30, 45, 55, 95, 120],
    labels=["14-25", "25-30", "30-45", "45-55", "55-95", "95-120"],
)
age_groups = (
    df.groupby(["Age_Group", "Credit_Score"])[
        "Outstanding_Debt", "Annual_Income", "Num_Bank_Accounts", "Num_Credit_Card"
    ]
    .sum()
    .reset_index()
)
age_groups


# In[295]:


g = sns.catplot(
    data=age_groups,
    x="Age_Group",
    y="Outstanding_Debt",
    height=7,
    aspect=1,
    col="Credit_Score",
    kind="bar",
    ci=None,
)
g.set_axis_labels("Age Group", "Outstanding Debt", size=27, fontweight="bold")
plt.show()


# **Comment:**
# <br>
# * Customers Between age of 30 and 45 the most category who have a lot of outstanding debts which mean that people in their youth age have a high purchase power and Cutomers between 45 to 55 their outstaning debt is less than young people.

# In[296]:


g = sns.catplot(
    data=age_groups,
    x="Age_Group",
    y="Annual_Income",
    height=7,
    aspect=1,
    col="Credit_Score",
    kind="bar",
    ci=None,
)
g.set_axis_labels("Age Group", "Annual Income", size=27, fontweight="bold")
plt.show()


# **Comment:**
# <br>
# * Customers between age 30 and 45 has the most Annual Income and the 2nd more group age are customers between 14 and 25 which mean not people from 25 and 30 which indicate that there are people who can make money in a young age more than the old people but as the same time as indication that the 2 largest Categories most of their credit score are Standard or Poor but the as for the people between 45 and 55 have more good credit score than the young people from 14 to 25

# In[297]:


g = sns.relplot(
    data=df,
    x="Num_Bank_Accounts",
    y="Num_Credit_Card",
    col="Credit_Score",
    height=7,
    aspect=1,
)
g.set_axis_labels(
    "Number of Bank Accounts", "Number of Credit Card", size=27, fontweight="bold"
)
plt.show()


# **Comment:**
# <br>
# * Most peopel have Accounts from 0 to 10 Accounts and the number of credit cards also from 0 to 10 which mean each account has at least one credit card

# # Prepare Data for Modeling

# In[298]:


df["AutoLoan"] = 0
df["Credit-BuilderLoan"] = 0
df["DebtConsolidationLoan"] = 0
df["HomeEquityLoan"] = 0
df["MortgageLoan"] = 0
df["NotSpecified"] = 0
df["PaydayLoan"] = 0
df["PersonalLoan"] = 0
df["StudentLoan"] = 0
index = 0
for i in df.Type_of_Loan:
    for j in i.split(","):
        df[j][index] = 1
    index += 1


# In[299]:


test["AutoLoan"] = 0
test["Credit-BuilderLoan"] = 0
test["DebtConsolidationLoan"] = 0
test["HomeEquityLoan"] = 0
test["MortgageLoan"] = 0
test["NotSpecified"] = 0
test["PaydayLoan"] = 0
test["PersonalLoan"] = 0
test["StudentLoan"] = 0
index = 0
for i in test.Type_of_Loan:
    for j in i.split(","):
        test[j][index] = 1
    index += 1


# In[300]:


df.head()


# In[301]:


test.head().T


# In[302]:


le = LabelEncoder()
df.Credit_Mix = le.fit_transform(df.Credit_Mix)
df.Credit_Mix.value_counts()

le = LabelEncoder()
test.Credit_Mix = le.fit_transform(test.Credit_Mix)
test.Credit_Mix.value_counts()


# In[303]:


le = LabelEncoder()
df.Payment_of_Min_Amount = le.fit_transform(df.Payment_of_Min_Amount)
df.Payment_of_Min_Amount.value_counts()

le = LabelEncoder()
test.Payment_of_Min_Amount = le.fit_transform(test.Payment_of_Min_Amount)
test.Payment_of_Min_Amount.value_counts()


# In[304]:


le = LabelEncoder()
df.Payment_Behaviour = le.fit_transform(df.Payment_Behaviour)
df.Payment_Behaviour.value_counts()

le = LabelEncoder()
test.Payment_Behaviour = le.fit_transform(test.Payment_Behaviour)
test.Payment_Behaviour.value_counts()


# In[305]:


le = LabelEncoder()
df.Credit_Score = le.fit_transform(df.Credit_Score)
df.Credit_Score.value_counts()


# In[318]:


# x = df.drop(
#    ["Month", "Age", "Occupation", "Type_of_Loan", "Credit_Score", "Age_Group"], axis=1
# ).values

x = df[
    [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Credit_Mix",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Payment_of_Min_Amount",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
        "Credit_History_Age_In_Years",
        "StudentLoan",
        "MortgageLoan",
        "PersonalLoan",
        "DebtConsolidationLoan",
        "Credit-BuilderLoan",
        "HomeEquityLoan",
        "NotSpecified",
        "AutoLoan",
        "PaydayLoan",
    ]
].values
y = df["Credit_Score"].values

# test_data = test.drop(["Month", "Age", "Occupation", "Type_of_Loan"], axis=1).values

test_data = test[
    [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Credit_Mix",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Payment_of_Min_Amount",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
        "Credit_History_Age_In_Years",
        "StudentLoan",
        "MortgageLoan",
        "PersonalLoan",
        "DebtConsolidationLoan",
        "Credit-BuilderLoan",
        "HomeEquityLoan",
        "NotSpecified",
        "AutoLoan",
        "PaydayLoan",
    ]
].values


# In[319]:


contam = 0.2
bootstrap = False
i_forest = IsolationForest(contamination=contam, bootstrap=bootstrap)
inlier = i_forest.fit_predict(x)


# In[320]:


mask = inlier != -1
x = x[mask, :]
y = y[mask]


# In[321]:


x.shape, y.shape


# In[322]:


test_data.shape


# # Modeling

# In[323]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=77)


# In[324]:


"""param_grid = {
    "classifier__max_depth": [3, 5, 7, 9, 11],
    "classifier__learning_rate": [0.1, 0.3, 0.01, 0.03, 0.001],
    "classifier__n_estimators": [100, 200, 300, 400],
    "classifier__subsample": [0.5, 0.6, 0.7, 0.8, 1],
}"""


# In[325]:


"""param = {
    "max_depth": [3, 5, 7, 9, 11],
    "learning_rate": [0.1, 0.4, 0.5],
    "n_estimators": [100, 200, 300, 400],
    "subsample": [0.5, 0.6, 0.7],
}"""


# In[327]:


pipe = Pipeline(
    [
        ["scaler", StandardScaler()],
        [
            "classifier",
            XGBClassifier(
                eval_metric="mlogloss",
                objective="multi:softmax",
                max_depth=9,
                learning_rate=0.3,
                n_estimators=1000,
                min_child_weight=9,
                reg_alpha=30,
                n_jobs=3,
                alpha=1,
            ),
        ],
    ]
)


# * max_depth=5, learning_rate=0.1, n_estimators=300, eta=0.01, subsample=0.7
# * RandomForestClassifier(n_estimators=700, criterion='gini', max_depth=10)
# * XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=700, eval_metric='mlogloss', min_child_weight=6, subsample=0.7, colsample_bytree=0.75, objective='multi:softmax', scale_pos_weight=1)
# * SVC(kernel="poly", c=0.5, gamma='auto')
# * pipe = Pipeline([
#     ["smote", SMOTE(random_state=77)],
#     ["scaler", StandardScaler()],
#     ["reducer", PCA()],
#     ["classifier", XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=700, eval_metric='mlogloss', min_child_weight=6, subsample=0.7, colsample_bytree=0.75, objective='multi:softmax', scale_pos_weight=1)]
# ])

# In[328]:


# scores = cross_val_score(pipe, xtrain, ytrain, cv=stratified_kfold)


# In[329]:


pipe.fit(xtrain, ytrain)


# In[330]:


pipe.score(xtrain, ytrain)


# In[331]:


pipe.score(xtest, ytest)


# In[376]:


ypred = pipe.predict(xtest)


# In[333]:


print(classification_report(ytest, ypred))


# In[379]:


print(confusion_matrix(ytest, ypred))
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(ytest, ypred), annot=True, vmin=0, vmax=7000)
plt.title("Confusion Matrix of XGBoost", size=27, fontweight="bold")
plt.savefig("Multiclass Conf_pipe", dpi=300)
plt.show()


# In[369]:


ypredpropa = pipe.predict_proba(xtest)
n_class = 3
fpr = {}
tpr = {}
thresh = {}

for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(ytest, ypredpropa[:, i], pos_label=i)

# plotting
plt.figure(figsize=(10, 7))
plt.plot(fpr[0], tpr[0], linestyle="--", color="orange", label="Class 0 vs Rest")
plt.plot(fpr[1], tpr[1], linestyle="--", color="green", label="Class 1 vs Rest")
plt.plot(fpr[2], tpr[2], linestyle="--", color="blue", label="Class 2 vs Rest")
plt.title("Multiclass ROC curve XGBoost", size=27, fontweight="bold")
plt.xlabel("False Positive Rate", size=27, fontweight="bold")
plt.ylabel("True Positive rate", size=27, fontweight="bold")
plt.legend(loc="best")
plt.savefig("Multiclass ROC_pipe", dpi=300)


# In[336]:


# knn = KNeighborsClassifier(n_neighbors=3)
pipe2 = Pipeline(
    [
        ["scaler", StandardScaler()],
        [
            "classifier",
            KNeighborsClassifier(n_neighbors=5),
        ],
    ]
)


# In[337]:


pipe2.fit(xtrain, ytrain)


# In[338]:


pipe2.score(xtrain, ytrain)


# In[339]:


pipe2.score(xtest, ytest)


# In[380]:


ypred = pipe2.predict(xtest)


# In[341]:


print(classification_report(ytest, ypred))


# In[342]:


print(confusion_matrix(ytest, ypred))


# In[381]:


print(confusion_matrix(ytest, ypred))
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(ytest, ypred), annot=True, vmin=0, vmax=6500)
plt.title("Confusion Matrix of KNN", size=27, fontweight="bold")
plt.savefig("Multiclass Conf_pipe2", dpi=300)
plt.show()


# In[370]:


ypredpropa = pipe2.predict_proba(xtest)
n_class = 3
fpr = {}
tpr = {}
thresh = {}

for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(ytest, ypredpropa[:, i], pos_label=i)

# plotting
plt.figure(figsize=(10, 7))
plt.plot(fpr[0], tpr[0], linestyle="--", color="orange", label="Class 0 vs Rest")
plt.plot(fpr[1], tpr[1], linestyle="--", color="green", label="Class 1 vs Rest")
plt.plot(fpr[2], tpr[2], linestyle="--", color="blue", label="Class 2 vs Rest")
plt.title("Multiclass ROC curve KNN", size=27, fontweight="bold")
plt.xlabel("False Positive Rate", size=27, fontweight="bold")
plt.ylabel("True Positive rate", size=27, fontweight="bold")
plt.legend(loc="best")
plt.savefig("Multiclass ROC_pipe2", dpi=300)


# In[345]:


pipe3 = Pipeline(
    [
        ["scaler", StandardScaler()],
        [
            "classifier",
            RandomForestClassifier(
                max_depth=9,
                max_leaf_nodes=7,
                n_jobs=3,
                n_estimators=1000,
                min_samples_split=3,
            ),
        ],
    ]
)


# In[346]:


pipe3.fit(xtrain, ytrain)


# In[347]:


pipe3.score(xtrain, ytrain)


# In[348]:


pipe3.score(xtest, ytest)


# In[382]:


ypred = pipe3.predict(xtest)


# In[350]:


print(classification_report(ytest, ypred))


# In[351]:


print(confusion_matrix(ytest, ypred))


# In[383]:


print(confusion_matrix(ytest, ypred))
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(ytest, ypred), annot=True, vmin=0, vmax=8000)
plt.title("Confusion Matrix of Random Forest", size=27, fontweight="bold")
plt.savefig("Multiclass Conf_pipe3", dpi=300)
plt.show()


# In[371]:


ypredpropa = pipe3.predict_proba(xtest)
n_class = 3
fpr = {}
tpr = {}
thresh = {}

for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(ytest, ypredpropa[:, i], pos_label=i)

# plotting
plt.figure(figsize=(10, 7))
plt.plot(fpr[0], tpr[0], linestyle="--", color="orange", label="Class 0 vs Rest")
plt.plot(fpr[1], tpr[1], linestyle="--", color="green", label="Class 1 vs Rest")
plt.plot(fpr[2], tpr[2], linestyle="--", color="blue", label="Class 2 vs Rest")
plt.title("Multiclass ROC curve Random Forest Classifier", size=27, fontweight="bold")
plt.xlabel("False Positive Rate", size=27, fontweight="bold")
plt.ylabel("True Positive rate", size=27, fontweight="bold")
plt.legend(loc="best")
plt.savefig("Multiclass ROC_pipe3", dpi=300)

