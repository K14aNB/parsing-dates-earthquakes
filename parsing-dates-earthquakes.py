# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] id="OtLtvKtc3jJI"
# # **Significant Earthquakes, 1965-2016**
#
# Date, time, and location of all earthquakes with magnitude of 5.5 or higher

# + [markdown] id="EOD4edXg4SU6"
# ## **About the Dataset**
#
# This dataset includes a record of the date, time, location, depth, magnitude, and source of every earthquake with a reported magnitude 5.5 or higher since 1965.

# + colab={"base_uri": "https://localhost:8080/"} id="MHfbGrvIzTnB" executionInfo={"status": "ok", "timestamp": 1708967634074, "user_tz": -330, "elapsed": 3290, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}} outputId="dcfc402a-6806-4749-8194-e4190bab2563"
from google.colab import drive, userdata
drive.mount('/content/drive')

# + colab={"base_uri": "https://localhost:8080/"} id="hNqXd6QtI5Aj" executionInfo={"status": "ok", "timestamp": 1708967634075, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}} outputId="1de66df3-9d71-46c6-91ac-bde4a15590c6"
# %cd /content/drive/MyDrive/Colab Notebooks/Parsing Dates Kaggle Earthquakes/parsing-dates-earthquakes

# + colab={"base_uri": "https://localhost:8080/"} id="AvNClLgxJyxK" executionInfo={"status": "ok", "timestamp": 1708967637055, "user_tz": -330, "elapsed": 2984, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}} outputId="55a3b39e-3d75-4ecf-f69c-c0f5940dad51"
# !jupyter nbconvert --to html parsing-dates-earthquakes.ipynb

# + colab={"base_uri": "https://localhost:8080/"} id="mf5BAoeKRh8J" executionInfo={"status": "ok", "timestamp": 1708967645389, "user_tz": -330, "elapsed": 8344, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}} outputId="12196dcf-4c4a-446b-e22f-e0651abfc4cd"
# !pip install jupytext

# + colab={"base_uri": "https://localhost:8080/"} id="7TtpunZoQHjJ" executionInfo={"status": "ok", "timestamp": 1708967646964, "user_tz": -330, "elapsed": 1584, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}} outputId="69cc6904-0a41-4088-c072-1c22ccba1314"
# !jupytext --set-formats ipynb,py parsing-dates-earthquakes.ipynb
# !jupytext --sync parsing-dates-earthquakes.ipynb

# + [markdown] id="Tg0AkV05Ag97"
# ## Import the libraries

# + id="k1nkxxdx5szD" executionInfo={"status": "ok", "timestamp": 1708967646964, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

# + id="uCbTWRgt6hoC" executionInfo={"status": "ok", "timestamp": 1708967646964, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}}
# Setting the environment variables
# os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
# os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')

# + id="LhceXfwv7a8K" executionInfo={"status": "ok", "timestamp": 1708967646964, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}}
# Download dataset from Kaggle
# # !mkdir data
# # !pip install kaggle
# # !kaggle datasets download usgs/earthquake-database
# # !unzip earthquake-database.zip -d data/
# # !rm earthquake-database.zip

# + id="Heg5hACcAg97" executionInfo={"status": "ok", "timestamp": 1708967647825, "user_tz": -330, "elapsed": 863, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}}
# read the data
earthquakes = pd.read_csv("./data/database.csv")

# set seed for reproducibility
np.random.seed(0)

# + [markdown] id="YiS0d5B4Ag98"
# ## Check the data type of our date column
#

# + colab={"base_uri": "https://localhost:8080/"} id="mP_uMQp0Ag98" executionInfo={"status": "ok", "timestamp": 1708967647826, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}} outputId="9bf1010c-00e7-4799-b812-44200df07baa"
earthquakes['Date'].dtype

# + [markdown] id="fXVS2xMwAg99"
# ## Convert our date columns to datetime
#
# Some entries follow a different datetime format compared to rest.

# + colab={"base_uri": "https://localhost:8080/", "height": 307} id="52EsWCNiAg99" executionInfo={"status": "ok", "timestamp": 1708967647826, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}} outputId="838ea19d-8718-4376-ae67-105586b157a2"
earthquakes[3378:3383]

# + [markdown] id="-ypkleDsAg99"
# This does appear to be an issue with data entry: ideally, all entries in the column have the same format.  We can get an idea of how widespread this issue is by checking the length of each entry in the "Date" column.

# + colab={"base_uri": "https://localhost:8080/"} id="EWUfoe3_Ag99" executionInfo={"status": "ok", "timestamp": 1708967647826, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}} outputId="232a7f17-a8c1-4eb9-8f9d-6eb3b44c2a58"
date_lengths = earthquakes.Date.str.len()
date_lengths.value_counts()

# + [markdown] id="kgEjtsjIAg9-"
# Looks like there are two more rows that has a date in a different format.

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="QKsq20j4Ag9-" executionInfo={"status": "ok", "timestamp": 1708967647826, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}} outputId="e6354fd1-b169-4b99-eba5-5e53d5174c93"
indices = np.where([date_lengths == 24])[1]
print('Indices with corrupted data:', indices)
earthquakes.loc[indices]

# + colab={"base_uri": "https://localhost:8080/"} id="LjkItKJJAg9-" executionInfo={"status": "ok", "timestamp": 1708967647826, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}} outputId="832c1a16-78b5-4cf6-a1a3-3842c78ebd37"
from datetime import date,datetime
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype,is_object_dtype


def parse_dates(date_obj_col: pd.core.series, dateonly: bool, date_format: str, sep='/'):
    '''
    parses pandas object into uniform string format and converts into pandas datetime64 format

    Arguments:
    date_obj_col: pandas series - column of dtype object which need to be converted to datetime64.
    dateonly: bool - if True, then pandas datetime64 object will be further parsed to return only date.
                     if False, then time part will not be excluded in the final result.
    date_format: str - can be '%m%d%y' or '%d%m%y' or '%y%m%d or '%m%d%Y' or '%d%m%Y' or '%Y%m%d'
                  where 'Y' represents 4 digit year and 'y' represents 2 digit year.
    sep: str - default value= '/'. Can be '-', '.'


    Returns:
    Only date component of pandas datetime64 if dateonly=True in the given format.
    pandas series of dtype datetime64 if dateonly=False in the given format.
    '''
    date_col_copy = date_obj_col.copy()
    if type(date_col_copy)==pd.core.series.Series and is_object_dtype(date_col_copy):
        if dateonly is True:
            if((date_col_copy.str.len()>10).any()):
                indices = date_col_copy.str.len()>10
                if date_col_copy[indices].size > 1:
                  date_col_copy[indices] = date_col_copy[indices].str.replace(r'[T\s][0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{3}Z', '', regex=True)
                  date_col_copy[indices] = date_col_copy[indices].str.replace(r'[T\s][0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{3}', '', regex=True)
                  date_col_copy[indices] = date_col_copy[indices].str.replace(r'[T\s][0-9]{2}:[0-9]{2}:[0-9]{2}', '', regex=True)
                  date_col_copy[indices] = date_col_copy[indices].str.replace(r'[T\s][0-9]{2}[0-9]{2}[0-9]{2}', '', regex=True)
                  date_col_copy[indices] = date_col_copy[indices].str.replace(r'[T\s][0-9]{2}[0-9]{2}[0-9]{2}.[0-9]{3}Z', '', regex=True)
                  date_col_copy[indices] = date_col_copy[indices].str.replace(r'[T\s][0-9]{2}[0-9]{2}[0-9]{2}.[0-9]{3}', '', regex=True)

                  date_col_copy[indices] = date_col_copy[indices].str.replace(r'[T\s][0-9]{2}:[0-9]{2}:[0-9]{2} [AaPp][Mm]', '')
                  date_col_copy[indices] = date_col_copy[indices].str.replace(r'[T\s][0-9]{2}:[0-9]{2} [AaPp][Mm]', '')
                elif date_col_copy[indices].size == 1:
                  date_col_copy[indices] = date_col_copy[indices].replace(r'[T\s][0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{3}Z', '', regex=True)
                  date_col_copy[indices] = date_col_copy[indices].replace(r'[T\s][0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{3}', '', regex=True)
                  date_col_copy[indices] = date_col_copy[indices].replace(r'[T\s][0-9]{2}:[0-9]{2}:[0-9]{2}', '', regex=True)
                  date_col_copy[indices] = date_col_copy[indices].replace(r'[T\s][0-9]{2}[0-9]{2}[0-9]{2}', '', regex=True)
                  date_col_copy[indices] = date_col_copy[indices].replace(r'[T\s][0-9]{2}[0-9]{2}[0-9]{2}.[0-9]{3}Z', '', regex=True)
                  date_col_copy[indices] = date_col_copy[indices].replace(r'[T\s][0-9]{2}[0-9]{2}[0-9]{2}.[0-9]{3}', '', regex=True)

                  date_col_copy[indices] = date_col_copy[indices].replace(r'[T\s][0-9]{2}:[0-9]{2}:[0-9]{2} [AaPp][Mm]', '')
                  date_col_copy[indices] = date_col_copy[indices].replace(r'[T\s][0-9]{2}:[0-9]{2} [AaPp][Mm]', '')



        if (date_col_copy.str.find('/')!=-1).any() or (date_col_copy.str.find('-')!=-1).any() or (date_col_copy.str.find('.')!=-1).any():
            if sep == '/':
                if (date_col_copy.str.find('-')!=-1).any():
                    date_col_copy = date_col_copy.str.replace('-', '/')
                if (date_col_copy.str.find('.')!=-1).any():
                    date_col_copy = date_col_copy.str.replace('.', '/')
            elif sep == '-':
                if (date_col_copy.str.find('/')!=-1).any():
                    date_col_copy = date_col_copy.str.replace('/', '-')
                if (date_col_copy.str.find('.')!=-1).any():
                    date_col_copy = date_col_copy.str.replace('.', '-')
            elif sep == '.':
                if (date_col_copy.str.find('/')!=-1).any():
                    date_col_copy = date_col_copy.str.replace('/', '.')
                if (date_col_copy.str.find('-')!=-1).any():
                    date_col_copy = date_col_copy.str.replace('-', '.')
        else:
            # To handle the case of no separator in dates like DDMMYY or DDMMYYYY or MMDDYY or MMDDYYYY
            # TO DO
            pass


       # To try converting dtype to pandas datetime64 object using pd.to_datetime
        try:
            if(is_datetime64_any_dtype(date_col_copy)):
                return date_col_copy
            else:
                date_col_copy_final = pd.to_datetime(date_col_copy, format=date_format[:2]+sep+date_format[2:4]+sep+date_format[4:6],errors='coerce')
                if(date_col_copy_final.isna().any()):
                    nat_condition = date_col_copy_final.isna()
                    nat_indices = date_col_copy_final[nat_condition].index.tolist()
                    date_col_nat = date_col_copy[nat_indices]
                    split_date = []
                    for key,val in dict(date_col_nat).items():
                      if sep in ['', None] or sep not in val:
                            raise ValueError(f'{val} at index {key} is not in any of the specified datetime format')
                      else:
                            # To try to rearrange the NaT dates as per date_format
                            split_date = val.split(sep)
                            if date_format == '%m%d%y' and sep not in ['', None]:
                                if ((len(split_date[0])==2) and int(split_date[0]) in range(0,int(str(date.today().year)[2:])+1)) or \
                              (len(split_date[0])==4) and int(split_date[0]) in range(1945, int(str(date.today().year))+1):
                                    if(len(split_date[1])==2 and int(split_date[1]) in range(1,13)):
                                        if(len(split_date[2])==2 and int(split_date[2])in range(1,32)):
                                            date_col_copy_final[nat_indices]=split_date[1]+sep+split_date[2]+sep+(split_date[0] if len(split_date[0])==2 else split_date[0][2:])
                                elif (len(split_date[0]==2) and int(split_date[0]) in range(1,13)):
                                    if(len(split_date[1])==2 and int(split_date[1]) in range(1,32)):
                                        if(len(split_date[2])==2 and int(split_date[2]) in range(0,int(str(date.today().year)[2:])+1)) or \
                                     (len(split_date[2])==4) and int(split_date[0]) in range(1945, int(str(date.today().year))+1):
                                            date_col_copy_final[nat_indices]=split_date[0]+sep+split_date[1]+sep+(split_date[2] if len(split_date[2])==2 else split_date[2][2:])
                                elif (len(split_date[0])==2) and int(split_date[0]) in range(1,32):
                                    if (len(split_date[1])==2) and split_date[1] in range(1,13):
                                        if ((len(split_date[2])==2) and int(split_date[2]) in range(0, int(str(date.today().year)[2:])+1)) or \
                                     (len(split_date[2])==4) and int(split_date[2]) in range(1945, int(str(date.today().year))+1):
                                            date_col_copy_final[nat_indices] = split_date[1]+sep+split_date[0]+sep+(split_date[2] if len(split_date[2])==2 else split_date[2][2:])
                            elif date_format == '%m%d%Y' and sep not in ['', None]:
                                if ((len(split_date[0])==2) and int(split_date[0]) in range(0,int(str(date.today().year)[2:])+1)) or \
                              (len(split_date[0])==4) and int(split_date[0]) in range(1945, int(str(date.today().year))+1):
                                    if(len(split_date[1])==2 and int(split_date[1]) in range(1,13)):
                                        if(len(split_date[2])==2 and int(split_date[2])in range(1,32)):
                                            date_col_copy_final[nat_indices]=split_date[1]+sep+split_date[2]+sep+ \
                                         (split_date[0] if len(split_date[0])==4 else ('19' if int(split_date[0])>45 else '20')+split_date[0])
                                elif (len(split_date[0]==2) and int(split_date[0]) in range(1,13)):
                                    if(len(split_date[1])==2 and int(split_date[1]) in range(1,32)):
                                        if(len(split_date[2])==2 and int(split_date[2]) in range(0,int(str(date.today().year)[2:])+1)) or \
                                     (len(split_date[2])==4) and int(split_date[0]) in range(1945, int(str(date.today().year))+1):
                                            date_col_copy_final[nat_indices]=split_date[0]+sep+split_date[1]+sep+ \
                                         (split_date[2] if len(split_date[2])==4 else ('19' if int(split_date[2])>45 else '20')+split_date[2])
                                elif (len(split_date[0])==2) and int(split_date[0]) in range(1,32):
                                    if (len(split_date[1])==2) and split_date[1] in range(1,13):
                                        if ((len(split_date[2])==2) and int(split_date[2]) in range(0, int(str(date.today().year)[2:])+1)) or \
                                     (len(split_date[2])==4) and int(split_date[2]) in range(1945, int(str(date.today().year))+1):
                                            date_col_copy_final[nat_indices] = split_date[1]+sep+split_date[0]+sep+ \
                                         (split_date[2] if len(split_date[2])==4 else ('19' if int(split_date[2])>45 else '20')+split_date[2])
                            elif date_format == '%d%m%y' and sep not in ['', None]:
                                if ((len(split_date[0])==2) and int(split_date[0]) in range(0,int(str(date.today().year)[2:])+1)) or \
                             (len(split_date[0])==4) and int(split_date[0]) in range(1945, int(str(date.today().year))+1):
                                    if(len(split_date[1])==2 and int(split_date[1]) in range(1,13)):
                                        if(len(split_date[2])==2 and int(split_date[2])in range(1,32)):
                                            date_col_copy_final[nat_indices]=split_date[2]+sep+split_date[1]+sep+(split_date[0] if len(split_date[0])==2 else split_date[0][2:])
                                elif (len(split_date[0]==2) and int(split_date[0]) in range(1,13)):
                                    if(len(split_date[1])==2 and int(split_date[1]) in range(1,32)):
                                        if(len(split_date[2])==2 and int(split_date[2]) in range(0,int(str(date.today().year)[2:])+1)) or \
                                     (len(split_date[2])==4) and int(split_date[0]) in range(1945, int(str(date.today().year))+1):
                                            date_col_copy_final[nat_indices]=split_date[1]+sep+split_date[0]+sep+(split_date[2] if len(split_date[2])==2 else split_date[2][2:])
                                elif (len(split_date[0])==2) and int(split_date[0]) in range(1,32):
                                    if (len(split_date[1])==2) and split_date[1] in range(1,13):
                                        if ((len(split_date[2])==2) and int(split_date[2]) in range(0, int(str(date.today().year)[2:])+1)) or \
                                     (len(split_date[2])==4) and int(split_date[2]) in range(1945, int(str(date.today().year))+1):
                                            date_col_copy_final[nat_indices] = split_date[0]+sep+split_date[1]+sep+(split_date[2] if len(split_date[2])==2 else split_date[2][2:])
                            elif date_format == '%d%m%Y' and sep not in ['', None]:
                                if ((len(split_date[0])==2) and int(split_date[0]) in range(0,int(str(date.today().year)[2:])+1)) or \
                             (len(split_date[0])==4) and int(split_date[0]) in range(1945, int(str(date.today().year))+1):
                                    if(len(split_date[1])==2 and int(split_date[1]) in range(1,13)):
                                        if(len(split_date[2])==2 and int(split_date[2])in range(1,32)):
                                            date_col_copy_final[nat_indices]=split_date[2]+sep+split_date[1]+sep+ \
                                         (split_date[0] if len(split_date[0])==4 else ('19' if int(split_date[0])>45 else '20')+split_date[0])
                                elif (len(split_date[0]==2) and int(split_date[0]) in range(1,13)):
                                    if(len(split_date[1])==2 and int(split_date[1]) in range(1,32)):
                                        if(len(split_date[2])==2 and int(split_date[2]) in range(0,int(str(date.today().year)[2:])+1)) or \
                                     (len(split_date[2])==4) and int(split_date[0]) in range(1945, int(str(date.today().year))+1):
                                            date_col_copy_final[nat_indices]=split_date[1]+sep+split_date[0]+sep+ \
                                         (split_date[2] if len(split_date[2])==4 else ('19' if int(split_date[2])>45 else '20')+split_date[2])
                                elif (len(split_date[0])==2) and int(split_date[0]) in range(1,32):
                                    if (len(split_date[1])==2) and split_date[1] in range(1,13):
                                        if ((len(split_date[2])==2) and int(split_date[2]) in range(0, int(str(date.today().year)[2:])+1)) or \
                                     (len(split_date[2])==4) and int(split_date[2]) in range(1945, int(str(date.today().year))+1):
                                            date_col_copy_final[nat_indices] = split_date[1]+sep+split_date[0]+sep+ \
                                         (split_date[2] if len(split_date[2])==4 else ('19' if int(split_date[2])>45 else '20')+split_date[2])
                            elif date_format == '%y%m%d' and sep not in ['', None]:
                                if ((len(split_date[0])==2) and int(split_date[0]) in range(0,int(str(date.today().year)[2:])+1)) or \
                             (len(split_date[0])==4) and int(split_date[0]) in range(1945, int(str(date.today().year))+1):
                                    if(len(split_date[1])==2 and int(split_date[1]) in range(1,13)):
                                        if(len(split_date[2])==2 and int(split_date[2])in range(1,32)):
                                            date_col_copy_final[nat_indices]=(split_date[0] if len(split_date[0])==2 else split_date[0][2:])+sep+split_date[1]+sep+split_date[2]
                                elif (len(split_date[0]==2) and int(split_date[0]) in range(1,13)):
                                    if(len(split_date[1])==2 and int(split_date[1]) in range(1,32)):
                                        if(len(split_date[2])==2 and int(split_date[2]) in range(0,int(str(date.today().year)[2:])+1)) or \
                                     (len(split_date[2])==4) and int(split_date[0]) in range(1945, int(str(date.today().year))+1):
                                            date_col_copy_final[nat_indices]=(split_date[2] if len(split_date[2])==2 else split_date[2][2:])+sep+split_date[0]+sep+split_date[1]
                                elif (len(split_date[0])==2) and int(split_date[0]) in range(1,32):
                                    if (len(split_date[1])==2) and split_date[1] in range(1,13):
                                        if ((len(split_date[2])==2) and int(split_date[2]) in range(0, int(str(date.today().year)[2:])+1)) or \
                                     (len(split_date[2])==4) and int(split_date[2]) in range(1945, int(str(date.today().year))+1):
                                            date_col_copy_final[nat_indices] = (split_date[2] if len(split_date[2])==2 else split_date[2][2:])+sep+split_date[1]+sep+split_date[0]
                            elif date_format == '%Y%m%d' and sep not in ['', None]:
                                if ((len(split_date[0])==2) and int(split_date[0]) in range(0,int(str(date.today().year)[2:])+1)) or \
                             (len(split_date[0])==4) and int(split_date[0]) in range(1945, int(str(date.today().year))+1):
                                    if(len(split_date[1])==2 and int(split_date[1]) in range(1,13)):
                                        if(len(split_date[2])==2 and int(split_date[2])in range(1,32)):
                                            date_col_copy_final[nat_indices]=(split_date[0] if len(split_date[0])==4 else ('19' if int(split_date[0])>45 else '20')+
                                         split_date[0])+sep+split_date[1]+sep+split_date[2]
                                elif (len(split_date[0]==2) and int(split_date[0]) in range(1,13)):
                                    if(len(split_date[1])==2 and int(split_date[1]) in range(1,32)):
                                        if(len(split_date[2])==2 and int(split_date[2]) in range(0,int(str(date.today().year)[2:])+1)) or \
                                     (len(split_date[2])==4) and int(split_date[0]) in range(1945, int(str(date.today().year))+1):
                                            date_col_copy_final[nat_indices]=(split_date[2] if len(split_date[2])==4 else ('19' if int(split_date[2])>45 else '20')+
                                         split_date[2])+sep+split_date[0]+sep+split_date[1]
                                elif (len(split_date[0])==2) and int(split_date[0]) in range(1,32):
                                    if (len(split_date[1])==2) and split_date[1] in range(1,13):
                                        if ((len(split_date[2])==2) and int(split_date[2]) in range(0, int(str(date.today().year)[2:])+1)) or \
                                     (len(split_date[2])==4) and int(split_date[2]) in range(1945, int(str(date.today().year))+1):
                                            date_col_copy_final[nat_indices] = (split_date[2] if len(split_date[2])==4 else ('19' if int(split_date[2])>45 else '20')+split_date[2])+ \
                                         sep+split_date[1]+sep+split_date[0]
                    date_col_copy_final[nat_indices] = pd.to_datetime(date_col_copy_final[nat_indices],format=date_format[:2]+sep+date_format[2:4]+sep+date_format[4:6])
                return date_col_copy_final
        except ValueError as v:
            print(v)
    else:
        print('date_obj_col should be of type pandas.core.series.Series')


earthquakes.loc[:,'date_parsed'] = parse_dates(date_obj_col=earthquakes.loc[:,'Date'],dateonly=True,date_format='%m%d%Y')


# + colab={"base_uri": "https://localhost:8080/"} id="nOy4LOC7Ag9-" executionInfo={"status": "ok", "timestamp": 1708967647826, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}} outputId="2e1637ff-a360-435e-f62c-e286974d646c"
earthquakes['date_parsed'].head()

# + [markdown] id="mSHz4j-KAg9_"
# ## Select the day of the month
#
# Create a Pandas Series `day_of_month_earthquakes` containing the day of the month from the "date_parsed" column.

# + id="sIkO2My8Ag9_" executionInfo={"status": "ok", "timestamp": 1708967647826, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}}
# try to get the day of the month from the date column
day_of_month_earthquakes = earthquakes.loc[:,'date_parsed'].dt.day

# + [markdown] id="NyxL00fsAg9_"
# ## Plot the day of the month to check the date parsing
#
# Plot the days of the month from earthquake dataset.

# + id="mAM-VNGUAg9_" colab={"base_uri": "https://localhost:8080/", "height": 472} executionInfo={"status": "ok", "timestamp": 1708967648684, "user_tz": -330, "elapsed": 863, "user": {"displayName": "Kiran S", "userId": "14416592016092004339"}} outputId="a94d05f5-1f69-402f-c678-24fe880f9b0b"
# Plot a distplot to visualize the distribution of earthquakes on different days of the month from year 1965-2016
sns.histplot(day_of_month_earthquakes,kde=True)
plt.xlabel('Days of the Month')
plt.title('Days of the Month vs Number of Earthquakes from 1965-2016')
plt.show()



# + [markdown] id="iTiuHrnczulJ"
# ### **Conclusion**
#
# **The graph shows a relatively even distribution of earthquakes across the days of the month,which is what we would expect.**

# + id="trpPhztEbDgD"

