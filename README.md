# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

## CODING:
## HANDLE MISSING VALUES

```
import pandas as pd
import numpy as np
from scipy import stats
import os

def load_data(file):
    if not os.path.exists(file):
        print("\nFile not found:", file)
        return None
    return pd.read_csv(file)
print("\nACTIVITY 1: DATA SET")

df = load_data("/Data_set.csv")

if df is not None:
    print("\nFirst 10 Rows\n", df.head(10))
    print("\nMissing Values\n", df.isnull().sum())

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print("\nMissing Values After Cleaning\n", df.isnull().sum())
    df.to_csv("Data_set_cleaned.csv", index=False)
```
## OUTPUT:
<img width="463" height="343" alt="Screenshot 2026-11-02 150803" src="https://github.com/user-attachments/assets/f98867c3-4c8f-4832-91de-d4f754ab5bf6" />
<img width="404" height="215" alt="Screenshot 2026-11-02 150818" src="https://github.com/user-attachments/assets/e5c0e8a0-3256-4476-969f-ec5bce4ebbeb" />

CODIND:
## ACTIVITY 2: REMOVE DUPLICATES AND INSPECT INCONSISTENCIES
```
print("\nACTIVITY 2: SAMPLEIDS")

df = load_data("/SAMPLEIDS.csv")

if df is not None:
    print("\nOriginal Data\n", df.head(10))
    print("\nDuplicate Rows:", df.duplicated().sum())

    for col in df.select_dtypes(include=["object"]).columns:
        print("\nInconsistent Values in", col)
        print(df[col].dropna().unique())

        df[col] = df[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    df = df.drop_duplicates()

    print("\nData After Removing Duplicates\n", df.head(10))
    print("\nDuplicates Remaining:", df.duplicated().sum())
    df.to_csv("SAMPLEIDS_cleaned.csv", index=False)
```
## OUTPUT:
<img width="512" height="282" alt="image" src="https://github.com/user-attachments/assets/bc423d73-2da3-4212-96e7-8fafd266c852" />
<img width="511" height="435" alt="image" src="https://github.com/user-attachments/assets/d1cfb29f-e701-43f6-b618-87ab7516e71b" />

## CODING:
## ACTIVITY 3: DETECT UNUSUAL HEIGHT MEASUREMENTS
```
print("\nACTIVITY 3: HEIGHTS")

df = load_data("/content/heights.csv")

if df is not None:
    print("\nHeight Data\n", df)

    col = "height"
    values = df[col].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print("\nQ1:", q1)
    print("Q3:", q3)
    print("IQR:", iqr)
    print("Lower Limit:", lower)
    print("Upper Limit:", upper)
    print("\nUnusual Height Measurements\n", outliers)
```
## OUTPUT:
<img width="454" height="394" alt="image" src="https://github.com/user-attachments/assets/9c2bee3a-2715-4408-bcd5-dbe691776a8b" />



## CODING:
## IQR AND Z-SCORE ON ALL NUMERICAL COLUMNS:
```
print("\nACTIVITY 4: IRIS")

df = load_data("/content/iris.csv")

if df is not None:
    numeric_cols = df.select_dtypes(include=np.number).columns

    iqr_mask = pd.Series(False, index=df.index)
    z_mask = pd.Series(False, index=df.index)

    for col in numeric_cols:
        values = df[col]

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (values < lower) | (values > upper)
        iqr_mask |= mask

        print("\nIQR Outliers in", col)
        print(df.loc[mask, [col]])

        if values.std() != 0:
            z = np.abs(stats.zscore(values))
            mask_z = pd.Series(z > 3, index=df.index)
            z_mask |= mask_z

            print("\nZ-Score Outliers in", col)
            print(df.loc[mask_z, [col]])

    df.loc[~iqr_mask].to_csv("iris_iqr_cleaned.csv", index=False)
    df.loc[~z_mask].to_csv("iris_zscore_cleaned.csv", index=False)
```

## OUTPUT:
<img width="430" height="419" alt="image" src="https://github.com/user-attachments/assets/64e7934a-467c-46e8-bb2c-24beee0c0f01" />



## CODING:
## CLEAN NUMERICAL AND CATEGORICAL DATA:
```
print("\nACTIVITY 5: LOAN DATA")

df = load_data("/content/Loan_data.csv")

if df is not None:
    print("\nOriginal Data\n", df.head(10))
    print("\nMissing Values\n", df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())

    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print("\nCleaned Data\n", df.head(10))
    print("\nMissing Values After Cleaning\n", df.isnull().sum())
    print("\nDuplicates After Cleaning:", df.duplicated().sum())

    df.to_csv("Loan_data_cleaned.csv", index=False)

print("\nALL AVAILABLE ACTIVITIES COMPLETED")
```
## OUTPUT:  
<img width="375" height="403" alt="image" src="https://github.com/user-attachments/assets/00462755-d7a7-4bf3-84b5-58121d6416af" />  

<img width="455" height="392" alt="image" src="https://github.com/user-attachments/assets/ef10d3ff-d2b4-4db6-92ea-40005ae8bf4a" />

## INTERPRETATION:  

Data Science involves collecting, cleaning, and analyzing data to obtain meaningful results. It handles missing values using median and mode, removes duplicate records, identifies inconsistencies, and detects outliers using IQR and Z-score methods. Finally, it helps in understanding data and making better decisions.


## Result:
Thus, the datasets were successfully cleaned by handling missing values, removing duplicates, identifying inconsistencies, and detecting outliers using IQR and Z-score methods.
         
