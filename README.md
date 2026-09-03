# Chasing-the-Right-Variables-Loan-Amount-Prediction
A regression project on a deliberately messy, generated loan dataset (finance_loan_DIRTY.csv), built to explore variable selection — specifically, how far pd.get_dummies() and VarianceThreshold can take you on their own, and where they run out of road.
### Goal
Predict the loan amount a bank would approve for an applicant (a regression problem) based on their financial profile.
## Data Cleaning
Standardized inconsistent categorical casing (region, employment_type had duplicate values like "Abuja" / "ABUJA" / "abuja")
Stripped currency symbols/commas from numeric-as-string columns and cast to float
Converted the string "pending review" in the target column to NaN instead of silently zeroing it
Dropped 5 exact duplicate rows
Imputed missing values: 0 for columns where zero is meaningful (debt, employment years, collateral, etc.), column mean for applicant_monthly_income_usd and credit_score
Removed outliers via the IQR method (1.5× IQR) on flagged columns — including an employment_years value of 200 — reducing the dataset from 216 to 183 rows
## Variable Selection
Dummy encoding — pd.get_dummies(df, drop_first=True) on categorical columns, expanding the dataset to 19 columns
Variance Threshold — scaled features with MinMaxScaler, then applied VarianceThreshold(threshold=0.03) to drop near-constant columns
OLS sanity check — a statsmodels OLS fit on the full feature set returned R² = 0.228 (p = 3.81 × 10⁻⁵), with only 3 of 15 features carrying real weight: applicant_monthly_income_usd, credit_score, collateral_value_usd
## Results
Split		
Train	MSE 43,457,359	R² 0.919
Test	MSE 8,501,703,563	 R² -0.058
A ~200x jump in MSE from train to test is a textbook overfitting signature. With only 183 rows and a double-digit feature count post-encoding, the LinearRegression model had more than enough freedom to memorize training noise rather than learn transferable signal

## Key Takeaway

Dummy encoding and VarianceThreshold answer "which variables carry information?" — not "will this model generalize?" They did their job correctly: VarianceThreshold removed near-constant columns, exactly as designed. It was never built to catch a parameter-count-vs-sample-size problem, and no threshold tuning would have fixed that.

The train/test split is where the hypothesis actually gets tested — not the training fit.

## Next Steps
 Ridge/Lasso regularization
