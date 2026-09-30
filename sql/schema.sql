-- SQLite Schema for AI Credit Risk Prediction (Give Me Some Credit + Auxiliary Tables)

DROP TABLE IF EXISTS loan_applications;
DROP TABLE IF EXISTS regional_macro;
DROP TABLE IF EXISTS applicants;

CREATE TABLE applicants (
    ApplicantID INTEGER PRIMARY KEY,
    SeriousDlqin2yrs INTEGER NOT NULL,
    RevolvingUtilizationOfUnsecuredLines REAL,
    age INTEGER,
    "NumberOfTime30-59DaysPastDueNotWorse" INTEGER,
    DebtRatio REAL,
    MonthlyIncome REAL,
    NumberOfOpenCreditLinesAndLoans INTEGER,
    NumberOfTimes90DaysLate INTEGER,
    NumberRealEstateLoansOrLines INTEGER,
    "NumberOfTime60-89DaysPastDueNotWorse" INTEGER,
    NumberOfDependents REAL
);

CREATE TABLE regional_macro (
    Region TEXT PRIMARY KEY,
    RegionalUnemploymentRate REAL,
    MedianHouseholdIncome INTEGER,
    CostOfLivingIndex REAL
);

CREATE TABLE loan_applications (
    ApplicationID INTEGER PRIMARY KEY,
    ApplicantID INTEGER NOT NULL,
    Region TEXT NOT NULL,
    LoanPurpose TEXT NOT NULL,
    EmploymentStatus TEXT NOT NULL,
    RequestedLoanAmount REAL NOT NULL,
    LoanTermMonths INTEGER NOT NULL,
    EstimatedAPR REAL NOT NULL,
    FOREIGN KEY (ApplicantID) REFERENCES applicants(ApplicantID),
    FOREIGN KEY (Region) REFERENCES regional_macro(Region)
);

CREATE INDEX idx_applicants_target ON applicants(SeriousDlqin2yrs);
CREATE INDEX idx_applicants_age ON applicants(age);
CREATE INDEX idx_loans_applicant ON loan_applications(ApplicantID);
CREATE INDEX idx_loans_region ON loan_applications(Region);
