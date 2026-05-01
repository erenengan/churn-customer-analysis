import pandas as pd
from sqlalchemy import create_engine

#Create engine and connection
connection_string = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=localhost,1433;"
    "DATABASE=ChurnCustomerDB;"
    "UID=sa;"
    "PWD=Pass@word123!;"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)

engine = create_engine(f"mssql+pyodbc:///?odbc_connect={connection_string}")

#List of  csv files
brand_health = pd.read_csv("brand_health.csv", low_memory=False)
brand_image = pd.read_csv("brand_image.csv", low_memory=False)
companion = pd.read_csv("Companion.csv", low_memory=False)
competitor = pd.read_csv("competitor.csv", low_memory=False)
day_of_week = pd.read_csv("Dayofweek.csv", low_memory=False)
day_part = pd.read_csv("Daypart.csv", low_memory=False)
need_state = pd.read_csv("NeedstateDayDaypart.csv", low_memory=False)
sa_var = pd.read_csv("sa_var.csv", low_memory=False)
segmentation = pd.read_csv("segmentation.csv", low_memory=False)


#Convert csv into sql
brand_health.to_sql("brand_health", engine, if_exists="replace", index=False)
brand_image.to_sql("brand_image", engine, if_exists="replace", index=False)
companion.to_sql("companion", engine, if_exists="replace", index=False)
competitor.to_sql("competitor", engine, if_exists="replace", index=False)
day_of_week.to_sql("day_of_week", engine, if_exists="replace", index=False)
day_part.to_sql("day_part", engine, if_exists="replace", index=False)
need_state.to_sql("need_state", engine, if_exists="replace", index=False)
sa_var.to_sql("sa_var", engine, if_exists="replace", index=False)
segmentation.to_sql("segmentation", engine, if_exists="replace", index=False)


print("Upload complete!")