#importing required modules/libraries
import snowflake.connector
import pandas as pd

#Set connection
try:
    conn = snowflake.connector.connect(
        user='EMORY_UNIV_TEAM_6',
        password='B2_AA.DataSciproject!',
        account='AA-ITOR_UNIVERSITY',
        warehouse='UNIVERSITY_READER',
        database='LOCAL_DATABASE',
        schema='ORAAUE', 
        role = 'EMORY_STUDENT_2025'
)
except Exception as e:
    print("Error in connection, Here is the error message....\n" + str(e))

#Extract and load data	
try:
    # Create cursor object
    cur = conn.cursor()
    
    # Prepare the SQL statement
    sqlquery = 'select*from LOCAL_DATABASE.ORAAUE.SEQUENCE_SPOILAGE'
    
    # Run the SQL statement and capture query output in a Pandas dataframe
    cur.execute(sqlquery)
    df = cur.fetch_pandas_all()
    
    # Load pandas dataframe into a file, make sure you escape the backslash if using windows in the filepath like this C:\\mylocation\\myfolder\\filename.csv
   ## filename = 'Flight Info.csv'
   ## df.to_csv(filename, sep='|',encoding='utf-8')
    
except Exception as e:
    print("Error: the error message is..... \n" + str(e))
finally:
    conn.close()