## End to End Data Science project

Kindly follow these steps to run the project:-

## Step1: Download the project folder on to your system

## Step2: Create a new environment
```
conda create -p venv python==3.10

conda activate venv 
```

## Step3: Install required packages
```pip install -r requirements.txt```

## Step4: Create a .env file in the directory and save your database credentials

## Step5: In the utils file, modify the table_name in ```Select * from [table_name]```

## Step6: Uncomment line number 33 and 72 and comment out line no 34 in the data_ingestion.py file inside components folder inside the src folder

## Step7: Then execute the following command:
```python src/pipeline/train_pipeline.py```

## Step7: After execution, kindly visit the data_ingestion.py file and this time comment out the line numbers 33 and 72 and uncomment line number 34.

## Step8: Now run the following command:
``` python app.py ```

## Step9: While app.py is running, paste the following in your browser
```http://127.0.0.1:3000/ ```

