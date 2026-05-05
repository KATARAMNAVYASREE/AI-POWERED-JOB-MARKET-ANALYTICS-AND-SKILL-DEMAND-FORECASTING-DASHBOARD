import pandas as pd

def load_and_preprocess():

    data = pd.read_csv("job_market_dataset.csv")

    # remove missing values
    data = data.dropna()

    # remove duplicates
    data = data.drop_duplicates()

    # convert year column
    data['work_year'] = data['work_year'].astype(int)

    # split skills column
    data['skills'] = data['skills'].str.split(',')

    # explode skills
    data = data.explode('skills')

    # clean spaces
    data['skills'] = data['skills'].str.strip()

    return data

