import pandas as pd
import random

prod_list = ['prod1', 'prod2', 'prod3', 'prod4']

df_list = []

for prod in prod_list:

    df = pd.DataFrame(
        {
            "Date": pd.date_range(
                start="2021-01-01", end="2022-10-22", freq="min", closed="left"
            )
        }
    )
    values_list = []
    for i in range(0,df.shape[0]):
        values_list.append(random.uniform(0.5,2.9))



    df['values'] = values_list

    df['prod'] = prod

    df_list.append(df)

final_df = pd.concat(df_list).reset_index(drop = True)

final_df.to_csv('final_time_series_data.csv')