#Imp libraries import
import pandas as pd, numpy as np
import argparse
from prophet import Prophet 
from prophet.plot import plot_plotly, plot_components_plotly
from statsmodels.tools.eval_measures import rmse
from prophet.serialize import model_to_json, model_from_json


def modelling(filename):
    df = pd.read_csv(filename + ".csv")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df[["Date","values"]]
    df.columns= ['ds','y']

    #Dataset split
    train = df.iloc[:len(df)-365]
    test = df.iloc[len(df)-365:]

    #test.plot(x='ds', y='y', figsize=(18,6))

    #FB Prophet object and fitting 
    m = Prophet()
    m.fit(train)
    future = m.make_future_dataframe(periods=365)
    foreacst = m.predict(future)
    #foreacst.tail()

    foreacst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    predictions = foreacst.iloc[-365:]['yhat']

    #Print Root Mean Squared Error
    print("RMSE b/w actual and predicted values:", rmse(predictions, test['y']))
    print("Mean value of test dataset:", test['y'].mean())

    #Save Model
    with open('Model_save.json', 'w') as fout:
        fout.write(model_to_json(m))  

def main():

    # command-line parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str, help="the file name of the csv file, no .csv is needed.")
    args = parser.parse_args()

    modelling(args.file_name)


if __name__ == "__main__":
    main()
