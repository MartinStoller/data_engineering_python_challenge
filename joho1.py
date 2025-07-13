import pandas as pd
import numpy as np
import csv
import os
import glob
import fastparquet
from pathlib import Path
from multiprocessing import Pool
import time

start_time = time.time()

PRODUCTS_CSV = "general_inputdata/products.csv"
EUR_USD_CSV = "general_inputdata/eur_usd_last10y.csv"
TRANSACTION_FOLDER = "benchmark_1_inputdata/"
CHUNKSIZE = 100_000
PARQUET_FOLDER = "parquet_files"
os.makedirs(PARQUET_FOLDER, exist_ok=True)
PARQUET_FILE_LOCATION = "parquet_files/sales_postprocessed_"

path = os.getcwd()
transaction_parquet_files = glob.glob(os.path.join(path, PARQUET_FOLDER, "*.parquet"))

for parquet_file in transaction_parquet_files:
    os.remove(parquet_file)

# products Tabelle ranholen
products = pd.read_csv(PRODUCTS_CSV)

# Umrachnungstabelle umbauen und mit Wochenenddaten auffuellen
euro_dollar_umrechnung = pd.read_csv(EUR_USD_CSV)
euro_dollar_umrechnung = euro_dollar_umrechnung.drop([0, 1])
euro_dollar_umrechnung = euro_dollar_umrechnung.rename(columns={"Price": "timestamp"})
euro_dollar_umrechnung['timestamp'] = pd.to_datetime(euro_dollar_umrechnung['timestamp']).dt.date
euro_dollar_umrechnung["eur_usd"] = pd.to_numeric(euro_dollar_umrechnung['eur_usd'])
euro_dollar_umrechnung.set_index("timestamp", drop=True, inplace=True)
euro_dollar_umrechnung = euro_dollar_umrechnung.sort_index()
time_range = pd.date_range(euro_dollar_umrechnung.iloc[0].name, euro_dollar_umrechnung.iloc[-1].name)
euro_dollar_umrechnung = euro_dollar_umrechnung.reindex(time_range, method="ffill")
euro_dollar_umrechnung = euro_dollar_umrechnung.reset_index(names=["timestamp"])
euro_dollar_umrechnung['timestamp'] = pd.to_datetime(euro_dollar_umrechnung['timestamp']).dt.date

# den Ordner mit allen Transactionen ranholen
path = os.getcwd()
transaction_files = glob.glob(os.path.join(path, TRANSACTION_FOLDER, "*.csv"))

# hilfs int fuer die Bennenung der ergebnis files


performance_product = pd.Series([])
agg_sales = pd.Series([])
agg_production_costs = pd.Series([])


def add_to_agg_net_profit_series(net_profit_chunk: pd.DataFrame):
    global performance_product
    if performance_product.empty:
        performance_product = pd.concat([performance_product, net_profit_chunk])
    else:
        performance_product = performance_product.add(net_profit_chunk, fill_value=0.0)
    return


def add_to_agg_sales_series(sales_chunk: pd.DataFrame):
    global agg_sales
    if agg_sales.empty:
        agg_sales = pd.concat([agg_sales, sales_chunk])
    else:
        agg_sales = agg_sales.add(sales_chunk, fill_value=0.0)
    return


def add_to_ag_prod_costs_series(prod_cost_chunk: pd.DataFrame):
    global agg_production_costs
    if agg_production_costs.empty:
        agg_production_costs = pd.concat([agg_production_costs, prod_cost_chunk])
    else:
        agg_production_costs = agg_production_costs.add(prod_cost_chunk, fill_value=0.0)
    return


def get_result_df(df: pd.DataFrame, agg_performance_values: pd.Series, agg_sales_values: pd.Series,
                  agg_prod_costs_values: pd.Series):
    df_grouped_net_profit = agg_performance_values.sort_values(ascending=False)
    df_product_name_best = \
    df.query("product_id ==" + str(df_grouped_net_profit.head(1).index.values[0]))["product_name"].values[0]
    df_product_name_worst = \
    df.query("product_id ==" + str(df_grouped_net_profit.tail(1).index.values[0]))["product_name"].values[0]
    df_agg_production_cost_best = agg_prod_costs_values.at[df_grouped_net_profit.head(1).index.values[0]]
    df_agg_production_cost_worst = agg_prod_costs_values.at[df_grouped_net_profit.tail(1).index.values[0]]
    df_agg_sales_best = agg_sales_values.at[df_grouped_net_profit.head(1).index.values[0]]
    df_agg_sales_worst = agg_sales_values.at[df_grouped_net_profit.tail(1).index.values[0]]

    return_df_data = {
        "product_id": [df_grouped_net_profit.head(1).index.values[0], df_grouped_net_profit.tail(1).index.values[0]],
        "product_name": [df_product_name_best, df_product_name_worst],
        "agg_costs": [df_agg_production_cost_best, df_agg_production_cost_worst],
        "agg_sales": [df_agg_sales_best, df_agg_sales_worst],
        "agg_net_profit": [df_grouped_net_profit.head(1).values[0], df_grouped_net_profit.tail(1).values[0]]
        }
    return_df = pd.DataFrame(return_df_data)

    return return_df


def process_csv(csv_file, filenumber):
    # damit intellisense funzt einmal hart typisieren
    chunk: pd.DataFrame
    parquet_file = Path(PARQUET_FILE_LOCATION + str(fileNumber) + ".parquet")

    # transcation gechunked als reader bereitstellen
    reader = pd.read_csv(csv_file, chunksize=CHUNKSIZE)

    for chunk in reader:
        # droppen der Null Values
        chunk = chunk.dropna()

        # alle producte dran basteln und die Zeilen mit nicht bekannten producten droppen
        chunk = chunk.merge(products, how="left", on="product_id")
        chunk = chunk.dropna()

        # timestamp string zum datum machen und das mit der umrechnungs tabelle joinen
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"]).dt.date
        chunk = chunk.merge(euro_dollar_umrechnung, how="left", on="timestamp")

        # spalte amount mit der umrechnungs spalte verrechnen
        chunk["amount"] = chunk["amount"] * chunk["eur_usd"]

        # net profit errechnen
        chunk["net_profit"] = chunk["amount"] - chunk["production_costs"]

        # datum zum timestamp aendern weil fastparquet sich sonst auf die Schnauze legt ¯\_(ツ)_/¯
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])

        add_to_agg_net_profit_series(chunk.groupby(by=["product_id"]).net_profit.sum())
        add_to_agg_sales_series(chunk["product_id"].value_counts())
        add_to_ag_prod_costs_series(chunk.groupby(by=["product_id"]).production_costs.sum())
        # zum parquet file schreiben
        if parquet_file.exists():
            chunk.to_parquet(parquet_file, engine="fastparquet", append=True)
        else:
            chunk.to_parquet(parquet_file, engine="fastparquet")


# hilfs int fuer die Bennenung der ergebnis files
fileNumber = 1

for file in transaction_files:
    process_csv(file, fileNumber)
    fileNumber += 1

df = pd.read_csv(PRODUCTS_CSV)
print(get_result_df(df, performance_product, agg_sales, agg_production_costs))

end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


