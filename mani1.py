import pandas as pd
import time
import numpy as np
import os
from multiprocessing import Pool
from functools import partial
import uuid
import shutil
from pathlib import Path


def read_conversion_file() -> pd.DataFrame:
    eur_usd_conversion = pd.read_csv('eur_usd_last10y.csv')
    eur_usd_conversion = eur_usd_conversion.rename(columns={'Price': 'date'})
    eur_usd_conversion['date'] = pd.to_datetime(eur_usd_conversion['date'], format='%Y-%m-%d', errors='coerce')
    eur_usd_conversion = eur_usd_conversion.dropna()
    eur_usd_conversion = eur_usd_conversion.set_index('date')
    eur_usd_conversion['eur_usd'] = eur_usd_conversion['eur_usd'].astype('float')
    eur_usd_conversion = eur_usd_conversion.reindex(pd.date_range(start='2014-06-02', end='2024-05-31', freq='D'), method='ffill')
    return eur_usd_conversion

def process_file(file_id: int, chunk_size: int, products: pd.DataFrame, eur_usd_conversion: pd.DataFrame) -> pd.DataFrame:
    os.mkdir("temp_" + str(file_id))
    with pd.read_csv('transactions/' + str(file_id) + '.csv', chunksize=chunk_size) as reader:
        result_df = pd.DataFrame()
        with Pool(processes=6) as pool:
            for intermediate_result in pool.imap_unordered(partial(process_chunk, eur_usd_conversion=eur_usd_conversion, file_id=file_id, products=products), reader):
                if result_df.empty:
                    result_df = intermediate_result
                else:
                    result_df = result_df.combine(intermediate_result, lambda x, y: x + y)
        merge_files(file_id)
        return result_df


def process_chunk(chunk, eur_usd_conversion: pd.DataFrame, file_id: int, products: pd.DataFrame) -> pd.DataFrame:
    transactions = chunk.dropna()
    transactions = transactions.join(products.set_index('product_id'), on='product_id', how='inner')
    transactions['product_id'] = transactions['product_id'].astype('int')
    transactions = transactions.sort_values('product_id')
    transactions['date'] = pd.to_datetime(transactions['timestamp'], errors='coerce').dt.normalize()
    transactions = transactions.join(eur_usd_conversion, on='date', how='left')
    transactions['amount'] = transactions['eur_usd'] * transactions['amount']
    transactions['amount'] = transactions['amount'].map(lambda x: np.floor(x * 1e2) * 1e-2)
    transactions = transactions.drop(columns=['eur_usd'])
    transactions['net_profit'] = transactions['amount'] - transactions['production_costs']
    transactions.to_parquet('temp_' + str(file_id) + '/' + str(uuid.uuid4()) + '.parquet', engine='fastparquet')
    return transactions[['product_id', 'amount', 'production_costs', 'net_profit']].groupby('product_id').sum()

def merge_files(file_id: int) -> None:
    output_filename = 'sales_postprocessed_' + str(file_id) + '.parquet'
    temp_directory = 'temp_' + str(file_id)
    directory = os.fsencode(temp_directory)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if output_filename.endswith('.parquet'):
            read_file = pd.read_parquet(temp_directory + '/' + str(filename), engine='fastparquet')
            output_path = Path(output_filename)
            if output_path.exists():
                read_file.to_parquet(output_filename, engine='fastparquet', append=True)
            else:
                read_file.to_parquet(output_filename, engine='fastparquet')
    shutil.rmtree(temp_directory)


if __name__ == '__main__':
    for file in os.listdir('./'):
        if file.endswith('.parquet'):
            os.remove(file)
    start = time.time()
    eur_usd_conversion = read_conversion_file()
    products = pd.read_csv('products.csv').dropna()

    directory = os.fsencode('./transactions')
    file_ids = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.csv'):
            filename_without_ext = filename.split('.')[0]
            file_ids.append(int(filename_without_ext))

    result = pd.DataFrame()
    for file_id in file_ids:
        intermediate_result = process_file(file_id, int(1e6), products, eur_usd_conversion)
        if result.empty:
            result = intermediate_result
        else:
            result = result.combine(intermediate_result, lambda x, y: x + y)



    max_profit = result['net_profit'].max()
    min_profit = result['net_profit'].min()

    print(pd.concat([result[result['net_profit'] == max_profit], result[result['net_profit'] == min_profit]]))
    end = time.time()
    print()
    print("Total runtime: ", end - start)