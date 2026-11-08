# - brauche global variablen damit prozess sich die teilen kann
# - nutze pyarrow weil read_csv mit pandas sehr langsam war und eh praktisch für parquet
import glob
import math
import multiprocessing as mp
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

start_time = time.time()

# pfade

TRANSACTION_PATH = "../benchmark_2_inputdata/transactions*.csv"
PRODUCTS_PATH = "../general_inputdata/products*.csv"
EXCHANGE_PATH = "../general_inputdata/eur_usd_last10y.csv"
OUTPUT_PATH = Path("./")

# random zeugs welches ich brauche

NUMBER_WORKERS = os.cpu_count()
MAX_WORKES = 2 * NUMBER_WORKERS
MIN_BLOCK = 4 * 2**20  # 4MB
MAX_BLOCK = 64 * 2**20  # 64MB

CSV_TYPES = {
    "transaction_id": pa.int64(),
    "customer_id": pa.int64(),
    "timestamp": pa.timestamp("s"),
    "product_id": pa.int64(),
    "amount": pa.float64(),
    "location": pa.string(),
}

CONVERT_OPTIONS = pacsv.ConvertOptions(
    column_types=CSV_TYPES,
    null_values=[""],
    strings_can_be_null=True,
)

OUT_SCHEMA = pa.schema([
    ("transaction_id", pa.int64()),
    ("customer_id", pa.int64()),
    ("timestamp", pa.timestamp("s")),
    ("product_id", pa.int64()),
    ("amount", pa.float64()),
    ("location", pa.string()),
    ("product_name", pa.string()),
    ("category", pa.string()),
    ("production_costs", pa.float64()),
    ("net_profit", pa.float64()),
])

products = pd.read_csv(
    glob.glob(PRODUCTS_PATH)[0],
    dtype={"product_id": "int64", "product_name": "string", "category": "string", "production_costs": "float64"},
)

exchange = pd.read_csv(EXCHANGE_PATH, skiprows=3, names=["date", "rate"], parse_dates=["date"])
exchange = exchange.set_index("date")["rate"].sort_index()
# stack overflow magic: ffill (forward fill) fuellt NaNs genau mit den vorherigen werten auf also genau was wir wollen
all_days = pd.date_range(exchange.index.min(), exchange.index.max(), freq="D")
exchange_daily = exchange.reindex(all_days).ffill()

# da stand das bringt extra performance haha also why not
exchange_rate = exchange_daily.to_numpy(dtype="float64")
exchange_first_day = int(np.datetime64(exchange_daily.index[0].date(), "D").astype(np.int64))

print(f"{len(products)} produkte, {len(exchange_daily)} viele tage ({len(exchange)} vor fill)")


# funktion wieder stark inspiriert von stackoverflow magic. die allgemeine idee ist dass pyarrow die daten eh als
# byte stream moechte und direkt auf bytes chunken es sehr viel einfach macht groesse von einzelnen chunks zu
# kontrollieren + man sparrt sich dekodierungs overhead. hier gab es ein paar gottlose bugs zb start vs start-1
#
# aber die funktion sollte grob vor allem dafür sorgen dass wir bytes immer eginnend in der naechsten anfangen zu
# lesen und mit der letzten sauber aufhoeren
def read_byte_chunks(path, start, end, data_start):
    with open(path, "rb") as f:
        if start > data_start:
            f.seek(start - 1)
            f.readline()
        else:
            f.seek(start)
        pos = f.tell()
        if pos >= end:
            return b""
        data = f.read(end - pos)
        if data and not data.endswith(b"\n"):
            data += f.readline()
    return data


def process_task(task):
    path, start, end, data_start, column_names = task

    data = read_byte_chunks(path, start, end, data_start)
    if not data:
        return None, None

    table = pacsv.read_csv(
        pa.BufferReader(data),
        read_options=pacsv.ReadOptions(column_names=column_names, use_threads=False),
        convert_options=CONVERT_OPTIONS,
    )
    del data

    # nulls weg damit man nicht unnoetig viel nach pandas konvertiert
    table = table.drop_null()
    df = table.to_pandas()
    del table

    # pandas join funktion
    df = df.merge(products, on="product_id", how="inner")

    # euro nach dollar
    days = df["timestamp"].to_numpy().astype("datetime64[D]").astype(np.int64) - exchange_first_day
    known = (days >= 0)
    if not known.all():
        df, days = df[known], days[known]
    # hier zahlt es sich in der theorie aus das wir numpy array verwenden
    rate = exchange_rate[np.minimum(days, len(exchange_rate) - 1)]
    df["amount"] = df["amount"].to_numpy() * rate

    df["net_profit"] = df["amount"] - df["production_costs"]

    # chunk fuer chunk aggregieren ist okay weil + assoziativ ist
    agg = df.groupby("product_id", sort=False)[["production_costs", "amount", "net_profit"]].sum()

    out = pa.Table.from_pandas(df, schema=OUT_SCHEMA, preserve_index=False)
    return out, agg


def build_tasks(paths):
    tasks = []
    for file_idx, path in enumerate(paths):
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            header = f.readline()
        data_start = len(header)
        column_names = header.decode().strip().split(",")

        block = math.ceil((size - data_start) / NUMBER_WORKERS)
        block = min(MAX_BLOCK, max(MIN_BLOCK, block))

        for start in range(data_start, size, block):
            end = min(start + block, size)
            tasks.append((file_idx, (path, start, end, data_start, column_names)))
    return tasks


transaction_paths = sorted(glob.glob(TRANSACTION_PATH))
output_paths = [OUTPUT_PATH / f"sales_postprocessed_{Path(p).stem}.parquet" for p in transaction_paths]
tasks = build_tasks(transaction_paths)

print(f"{len(transaction_paths)} tabellen, {len(tasks)} aufgaben")

# Das hier hat wieder einiges an Stackoverflow Recherche grebraucht bevor es lief. Aber hoffe es passt jetzt so

final_agg = None
writer = None
current_file = None
written_paths = []

ctx = mp.get_context("fork")

# try block wichtig weil parquet anscheinend irgendwelche footer haben die geschrieben werden muessen
# weil sonst datei unleserlich und das passiert erst bei writer.close
try:
    with ctx.Pool(NUMBER_WORKERS) as pool:
        task_iter = iter(tasks)
        pending = deque()

        def submit_next():
            nxt = next(task_iter, None)
            if nxt is not None:
                file_idx, task = nxt
                pending.append((file_idx, pool.apply_async(process_task, (task,))))

        for _ in range(MAX_WORKES):
            submit_next()

        while pending:
            file_idx, async_result = pending.popleft()
            table, agg = async_result.get()
            submit_next()

            # hier muss man aufpassen weil man leicht in die falsche datei schreibt. deshalb wichtig file_idx zu merken
            if file_idx != current_file:
                if writer is not None:
                    writer.close()
                writer = pq.ParquetWriter(output_paths[file_idx], OUT_SCHEMA)
                written_paths.append(output_paths[file_idx])
                current_file = file_idx

            if table is not None and table.num_rows:
                writer.write_table(table)
            if agg is not None:
                final_agg = agg if final_agg is None else final_agg.add(agg, fill_value=0)
finally:
    if writer is not None:
        writer.close()

print("geschrieben:", *[str(p) for p in written_paths], sep="\n  ")

summary = (
    final_agg
    .join(products.set_index("product_id")["product_name"])
    .rename(columns={"production_costs": "total_costs",
                     "amount": "total_sales",
                     "net_profit": "total_net_profit"})
)
result = (
    summary.loc[[summary["total_net_profit"].idxmax(), summary["total_net_profit"].idxmin()]]
    .reset_index()
    [["product_id", "product_name", "total_costs", "total_sales", "total_net_profit"]]
)
print(result.to_string(index=False))

end_time = time.time()
print()
print("Total runtime: ", end_time - start_time)
