import pandas as pd


def read_csv_by_chunk(file, chunkSize=10000):
    try:
        df = pd.read_csv(file, iterator=True)
    except:
        df = pd.read_csv(file, iterator=True, encoding='gbk')
    loop = True
    chunks = []
    while loop:
        try:
            chunk = df.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    df2 = pd.concat(chunks, ignore_index=True)
    return df2.drop_duplicates(keep='first')