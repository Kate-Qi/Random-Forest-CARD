import pandas as pd
import numpy as np
import base64
import io
from tqdm import tqdm

def decode_embedding(encoded_string):
    if encoded_string == 'NA' or pd.isna(encoded_string):
        return None
    try:
        binary_data = base64.b64decode(encoded_string)
        buffer = io.BytesIO(binary_data)
        return np.load(buffer)
    except:
        return None

input_file = "embeddings.csv"
output_file = "embeddings.npz"
embedding_column = "protein_embedding"

df = pd.read_csv(input_file)
decoded_embeddings = []
valid_indices = []

for i, encoded in tqdm(enumerate(df[embedding_column]), total=len(df)):
    embedding = decode_embedding(encoded)
    if embedding is not None:
        decoded_embeddings.append(embedding)
        valid_indices.append(i)

if decoded_embeddings:
    embeddings_array = np.stack(decoded_embeddings)
    np.savez(output_file, embeddings=embeddings_array, valid_indices=np.array(valid_indices))
