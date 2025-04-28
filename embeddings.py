# Run python embeddings.py --input original.csv.
# Supposing input file has the following columns: ARO, drugclass, DNAseq, Proteinseq

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import base64
import io
from tqdm import tqdm
import argparse
import os

def generate_embedding(sequence, tokenizer, model, max_length=1022):
    if pd.isna(sequence) or sequence == "NA" or not isinstance(sequence, str):
        return None
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
    if ';' in sequence:
        sequence = sequence.split(';')[0]
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def encode_embedding(embedding):
    if embedding is None:
        return "NA"
    buffer = io.BytesIO()
    np.save(buffer, embedding)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def main():
    parser = argparse.ArgumentParser(description='Generate protein embeddings with ESM-2')
    parser.add_argument('-i', '--input', required=True, help='Input CSV path')
    parser.add_argument('-o', '--output', help='Output CSV path')
    parser.add_argument('-p', '--protein_col', default='Proteinseq', help='Protein column name')
    parser.add_argument('-e', '--embedding_col', default='protein_embedding', help='Embedding column name')
    parser.add_argument('-m', '--model', default='facebook/esm2_t12_35M_UR50D', help='ESM-2 model')
    args = parser.parse_args()

    base, _ = os.path.splitext(args.input)
    output_path = args.output or f"{base}_with_embeddings.csv"

    df = pd.read_csv(args.input)
    if args.protein_col not in df.columns:
        raise ValueError(f"Column '{args.protein_col}' not found")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    proteins = df[args.protein_col].dropna().unique()
    protein_embeddings = {}
    for seq in tqdm(proteins, desc="Embedding proteins"):
        emb = generate_embedding(seq, tokenizer, model)
        if emb is not None:
            protein_embeddings[seq] = emb

    encoded = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
        seq = row[args.protein_col]
        if pd.isna(seq) or seq == "NA":
            encoded.append("NA")
        else:
            key = seq.split(';')[0] if isinstance(seq, str) and ';' in seq else seq
            encoded.append(encode_embedding(protein_embeddings.get(key)))

    df[args.embedding_col] = encoded
    df.to_csv(output_path, index=False)
    print(f"Saved embeddings to {output_path}")

if __name__ == "__main__":
    main()

