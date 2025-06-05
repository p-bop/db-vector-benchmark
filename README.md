# db-vector-benchmark

This repository contains the full code and benchmarking setup used for my Honours thesis:  
**“Benchmarking General Purpose Databases for Vector Search: A Comparative Study Across MongoDB, Neo4j, Redis, and TiDB.”**

The project evaluates the vector search performance of four widely used general-purpose databases using both brute-force and native approximate nearest neighbour (ANN) methods.

---

## 🧪 What’s Inside

The experiments compare:
- **Recall@1**, **Average Query Time**, and **Throughput**
- **Brute-force vs. HNSW-based native search**
- Small-scale (GloVe, 100d) and large-scale (QQP, 384d) vector datasets

All results were generated from the code in this repository.

---

## 📁 Repository Structure

db-vector-benchmark/
├── ca.pem # TLS certificate for TiDB Cloud access
├── docker/ # Docker Compose files for Redis, Neo4j, and local TiDB
├── notebooks/ # Jupyter notebooks for benchmarking each system
├── results/ # CSV logs of Recall@1, query time, and throughput
├── scripts/ # Data loading + benchmark execution scripts
├── LICENSE # MIT License
├── README.md # You’re here

---

## 📦 Datasets

The datasets used in this project are hosted on Google Drive due to their size:

👉 Download GloVe and QQP embeddings [here]([url](https://drive.google.com/drive/folders/1hMHZysRs5k1jRDsScjTBICoHBefdGzEE?usp=drive_link))

- `glove/`: Preprocessed GloVe 100-dimensional word vectors (400,000 vectors)
- `large/`: Combined sentence embeddings from Quora Question Pairs (QQP)

---

## 📄 Thesis Report

The full thesis PDF will be uploaded here soon after final submission and university approval.

---

## 💡 Highlights

- Redis achieved sub-millisecond latency and 100% Recall@1 on both datasets.
- MongoDB offered consistent hybrid query performance and high recall.
- Neo4j enabled structural reasoning within vector similarity queries.
- TiDB provided SQL-native integration with moderate performance at scale.

---

## 🧠 Code Availability

You can reuse the benchmarking pipeline with your own data by modifying paths and configs in `scripts/` and `notebooks/`.

If you use this work or build on it, please consider acknowledging the repository.  
Questions or feedback are always welcome!
