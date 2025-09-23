import sqlite3
import pandas as pd
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from typing import List, Dict, Any
import re
import json

API_KEY_QWEN = "c9b5bdc2162847418f1dc147f1c3ea17.2ew6MArRBcAGNAzt"
API_KEY_DEEPSEEK = ""


class CBDBRAGSystem:
    def __init__(self, db_path: str, embedding_model_name: str = 'BAAI/bge-small-zh-v1.5'):
        """
         初始化系统
        :param db_path: CBDB数据库路径
        :param embedding_model_name: embedding模型名称
        """
        self.db_conn = sqlite3.connect(db_path)
        self.embedding = SentenceTransformer(embedding_model_name)
        self.client = OpenAI(api_key=API_KEY_QWEN)

        # FAISS配置
        self.index = None
        self.documents = []
        self.metadata = []
        self.vector_dim = 384

        self.faiss_index_path = './faiss_index'
        self._load_or_create_faiss_index()

        self._load_database_schema()

    def _load_database_schema(self):
        """加载数据库表结构信息"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        self.tables = [table[0] for table in cursor.fetchall()]
        print(f"发现表: {self.tables}")

    def _load_or_create_faiss_index(self):
        """加载或创建FAISS索引"""
        if os.path.exists(f"{self.faiss_index_path}.index"):
            print("加载已有的FAISS索引......")
            self.index = faiss.read_index(f"{self.faiss_index_path}.index")
            with open(f"{self.faiss_index_path}_meta.pkl", "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadata = data["metadata"]
            print(f"索引加载完成，包含{len(self.documents)}个文档")
        else:
            print("未找到现有索引，需要重新构建")
            self.index = None

    def _save_faiss_index(self):
        """保存FAISS索引"""
        if self.index is not None:
            # 创建目录
            os.makedirs(os.path.dirname(self.faiss_index_path) or ".",
                        exist_ok=True)

            # 保存索引
            faiss.write_index(self.index, f"{self.faiss_index_path}.index")

            # 保存元数据
            with open(f"{self.faiss_index_path}_meta.pkl", "wb") as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata
                }, f)
            print("FAISS索引和元数据已保存")

    def setup_vector_database(self, batch_size: int = 100, limit: int = None):
        """
        将CBDB数据向量化并存储到FAISS中
        :param batch_size: 每次处理的数据量
        :param limit: 处理的表行数限制
        """
        print("正在将CBDB数据向量化并保存到FAISS中......")

        # 查询人物基本信息
        query = """
        SELECT p.c_personid, p.c_name_chn, p.c_index_year, p.c_deathyear,
               p.c_index_addr_id, p.c_notes, b.c_name_chn as birthplace_name
        FROM biog_main p
        LEFT JOIN addresses b ON p.c_index_addr_id = b.c_addr_id
        WHERE p.c_notes IS NOT NULL AND LENGTH(p.c_notes) > 10
        """

        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql_query(query, self.db_conn)
        print(f"从数据库检索到 {len(df)} 条记录")

        all_embeddings = []
        self.documents = []
        self.metadata = []

        # 处理数据并生成嵌入
        for _, row in df.iterrows():
            # 构建文档文本
            doc_text = f"人物: {row['c_name_chn']}"
            if pd.notna(row['c_index_year']):
                birth_year = int(row['c_index_year']) if not pd.isna(row['c_index_year']) else None
                death_year = int(row['c_deathyear']) if not pd.isna(row['c_deathyear']) else None
                doc_text += f", 生卒年: {birth_year or '?'}-{death_year or '?'}"
            if pd.notna(row['birthplace_name']):
                doc_text += f", 籍贯: {row['birthplace_name']}"
            if pd.notna(row['c_notes']):
                doc_text += f", 生平: {row['c_notes'][:500]}"  # 限制文本长度

            self.documents.append(doc_text)
            self.metadata.append({
                "person_id": row['c_personid'],
                "name": row['c_name_chn'],
                "birth_year": row['c_index_year'],
                "death_year": row['c_deathyear'],
                "birthplace": row['birthplace_name'],
                "biography": row['c_notes'][:1000] if pd.notna(row['c_notes']) else ""
            })

        # 批量生成嵌入
        print("生成文本嵌入...")
        for i in range(0, len(self.documents), batch_size):
            batch_docs = self.documents[i:i + batch_size]
            batch_embeddings = self.embedding.encode(batch_docs)
            all_embeddings.append(batch_embeddings)
            print(f"已处理 {min(i + batch_size, len(self.documents))}/{len(self.documents)} 条记录")

        # 合并所有嵌入
        all_embeddings = np.vstack(all_embeddings).astype('float32')
        self.vector_dim = all_embeddings.shape[1]

        # 创建FAISS索引
        print("创建FAISS索引...")
        self.index = faiss.IndexFlatIP(self.vector_dim)  # 使用内积相似度
        # 或者使用 IndexFlatL2 用于欧氏距离

        # 归一化向量以便使用内积计算余弦相似度
        faiss.normalize_L2(all_embeddings)
        self.index.add(all_embeddings)

        print(f"FAISS索引构建完成，包含 {self.index.ntotal} 个向量")

        # 保存索引
        self._save_faiss_index()

    def search_cbdb_semantic(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        使用FAISS进行语义搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        if self.index is None or len(self.documents) == 0:
            print("FAISS索引未初始化，请先调用 setup_vector_database()")
            return []

        # 生成查询嵌入
        query_embedding = self.embedding.encode([query])
        query_embedding = query_embedding.astype('float32')

        # 归一化查询向量
        faiss.normalize_L2(query_embedding)

        # 搜索
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # 确保索引有效
                results.append({
                    "document": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "score": float(distances[0][i])  # 相似度分数
                })

        return results
