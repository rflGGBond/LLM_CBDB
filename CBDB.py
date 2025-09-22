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
        # query = """
        # SELECT p.c_personid, p.c_name_chn, p.c_index_year, p.c_death_year,
        #        p.c_birthplace_id, p.c_notes_chn, b.c_name_chn as birthplace_name
        # FROM biog_main p
        # LEFT JOIN addresses b ON p.c_birthplace_id = b.c_addr_id
        # WHERE p.c_notes_chn IS NOT NULL AND LENGTH(p.c_notes_chn) > 10
        # """