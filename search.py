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

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DATABASE_NAME = "latest.db"
API_KEY_QWEN = "sk-3ebfc5913e51470b86b252730364ba16"


class CBDBRAGSystem:
    def __init__(self, db_path: str, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        """
        初始化CBDB RAG系统（使用FAISS）
        """
        self.db_conn = sqlite3.connect(db_path)
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # 使用本地生成回答，避免API超时
        self.use_api = False  # 设置为False使用本地生成

        if self.use_api:
            self.client = OpenAI(api_key=API_KEY_QWEN)

        # FAISS相关变量
        self.index = None
        self.documents = []
        self.metadata = []
        self.vector_dim = 384

        # 初始化kin_code映射
        self._init_kin_code_mapping()

        # 加载或创建FAISS索引
        self.faiss_index_path = "./faiss_index"
        self._load_or_create_faiss_index()

        # 预加载数据库结构信息
        self._load_database_schema()

    def _init_kin_code_mapping(self):
        """
        初始化kin_code映射关系
        """
        self.kin_code_mapping = {
            1: "父亲", 2: "母亲", 3: "儿子", 4: "女儿", 5: "兄弟",
            6: "姐妹", 7: "丈夫", 8: "妻子", 9: "祖父", 10: "祖母",
            11: "孙子", 12: "孙女", 13: "叔父", 14: "伯父", 15: "舅父",
            16: "岳父", 17: "女婿", 18: "侄子", 19: "外甥", 20: "堂兄弟",
        }

    def _load_database_schema(self):
        """加载数据库表结构信息"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        self.tables = [table[0] for table in cursor.fetchall()]
        print(f"发现表: {self.tables}")

    def _load_or_create_faiss_index(self):
        """加载或创建FAISS索引"""
        if os.path.exists(f"{self.faiss_index_path}.index"):
            print("加载已有的FAISS索引...")
            self.index = faiss.read_index(f"{self.faiss_index_path}.index")
            with open(f"{self.faiss_index_path}_meta.pkl", "rb") as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
            print(f"索引加载完成，包含 {len(self.documents)} 个文档")
        else:
            print("未找到现有索引，需要重新构建")
            self.index = None

    def _save_faiss_index(self):
        """保存FAISS索引和元数据"""
        if self.index is not None:
            os.makedirs(os.path.dirname(self.faiss_index_path) or ".", exist_ok=True)
            faiss.write_index(self.index, f"{self.faiss_index_path}.index")

            with open(f"{self.faiss_index_path}_meta.pkl", "wb") as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata
                }, f)
            print("FAISS索引和元数据已保存")

    def setup_vector_database(self, batch_size: int = 100, limit: int = None):
        """
        将CBDB数据向量化并存储到FAISS中
        """
        print("开始构建FAISS向量数据库...")

        # 根据实际的BIOG_MAIN表结构修改查询
        query = """
        SELECT 
            p.c_personid, 
            p.c_name_chn, 
            p.c_index_year, 
            p.c_deathyear, 
            p.c_birthyear,
            p.c_notes, 
            b.c_name_chn as birthplace_name
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
            # 构建文档文本 - 使用正确的列名
            doc_text = f"人物: {row['c_name_chn']}"

            # 使用正确的生卒年列名
            birth_year = row['c_birthyear'] if pd.notna(row['c_birthyear']) else row['c_index_year']
            death_year = row['c_deathyear'] if pd.notna(row['c_deathyear']) else None

            if pd.notna(birth_year):
                doc_text += f", 生卒年: {int(birth_year) if not pd.isna(birth_year) else '?'}-{int(death_year) if pd.notna(death_year) else '?'}"

            if pd.notna(row['birthplace_name']):
                doc_text += f", 籍贯: {row['birthplace_name']}"

            if pd.notna(row['c_notes']):
                doc_text += f", 生平: {row['c_notes'][:500]}"

            self.documents.append(doc_text)
            self.metadata.append({
                "person_id": row['c_personid'],
                "name": row['c_name_chn'],
                "birth_year": birth_year,
                "death_year": death_year,
                "birthplace": row['birthplace_name'],
                "biography": row['c_notes'][:1000] if pd.notna(row['c_notes']) else ""
            })

        # 批量生成嵌入
        print("生成文本嵌入...")
        for i in range(0, len(self.documents), batch_size):
            batch_docs = self.documents[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch_docs)
            all_embeddings.append(batch_embeddings)
            print(f"已处理 {min(i + batch_size, len(self.documents))}/{len(self.documents)} 条记录")

        # 合并所有嵌入
        all_embeddings = np.vstack(all_embeddings).astype('float32')
        self.vector_dim = all_embeddings.shape[1]

        # 创建FAISS索引
        print("创建FAISS索引...")
        self.index = faiss.IndexFlatIP(self.vector_dim)
        faiss.normalize_L2(all_embeddings)
        self.index.add(all_embeddings)

        print(f"FAISS索引构建完成，包含 {self.index.ntotal} 个向量")
        self._save_faiss_index()

    def search_cbdb_semantic(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        使用FAISS进行语义搜索
        """
        if self.index is None or len(self.documents) == 0:
            print("FAISS索引未初始化，请先调用 setup_vector_database()")
            return []

        query_embedding = self.embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "score": float(distances[0][i])
                })

        return results

    def search_cbdb_keyword(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        关键词搜索CBDB数据库 - 使用正确的列名
        """
        cursor = self.db_conn.cursor()

        name_keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', query)

        if name_keywords:
            name_conditions = " OR ".join(["p.c_name_chn LIKE ?" for _ in name_keywords])
            sql = f"""
            SELECT 
                p.c_personid, 
                p.c_name_chn, 
                p.c_index_year, 
                p.c_deathyear,
                p.c_birthyear,
                p.c_index_addr_id, 
                p.c_notes, 
                b.c_name_chn as birthplace_name
            FROM biog_main p
            LEFT JOIN addresses b ON p.c_index_addr_id = b.c_addr_id
            WHERE ({name_conditions}) OR p.c_notes LIKE ?
            LIMIT ?
            """

            params = [f"%{name}%" for name in name_keywords] + [f"%{query}%", top_k]
            cursor.execute(sql, params)
        else:
            sql = """
            SELECT 
                p.c_personid, 
                p.c_name_chn, 
                p.c_index_year, 
                p.c_deathyear,
                p.c_birthyear,
                p.c_index_addr_id, 
                p.c_notes, 
                b.c_name_chn as birthplace_name
            FROM biog_main p
            LEFT JOIN addresses b ON p.c_index_addr_id = b.c_addr_id
            WHERE p.c_notes LIKE ? OR p.c_name_chn LIKE ?
            LIMIT ?
            """
            cursor.execute(sql, (f"%{query}%", f"%{query}%", top_k))

        results = []
        for row in cursor.fetchall():
            birth_year = row[4] if pd.notna(row[4]) else row[2]  # 优先使用c_birthyear
            results.append({
                "person_id": row[0],
                "name": row[1],
                "index_year": row[2],
                "death_year": row[3],
                "birth_year": birth_year,
                "birthplace_id": row[5],
                "biography": row[6],
                "birthplace_name": row[7],
                "score": 1.0
            })

        return results

    def get_person_relationships(self, person_id: int) -> List[Dict]:
        """
        获取人物的亲属关系
        """
        cursor = self.db_conn.cursor()

        cursor.execute("""
        SELECT r.c_kin_id, r.c_kin_code, r.c_notes, p.c_name_chn
        FROM kin_data r 
        JOIN biog_main p ON r.c_kin_id = p.c_personid 
        WHERE r.c_personid = ?
        """, (person_id,))

        relationships = []
        for row in cursor.fetchall():
            kin_id, kin_code, notes, kin_name = row
            relation_type = self.kin_code_mapping.get(kin_code, f"未知关系(代码:{kin_code})")

            relationships.append({
                "kin_id": kin_id,
                "kin_code": kin_code,
                "relation_type": relation_type,
                "kin_name": kin_name,
                "notes": notes
            })

        return relationships

    def get_person_details(self, person_id: int) -> Dict:
        """
        获取人物的详细信息 - 使用正确的列名
        """
        cursor = self.db_conn.cursor()

        # 使用正确的BIOG_MAIN表列名
        cursor.execute("""
        SELECT 
            p.c_personid, 
            p.c_name_chn, 
            p.c_index_year, 
            p.c_deathyear,
            p.c_birthyear,
            p.c_index_addr_id, 
            p.c_notes, 
            b.c_name_chn as birthplace_name,
            p.c_female,
            p.c_ethnicity_code,
            p.c_household_status_code
        FROM biog_main p
        LEFT JOIN addresses b ON p.c_index_addr_id = b.c_addr_id
        WHERE p.c_personid = ?
        """, (person_id,))

        row = cursor.fetchone()
        if not row:
            return None

        person_info = {
            "person_id": row[0],
            "name": row[1],
            "index_year": row[2],
            "death_year": row[3],
            "birth_year": row[4],
            "birthplace_id": row[5],
            "biography": row[6],
            "birthplace_name": row[7],
            "female": bool(row[8]) if pd.notna(row[8]) else None,
            "ethnicity": row[9],
            "household_status": row[10]
        }

        # 获取亲属关系
        relationships = self.get_person_relationships(person_id)
        person_info["relationships"] = relationships

        return person_info

    def generate_answer(self, query: str, context: List[Dict]) -> str:
        """
        生成回答 - 使用本地生成避免API超时
        """
        if not context:
            return "在CBDB数据库中未找到相关信息。"

        # 构建详细的回答
        answer = f"根据CBDB数据库检索，关于『{query}』找到以下信息：\n\n"

        for i, item in enumerate(context):
            answer += f"【结果 {i + 1}】\n"

            if 'metadata' in item:
                meta = item['metadata']
                answer += f"姓名: {meta.get('name', '未知')}\n"

                birth_year = meta.get('birth_year')
                death_year = meta.get('death_year')
                if birth_year or death_year:
                    answer += f"生卒年: {birth_year or '?'}-{death_year or '?'}\n"

                if meta.get('birthplace'):
                    answer += f"籍贯: {meta.get('birthplace')}\n"

                if meta.get('biography'):
                    bio = meta.get('biography', '')
                    answer += f"生平: {bio[:200]}{'...' if len(bio) > 200 else ''}\n"

            elif 'name' in item:
                answer += f"姓名: {item['name']}\n"
                if item.get('birth_year'):
                    answer += f"生卒年: {item.get('birth_year')}-{item.get('death_year', '?')}\n"
                if item.get('biography'):
                    bio = item.get('biography', '')
                    answer += f"生平: {bio[:200]}{'...' if len(bio) > 200 else ''}\n"

            answer += f"相似度: {item.get('score', 1.0):.3f}\n\n"

        answer += f"共找到 {len(context)} 条相关记录。"
        return answer

    def query(self, question: str, search_type: str = "hybrid", top_k: int = 5) -> str:
        """
        执行查询
        """
        print(f"正在处理问题: {question}")

        if search_type == "semantic":
            context = self.search_cbdb_semantic(question, top_k)
        elif search_type == "keyword":
            context = self.search_cbdb_keyword(question, top_k)
        else:  # hybrid
            semantic_results = self.search_cbdb_semantic(question, top_k)
            keyword_results = self.search_cbdb_keyword(question, top_k)

            context = []
            seen_ids = set()

            for result in keyword_results:
                if result['person_id'] not in seen_ids:
                    context.append(result)
                    seen_ids.add(result['person_id'])

            for result in semantic_results:
                person_id = result['metadata']['person_id']
                if person_id not in seen_ids:
                    context.append(result)
                    seen_ids.add(person_id)

            context.sort(key=lambda x: x.get('score', 0), reverse=True)
            context = context[:top_k]

        if not context:
            return "抱歉，在CBDB数据库中未找到相关信息。"

        answer = self.generate_answer(question, context)
        return answer

    def get_person_by_name(self, name: str) -> List[Dict]:
        """
        根据姓名精确查找人物
        """
        cursor = self.db_conn.cursor()

        cursor.execute("""
        SELECT 
            c_personid, 
            c_name_chn, 
            c_index_year, 
            c_deathyear,
            c_birthyear,
            c_notes
        FROM biog_main 
        WHERE c_name_chn = ? OR c_name_chn LIKE ?
        LIMIT 10
        """, (name, f"%{name}%"))

        results = []
        for row in cursor.fetchall():
            results.append({
                "person_id": row[0],
                "name": row[1],
                "index_year": row[2],
                "death_year": row[3],
                "birth_year": row[4],
                "biography": row[5]
            })

        return results


# 使用示例
def main():
    # 初始化系统
    rag_system = CBDBRAGSystem(DATABASE_NAME)

    # 检查是否需要构建向量数据库
    if not os.path.exists(f"{rag_system.faiss_index_path}.index"):
        print("首次运行，正在构建向量数据库...")
        rag_system.setup_vector_database(limit=1000)  # 先处理1000条测试

    # 测试查询
    test_questions = [
        "苏轼",
        "王安石",
        "李白",
        "杜甫",
        "李清照的丈夫是谁"
    ]

    for question in test_questions:
        print(f"\n{'=' * 60}")
        print(f"问题: {question}")
        print(f"{'=' * 60}")

        # 先测试精确查询
        exact_results = rag_system.get_person_by_name(question)
        if exact_results:
            print(f"精确找到 {len(exact_results)} 个同名人物")
            for result in exact_results[:3]:  # 显示前3个
                print(f"  - {result['name']} (ID: {result['person_id']})")

        # 生成回答
        answer = rag_system.query(question, search_type="hybrid", top_k=3)
        print(f"回答: {answer}")

    # 测试详细信息查询
    print(f"\n{'=' * 60}")
    print("测试详细信息查询")
    print(f"{'=' * 60}")

    # 尝试查询第一个找到的人物的详细信息
    test_results = rag_system.get_person_by_name("苏轼")
    if test_results:
        person_id = test_results[0]['person_id']
        details = rag_system.get_person_details(person_id)
        if details:
            print(f"人物详情: {details['name']}")
            print(f"生卒年: {details.get('birth_year', '?')}-{details.get('death_year', '?')}")
            print(f"籍贯: {details.get('birthplace_name', '未知')}")
            if details.get('relationships'):
                print(f"亲属关系: {len(details['relationships'])} 条")
                for rel in details['relationships'][:5]:  # 显示前5个关系
                    print(f"  - {rel['relation_type']}: {rel['kin_name']}")


if __name__ == "__main__":
    main()