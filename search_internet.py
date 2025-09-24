import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from typing import List, Dict, Any
import re
import json
import requests
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DATABASE_NAME = "latest.db"
API_KEY_QWEN = "sk-3ebfc5913e51470b86b252730364ba16"


class CBDBRAGSystem:
    def __init__(self, db_path: str, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        """
        初始化CBDB RAG系统（使用通义千问模型）
        """
        self.db_conn = sqlite3.connect(db_path)
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # FAISS相关变量
        self.index = None
        self.documents = []
        self.metadata = []
        self.vector_dim = 384

        # 通义千问API配置
        self.qwen_api_key = API_KEY_QWEN  # 请替换为您的API密钥
        self.qwen_api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/text-generation/generation"

        # 初始化kin_code映射
        self._init_kin_code_mapping()

        # 加载或创建FAISS索引
        self.faiss_index_path = "./faiss_index"
        self._load_or_create_faiss_index()

        # 预加载数据库结构信息
        self._load_database_schema()

        # 联网搜索配置
        self.enable_web_search = True
        self.search_timeout = 10

    def _init_kin_code_mapping(self):
        """初始化kin_code映射关系"""
        self.kin_code_mapping = {
            1: "父亲", 2: "母亲", 3: "儿子", 4: "女儿", 5: "兄弟",
            6: "姐妹", 7: "丈夫", 8: "妻子", 9: "祖父", 10: "祖母",
            11: "孙子", 12: "孙女", 13: "叔父", 14: "伯父", 15: "舅父",
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

    def setup_vector_database(self, batch_size: int = 100, limit: int = None):
        """
        将CBDB数据向量化并存储到FAISS中
        """
        print("开始构建FAISS向量数据库...")

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

        for _, row in df.iterrows():
            doc_text = f"人物: {row['c_name_chn']}"

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

        print("生成文本嵌入...")
        for i in range(0, len(self.documents), batch_size):
            batch_docs = self.documents[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch_docs)
            all_embeddings.append(batch_embeddings)
            print(f"已处理 {min(i + batch_size, len(self.documents))}/{len(self.documents)} 条记录")

        all_embeddings = np.vstack(all_embeddings).astype('float32')
        self.vector_dim = all_embeddings.shape[1]

        print("创建FAISS索引...")
        self.index = faiss.IndexFlatIP(self.vector_dim)
        faiss.normalize_L2(all_embeddings)
        self.index.add(all_embeddings)

        print(f"FAISS索引构建完成，包含 {self.index.ntotal} 个向量")
        self._save_faiss_index()

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

    def search_cbdb_semantic(self, query: str, top_k: int = 5) -> List[Dict]:
        """使用FAISS进行语义搜索"""
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
        """关键词搜索CBDB数据库"""
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
            birth_year = row[4] if pd.notna(row[4]) else row[2]
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

    def web_search(self, query: str, max_results: int = 3) -> List[Dict]:
        """联网搜索获取额外信息"""
        if not self.enable_web_search:
            return []

        try:
            # 使用DuckDuckGo搜索
            return self._duckduckgo_search(query, max_results)
        except Exception as e:
            print(f"联网搜索失败: {e}")
            return []

    def _duckduckgo_search(self, query: str, max_results: int) -> List[Dict]:
        """使用DuckDuckGo API进行搜索"""
        try:
            from ddgs import DDGS

            ddgs = DDGS()
            results = ddgs.text(query, max_results=max_results)

            formatted_results = []
            for result in results:
                formatted_results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', '')[:200] + '...',
                    'source': '网络搜索',
                    'timestamp': datetime.now().isoformat()
                })

            return formatted_results[:max_results]

        except ImportError:
            print("请安装duckduckgo-search: pip install duckduckgo-search")
            return []
        except Exception as e:
            print(f"DuckDuckGo搜索失败: {e}")
            return []

    def call_qwen_api(self, prompt: str, max_tokens: int = 1000) -> str:
        """调用通义千问API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.qwen_api_key}',
                'Content-Type': 'application/json'
            }

            data = {
                "model": "qwen3-32b",  # 或 qwen-plus, qwen-max
                "input": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一个专业的历史研究助手，擅长分析中国历史人物和事件。请根据提供的信息给出准确、专业的回答。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                "parameters": {
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    "enable_search":  True
                }
            }

            response = requests.post(
                self.qwen_api_url,
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result['output']['choices'][0]['message']['content']
            else:
                return f"API请求失败: {response.status_code} - {response.text}"

        except Exception as e:
            return f"通义千问API调用失败: {str(e)}"

    def generate_comprehensive_answer(self, query: str, db_context: List[Dict], web_context: List[Dict]) -> str:
        """使用通义千问生成综合回答"""

        # 构建提示词
        prompt = self._build_qwen_prompt(query, db_context, web_context)

        # 调用通义千问API
        answer = self.call_qwen_api(prompt)

        return answer

    def _build_qwen_prompt(self, query: str, db_context: List[Dict], web_context: List[Dict]) -> str:
        """构建通义千问的提示词"""

        # 格式化数据库上下文
        db_context_str = "【CBDB数据库检索结果】\n"
        if db_context:
            for i, item in enumerate(db_context):
                db_context_str += f"\n第{i + 1}条记录：\n"
                if 'metadata' in item:
                    meta = item['metadata']
                    db_context_str += f"姓名：{meta.get('name', '未知')}\n"
                    if meta.get('birth_year') or meta.get('death_year'):
                        db_context_str += f"生卒年：{meta.get('birth_year', '?')}-{meta.get('death_year', '?')}\n"
                    if meta.get('birthplace'):
                        db_context_str += f"籍贯：{meta.get('birthplace')}\n"
                    if meta.get('biography'):
                        db_context_str += f"生平：{meta.get('biography')}\n"
                db_context_str += f"相似度：{item.get('score', 1.0):.3f}\n"
        else:
            db_context_str += "在CBDB数据库中未找到相关信息。\n"

        # 格式化网络上下文
        web_context_str = "【网络补充信息】\n"
        if web_context:
            for i, item in enumerate(web_context):
                web_context_str += f"\n第{i + 1}条网络信息：\n"
                web_context_str += f"标题：{item.get('title', '无标题')}\n"
                web_context_str += f"摘要：{item.get('snippet', '无内容')}\n"
                web_context_str += f"来源：{item.get('source', '未知')}\n"
        else:
            web_context_str += "网络搜索未获得额外信息。\n"

        prompt = f"""你是一个专业的历史研究助手。请根据以下信息回答用户的问题。

{db_context_str}

{web_context_str}

用户问题：{query}

请根据以上信息，结合你的知识，给出专业、准确、全面的回答。回答要求：
1. 首先总结CBDB数据库中的核心信息
2. 然后结合网络信息进行补充和分析
3. 如果信息有冲突，以CBDB数据为准
4. 可以适当扩展相关的历史背景知识
5. 注明信息来源，区分数据库信息和网络信息

请用中文回答，保持专业性和可读性："""

        return prompt

    def query_with_web_search(self, question: str, search_type: str = "hybrid",
                              top_k: int = 5, enable_web: bool = True) -> str:
        """
        执行查询并结合联网搜索

        Args:
            question: 用户问题
            search_type: 搜索类型
            top_k: 返回结果数量
            enable_web: 是否启用联网搜索
        """
        print(f"正在处理问题: {question}")

        # 1. 从CBDB数据库检索信息
        if search_type == "semantic":
            db_context = self.search_cbdb_semantic(question, top_k)
        elif search_type == "keyword":
            db_context = self.search_cbdb_keyword(question, top_k)
        else:  # hybrid
            semantic_results = self.search_cbdb_semantic(question, top_k)
            keyword_results = self.search_cbdb_keyword(question, top_k)

            db_context = []
            seen_ids = set()

            for result in keyword_results:
                if result['person_id'] not in seen_ids:
                    db_context.append(result)
                    seen_ids.add(result['person_id'])

            for result in semantic_results:
                person_id = result['metadata']['person_id']
                if person_id not in seen_ids:
                    db_context.append(result)
                    seen_ids.add(person_id)

            db_context.sort(key=lambda x: x.get('score', 0), reverse=True)
            db_context = db_context[:top_k]

        print(f"从CBDB数据库找到 {len(db_context)} 条相关信息")

        # 2. 联网搜索获取补充信息
        web_context = []
        if enable_web and self.enable_web_search:
            print("正在进行联网搜索...")
            web_context = self.web_search(question, max_results=3)
            print(f"联网搜索找到 {len(web_context)} 条补充信息")

        # 3. 使用通义千问生成综合回答
        print("正在使用通义千问生成回答...")
        answer = self.generate_comprehensive_answer(question, db_context, web_context)

        return answer

    def get_search_summary(self, question: str, db_context: List[Dict], web_context: List[Dict]) -> str:
        """生成搜索摘要"""
        summary = f"搜索摘要：\n"
        summary += f"问题：{question}\n"
        summary += f"CBDB数据库结果：{len(db_context)} 条\n"
        summary += f"网络搜索结果：{len(web_context)} 条\n"

        if db_context:
            summary += "相关人物："
            names = []
            for item in db_context:
                if 'metadata' in item:
                    names.append(item['metadata'].get('name', '未知'))
                elif 'name' in item:
                    names.append(item['name'])
            summary += "，".join(set(names)) + "\n"

        return summary


# 使用示例
def main():
    # 初始化系统
    rag_system = CBDBRAGSystem(DATABASE_NAME)

    # 检查是否需要构建向量数据库
    if not os.path.exists(f"{rag_system.faiss_index_path}.index"):
        print("首次运行，正在构建向量数据库...")
        rag_system.setup_vector_database(limit=1000)

    # 测试查询
    test_questions = [
        "苏轼的父亲和兄弟都有谁",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 60}")
        print(f"查询 {i}: {question}")
        print(f"{'=' * 60}")

        try:
            # 使用联网搜索的综合查询
            answer = rag_system.query_with_web_search(
                question,
                search_type="hybrid",
                top_k=3,
                enable_web=True
            )

            print(f"回答：\n{answer}")

        except Exception as e:
            print(f"查询失败: {e}")
            #  fallback: 只使用数据库查询
            try:
                answer = rag_system.query_with_web_search(
                    question,
                    search_type="hybrid",
                    top_k=3,
                    enable_web=False
                )
                print(f"（仅数据库）回答：\n{answer}")
            except:
                print("所有查询方法都失败了")


if __name__ == "__main__":
    main()