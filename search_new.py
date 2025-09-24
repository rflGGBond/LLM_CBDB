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
        初始化CBDB RAG系统（仅使用通义千问联网功能）
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
        self.qwen_api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

        # 初始化kin_code映射
        self._init_kin_code_mapping()

        # 加载或创建FAISS索引
        self.faiss_index_path = "./faiss_index"
        self._load_or_create_faiss_index()

        # 预加载数据库结构信息
        self._load_database_schema()

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

    def call_qwen_api(self, prompt: str, enable_search: bool = True, max_tokens: int = 2000) -> str:
        """
        调用通义千问API，支持联网搜索功能

        Args:
            prompt: 提示词
            enable_search: 是否启用联网搜索
            max_tokens: 最大生成长度
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.qwen_api_key}',
                'Content-Type': 'application/json',
                'X-DashScope-SSE': 'enable'  # 启用流式输出（可选）
            }

            # 构建请求数据
            data = {
                "model": "qwen-plus",  # 或 qwen-max，这些版本支持联网搜索
                "input": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一个专业的历史研究助手。请根据用户提供的信息和联网搜索功能，给出准确、全面的历史分析。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                "parameters": {
                    "result_format": "message",
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    # 联网搜索相关参数（根据通义千问API文档调整）
                    "enable_search": enable_search,
                    "incremental_output": True
                }
            }

            response = requests.post(
                self.qwen_api_url,
                headers=headers,
                json=data,
                timeout=60  # 增加超时时间，因为联网搜索可能需要更长时间
            )

            print(f"API响应状态码: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"API响应结构: {list(result.keys())}")

                # 调试：打印完整的响应结构
                if 'output' in result:
                    print(f"Output结构: {list(result['output'].keys())}")
                    if 'choices' in result['output']:
                        return result['output']['choices'][0]['message']['content']
                    else:
                        # 尝试其他可能的结构
                        if 'message' in result['output']:
                            return result['output']['message']['content']
                        else:
                            return f"响应结构异常: {result['output']}"
                else:
                    return f"响应缺少output字段: {result}"

            else:
                error_msg = f"API请求失败: {response.status_code} - {response.text}"
                print(error_msg)
                return error_msg

        except Exception as e:
            error_msg = f"通义千问API调用异常: {str(e)}"
            print(error_msg)
            return error_msg

    def generate_qwen_answer(self, query: str, db_context: List[Dict], enable_search: bool = True) -> str:
        """
        使用通义千问生成回答，可选择是否启用联网搜索

        Args:
            query: 用户问题
            db_context: 数据库检索结果
            enable_search: 是否启用通义千问的联网搜索
        """
        # 格式化数据库上下文
        db_context_str = self._format_db_context(db_context)

        # 构建智能提示词
        prompt = self._build_qwen_prompt(query, db_context_str, enable_search)

        print(f"启用联网搜索: {enable_search}")
        print(f"提示词长度: {len(prompt)} 字符")

        # 调用通义千问API
        answer = self.call_qwen_api(prompt, enable_search=enable_search)

        return answer

    def _format_db_context(self, db_context: List[Dict]) -> str:
        """格式化数据库上下文"""
        if not db_context:
            return "在CBDB数据库中未找到相关信息。"

        context_str = "【CBDB数据库检索结果】\n"
        for i, item in enumerate(db_context):
            context_str += f"\n第{i + 1}条记录：\n"

            if 'metadata' in item:
                meta = item['metadata']
                context_str += f"姓名：{meta.get('name', '未知')}\n"
                if meta.get('birth_year') or meta.get('death_year'):
                    context_str += f"生卒年：{meta.get('birth_year', '?')}-{meta.get('death_year', '?')}\n"
                if meta.get('birthplace'):
                    context_str += f"籍贯：{meta.get('birthplace')}\n"
                if meta.get('biography'):
                    # 限制传记长度，避免提示词过长
                    bio = meta.get('biography', '')
                    context_str += f"生平：{bio[:300]}{'...' if len(bio) > 300 else ''}\n"
            else:
                context_str += f"姓名：{item.get('name', '未知')}\n"
                if item.get('birth_year'):
                    context_str += f"生卒年：{item.get('birth_year')}-{item.get('death_year', '?')}\n"
                if item.get('biography'):
                    bio = item.get('biography', '')
                    context_str += f"生平：{bio[:300]}{'...' if len(bio) > 300 else ''}\n"

            context_str += f"相关度：{item.get('score', 1.0):.3f}\n"

        return context_str

    def _build_qwen_prompt(self, query: str, db_context: str, enable_search: bool) -> str:
        """构建通义千问提示词"""

        if enable_search:
            search_instruction = """
请结合以下CBDB数据库的权威历史信息，并利用联网搜索功能获取最新研究成果和补充资料，给出全面、准确的分析。

注意：优先以CBDB数据库信息为准，网络信息作为补充和验证。"""
        else:
            search_instruction = """
请基于以下CBDB数据库信息进行分析，不使用外部网络信息。"""

        prompt = f"""{search_instruction}

{db_context}

用户问题：{query}

请按照以下要求回答：
1. 首先总结CBDB数据库中的核心信息
2. {'结合联网搜索的最新信息进行深入分析' if enable_search else '基于数据库信息进行分析'}
3. 保持客观、专业的学术风格
4. 注明信息主要来源
5. 如有不确定的信息请明确说明

请用中文回答："""

        return prompt

    def smart_query(self, question: str, search_type: str = "hybrid",
                    top_k: int = 3, auto_enable_search: bool = True) -> str:
        """
        智能查询：自动判断是否需要联网搜索

        Args:
            question: 用户问题
            search_type: 数据库搜索类型
            top_k: 返回结果数量
            auto_enable_search: 是否自动判断是否启用搜索
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

        # 2. 智能判断是否需要联网搜索
        if auto_enable_search:
            enable_search = self._should_enable_search(question, db_context)
        else:
            enable_search = auto_enable_search

        print(f"智能判断启用联网搜索: {enable_search}")

        # 3. 使用通义千问生成回答
        print("正在使用通义千问生成回答...")
        answer = self.generate_qwen_answer(question, db_context, enable_search)

        return answer

    def _should_enable_search(self, question: str, db_context: List[Dict]) -> bool:
        """
        智能判断是否应该启用联网搜索
        """
        # 这些问题通常需要最新信息
        search_keywords = [
            '最新研究', '现代观点', '当前评价', '近年发现',
            '争议', '讨论', '研究进展', '学术动态',
            '图片', '地图', '地理位置', '现代地名'
        ]

        # 这些问题通常数据库信息就足够了
        local_keywords = [
            '生平', '生卒年', '籍贯', '亲属', '家族',
            '官职', '作品', '著作', '基本介绍', '简单介绍'
        ]

        has_search_needs = any(keyword in question for keyword in search_keywords)
        has_local_sufficiency = any(keyword in question for keyword in local_keywords)

        # 如果问题明确需要最新信息，或者数据库信息不足，则启用搜索
        if has_search_needs:
            return True
        elif not db_context and not has_local_sufficiency:
            return True  # 数据库没找到，尝试搜索
        else:
            return False  # 数据库信息足够，不启用搜索

    def manual_query(self, question: str, enable_search: bool = True,
                     search_type: str = "hybrid", top_k: int = 3) -> str:
        """
        手动控制查询：明确指定是否启用搜索

        Args:
            question: 用户问题
            enable_search: 是否启用联网搜索
            search_type: 数据库搜索类型
            top_k: 返回结果数量
        """
        print(f"手动模式查询: {question}")
        print(f"联网搜索: {'启用' if enable_search else '禁用'}")

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

        answer = self.generate_qwen_answer(question, db_context, enable_search)
        return answer


# 使用示例
def main():
    # 初始化系统
    rag_system = CBDBRAGSystem(DATABASE_NAME)

    # 检查是否需要构建向量数据库
    if not os.path.exists(f"{rag_system.faiss_index_path}.index"):
        print("首次运行，正在构建向量数据库...")
        rag_system.setup_vector_database(limit=1000)

    # 测试查询示例
    test_cases = [
        # (问题, 是否启用搜索, 描述)
        ("苏轼的父亲", False, "基础信息-不需要搜索"),
        ("李清照的丈夫", True, "最新研究-需要搜索"),
        ("王安石变法的现代评价", True, "现代观点-需要搜索")
    ]

    for question, enable_search, description in test_cases:
        print(f"\n{'=' * 60}")
        print(f"测试: {description}")
        print(f"问题: {question}")
        print(f"{'=' * 60}")

        try:
            # 使用手动模式明确控制
            answer = rag_system.manual_query(
                question=question,
                enable_search=enable_search,
                search_type="hybrid",
                top_k=3
            )

            print(f"回答：\n{answer}")

        except Exception as e:
            print(f"查询失败: {e}")

    # 测试智能模式
    print(f"\n{'=' * 60}")
    print("测试智能模式")
    print(f"{'=' * 60}")

    smart_questions = [
        "李白的出生地",
        "杜甫的生卒年"
    ]

    for question in smart_questions:
        print(f"\n智能查询: {question}")
        try:
            answer = rag_system.smart_query(question, auto_enable_search=True)
            print(f"回答：\n{answer[:500]}...")  # 限制输出长度
        except Exception as e:
            print(f"智能查询失败: {e}")


if __name__ == "__main__":
    main()