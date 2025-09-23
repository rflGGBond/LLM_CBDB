import sqlite3

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DATABASE_NAME = "latest.db"


def quick_test():
    """快速测试数据库连接和基本功能"""

    # 测试数据库连接
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        # 测试表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='biog_main'")
        if cursor.fetchone():
            print("✓ biog_main表存在")

        # 测试数据量
        cursor.execute("SELECT COUNT(*) FROM biog_main")
        count = cursor.fetchone()[0]
        print(f"✓ biog_main表有 {count} 条记录")

        # 测试kin_data表
        cursor.execute("SELECT COUNT(*) FROM kin_data")
        kin_count = cursor.fetchone()[0]
        print(f"✓ kin_data表有 {kin_count} 条记录")

        conn.close()

        # 测试嵌入模型
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embeddings = model.encode(["测试文本"])
        print(f"✓ 嵌入模型正常工作，向量维度: {embeddings.shape[1]}")

        print("\n所有基础功能测试通过！")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    quick_test()