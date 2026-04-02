# test_query.py
from hybrid_query_engine import ask

if __name__ == "__main__":
    # 测试问题列表
    questions = [
        "理财产品的收益说明？",
        "基金的管理人是谁？",
        "产品的风险等级是什么？",
    ]
    
    for q in questions:
        print("\n" + "="*60)
        print(f"问题: {q}")
        print("-"*60)
        try:
            answer = ask(q)
            print(f"回答: {answer}")
        except Exception as e:
            print(f"错误: {e}")
        print("="*60)


