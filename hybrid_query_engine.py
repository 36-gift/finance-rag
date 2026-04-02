import os
import re
import jieba
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate

load_dotenv()

custom_prompt = PromptTemplate(
    "你是一个专业的金融客服助手。请基于以下参考资料，详细、专业地回答用户问题。\n"
    "如果资料中有具体数字、百分比、风险提示，请完整引用。\n"
    "如果问题涉及收益，请同时说明收益计算方式和风险。\n\n"
    "参考资料：\n{context_str}\n\n"
    "用户问题：{query_str}\n\n"
    "专业回答："
)

# 配置
embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH")
llm_model_path = os.getenv("LLM_MODEL_PATH")
chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
persist_dir = os.getenv("PERSIST_DIR", "./storage")
input_dir = os.getenv("INPUT_DIR", "./docs")
collection_name = "finance_products"

# 自定义 BM25 检索器
class BM25Retriever(BaseRetriever):
    def __init__(self, nodes, top_k=5):
        self.nodes = nodes
        self.top_k = top_k
        self._build_bm25()
        super().__init__()

    def _tokenize(self, text):
        words = jieba.cut(text.strip())
        return [w for w in words if len(w.strip()) > 0]

    def _build_bm25(self):
        corpus = [self._tokenize(n.get_content()) for n in self.nodes]
        self.bm25 = BM25Okapi(corpus)

    def _retrieve(self, query_bundle):
        tokenized = self._tokenize(query_bundle.query_str)
        scores = self.bm25.get_scores(tokenized)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]
        return [NodeWithScore(node=self.nodes[i], score=float(scores[i])) for i in top_indices]

# 混合检索器
class HybridRetriever(BaseRetriever):
    def __init__(self, bm25_retriever, vector_retriever, top_k=5):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.top_k = top_k

    def _retrieve(self, query_bundle):
        bm25_nodes = self.bm25.retrieve(query_bundle)
        vector_nodes = self.vector.retrieve(query_bundle)
        seen = set()
        merged = []
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in seen:
                seen.add(n.node.node_id)
                merged.append(n)
        merged.sort(key=lambda x: x.score or 0, reverse=True)
        return merged[:self.top_k]

# 全局引擎
_query_engine = None

def get_engine():
    global _query_engine
    if _query_engine:
        return _query_engine

    print("【INFO】初始化 Embedding 模型...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_path, device="cpu")

    print("【INFO】初始化 LLM...")
    Settings.llm = HuggingFaceLLM(
        model_name=llm_model_path,
        tokenizer_name=llm_model_path,
        max_new_tokens=256,
        model_kwargs={"trust_remote_code": True},
    )

    print("【INFO】加载文档用于 BM25...")
    docs = SimpleDirectoryReader(input_dir, required_exts=[".pdf"]).load_data()
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(docs)

    print("【INFO】连接 Chroma 向量库...")
    client = chromadb.PersistentClient(path=chroma_db_path)
    collection = client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=Settings.embed_model)

    bm25_ret = BM25Retriever(nodes, top_k=5)
    vector_ret = VectorIndexRetriever(index, similarity_top_k=5)
    hybrid_ret = HybridRetriever(bm25_ret, vector_ret)

    _query_engine = RetrieverQueryEngine(retriever=hybrid_ret)
    _query_engine.update_prompts({"response_synthesizer:text_qa_template": custom_prompt})
    return _query_engine

def ask(query: str) -> str:
    engine = get_engine()
    response = engine.query(query)
    return str(response)


