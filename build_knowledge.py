import os
import time
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
import chromadb

load_dotenv()

# 配置
embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH")
embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")
input_dir = os.getenv("INPUT_DIR", "./docs")
chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
collection_name = "finance_products"
persist_dir = os.getenv("PERSIST_DIR", "./storage")

# 初始化 Embedding 模型
print(f"【INFO】加载 Embedding 模型: {embedding_model_path}")
Settings.embed_model = HuggingFaceEmbedding(
    model_name=embedding_model_path,
    device=embedding_device,
)

# 加载文档
print(f"【INFO】从 {input_dir} 加载 PDF 文档...")
documents = SimpleDirectoryReader(
    input_dir=input_dir,
    required_exts=[".pdf"],
    recursive=True,
).load_data()

if not documents:
    raise RuntimeError(f"未在 {input_dir} 中找到任何 PDF 文件！")

print(f"【INFO】成功加载 {len(documents)} 个文档")

# 分块
node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
print(f"【INFO】生成 {len(nodes)} 个文本块")

# 初始化 Chroma
client = chromadb.PersistentClient(path=chroma_db_path)
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)
vector_store = ChromaVectorStore(chroma_collection=collection)

# 构建索引
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    show_progress=True,
)
index.set_index_id("finance_index")

# 持久化
storage_context.persist(persist_dir=persist_dir)
time.sleep(2)

print(f"【INFO】知识库构建完成！")
print(f"   - 向量数: {collection.count()}")
print(f"   - 存储路径: {chroma_db_path}")


