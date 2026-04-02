from fastapi import FastAPI, Query
from hybrid_query_engine import ask

app = FastAPI(title="金融RAG问答API")

@app.get("/ask")
def ask_api(query: str = Query(..., description="您的问题")):
    answer = ask(query)
    return {"query": query, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)



