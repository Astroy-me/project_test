from sentence_transformers import CrossEncoder
class My_Reranker:
    def __init__(self, 
                reranker_path = "/checkpoint/BAAI/bge-reranker-v2-m3", 
                device = "cuda:3"
                ):
        self.reranker = CrossEncoder(model_name=reranker_path,device=device)

    def rerank_documents(self, query : str, search_doc, top_k : int):
        documents = [chunk["text"] for chunk in search_doc ]
        filenames = [chunk["filename"] for chunk in search_doc ]
        reranker_score = self.reranker.predict([(query,doc) for doc in documents])
        reranked_docs = sorted(
            zip(documents, filenames, reranker_score),
            key = lambda x:x[2],
            reverse = True
            )
        
        return reranked_docs[:top_k]
    
    def document2str(self,search_doc):
        docs =""
        for chunk in search_doc:
            docs += "文档:" + chunk[0] + "来自" + chunk[1] + "\n"
        return docs
