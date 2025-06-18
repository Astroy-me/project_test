import os
import sys
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, Settings, load_index_from_storage, StorageContext,SimpleDirectoryReader,PromptTemplate,ServiceContext,GPTVectorStoreIndex
from document_loader import DocxParser, process_files
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.modelscope import ModelScopeLLM

from llama_index.core.selectors import LLMSingleSelector 
import torch
import faiss

class KnowledgeBase:
    def __init__(self):
        # 加载编码器
        self.embedding_model = HuggingFaceEmbedding(
            model_name="/checkpoint/BAAI/bge-large-zh-v1.5",
            device="cuda:3",
            normalize =True,
            embed_batch_size = 32,
            )
        SYSTEM_PROMPT = """你是一个中文知识库，请根据提供的内容回答问题。请尽量详细。"""
        query_wrapper_prompt = PromptTemplate(
                "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
            )
        llm_path = "/checkpoint/Qwen/Qwen3-8B"
        # 加载大语言模型
        self.llms = HuggingFaceLLM(
            context_window=2048,
            max_new_tokens=1024,
            generate_kwargs={
                "temperature": 0.7, 
                "top_k": 50, 
                "top_p": 0.95,
                "repetition_penalty": 1.2  # 关键参数：抑制重复
                },
            # query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=llm_path,
            model_name=llm_path,
            device_map="cuda:1",`
            # model_kwargs={"torch_dtype": torch.float16},
        )
        
        # ModelScopeLLM(
        #     model_name="checkpoint/openbmb/MiniCPM4-8B",
        #     task_name="text-generation",
        #     device_map="cuda:2",
        #     model_kwargs={"torch_dtype": torch.float16},
        # )
        # 环境设置
        Settings.llm = self.llms
        Settings.embed_model = self.embedding_model 
        self.vector_index = None
        self.query_engine = None
        self.chat = None
    def load_documents(self, file_paths: list[str]):
        # docs = process_files(
        #     file_paths=file_paths,
        #     image_output_root="ollamaindex/image_parser"
        # )
        # 读取文档
        docs = SimpleDirectoryReader(file_paths[0]).load_data()
        sentenceSplitter = SentenceSplitter(
            chunk_size=128,
            chunk_overlap=20,
            # separator="\n",
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^，。！？]+[，。！？]"  # 按句子分割
            
        )
        faiss_index = faiss.IndexFlatL2()
        vectorStore = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vectorStore)
        self.vector_index = GPTVectorStoreIndex.from_documents(docs, 
                                                               transformations=[sentenceSplitter],
                                                               storage_context=storage_context)
    
    def load_query_engine(self):
        if self.vector_index is None:
            print("请先加载知识库")
            return
        else:
            self.query_engine = self.vector_index.as_query_engine(similarity_top_k=2)
    
    def chat_query(self, query):
        if self.vector_index is None or self.query_engine is None:
            print("请先加载知识库")
            return
        else:
            print("处理中....")
            #response = self.query_engine.retrieve(query)  
            response = self.query_engine.query("阿司匹林的禁忌症有哪些？")
            return response
            
        
    def load_knowledge_base(self, store_path : str):
        storage_context = StorageContext.from_defaults(persist_dir=store_path)
        
        self.vector_index= load_index_from_storage(storage_context=storage_context)

if __name__ == "__main__":
    knowledgeBase=KnowledgeBase()
    knowledgeBase.load_documents(["rag/知识库数据集/知识"])
    knowledgeBase.load_query_engine()
    result=knowledgeBase.chat_query("我感到虚热，寒冷，头疼，是的了什么并病")
    print(result)
    
           
        
