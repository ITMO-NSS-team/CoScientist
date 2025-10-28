from collections import deque
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain.prompts import ChatPromptTemplate
import numpy as np
import os
import requests
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from protollm.agents.builder import GraphBuilder
from langchain_core.messages import AIMessage

#from dotenv import load_dotenv 
#load_dotenv('/app/config.env')


RESOLVE_PROMPT = ChatPromptTemplate.from_template("""
You are an intelligent assistant working within a multi-agent system.

Your task is to **enrich the user's input** using relevant factual information 
available in the system’s memory. You have access to both the recent dialogue 
history and semantically related context retrieved from long-term memory.

Use this information to make the user’s message **more explicit, grounded, and complete**, 
so that downstream agents can execute it without ambiguity.

---

### HISTORY (recent dialogue)
{history}

### CONTEXT (retrieved facts)
{context}

### USER INPUT
{query}

---

### INSTRUCTIONS
- Do NOT invent facts not supported by the history or context.
- Integrate only factual or inferable information.
- You must reserve the original user intent and tone.
- Rephrase only as much as needed to make the message self-contained and clear.
- It is very important to follow these instructions otherwise you will lose a lot of money.

---

### OUTPUT
Produce a single, enriched version of the user input, with incorporated factual context.
""")

embedding_host = os.environ.get('EMBEDDING_HOST')
embedding_port = os.environ.get('EMBEDDING_PORT')
embedding_endpoint = "/embed"

class CustomEmbeddings(Embeddings):

    @staticmethod
    def get_embeddings(texts: List[str]) -> List[np.ndarray]:
        """
        Retrieves numerical representations of text for semantic understanding and comparison.

        Args:
            texts: A list of strings to be converted into embeddings.

        Returns:
            list[np.ndarray]: A list of embedding vectors, one corresponding to each input text.

        Raises:
            Exception: If the embedding service is unavailable or returns an error.
        """
        embedding_service_url = "http://" + embedding_host + ":"\
                                + str(embedding_port)\
                                + embedding_endpoint
        try:

            response = requests.post(
                embedding_service_url,
                json=texts,
                timeout=1000
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            #logger.error(f"Embedding service error: {str(e)}")
            print(f"Embedding service error: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Embed list of texts."""
        return self.get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

class HybridMemoryManager:
    def __init__(self, llm: BaseChatModel, 
                        short_memory_size: int = 3, 
                        embeddings: Embeddings = CustomEmbeddings(),
                        logger = None):

        self.short_memory = deque(maxlen=short_memory_size)
        self.embedding = embeddings

        index = faiss.IndexFlatL2(len(self.embedding.get_embeddings(["hello world"])[0]))
        self.vectorstore = FAISS(embedding_function=self.embedding, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})

        self.llm = llm
        self.logger = logger
    
    def add_message(self, role: str, content: str):
        """Store message in short memory and vectorstore"""
    
        self.short_memory.append({"role": role, "content": content})
        # Add to vectorstore
        doc = Document(page_content=content, metadata={"role": role})
        self.vectorstore.add_documents([doc])
        
        # Check if document was added
        doc_count = len(self.vectorstore.index_to_docstore_id)
    
    def get_recent_history(self) -> str:
        """Return short recent dialogue context"""
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.short_memory])
    
    def retrieve_semantic_context(self, query: str, k: int =3) -> List[str]:
        """Semantic retrieval from FAISS store"""
        results = self.vectorstore.similarity_search(query, k=k)
        return [(r.metadata['role'], r.page_content) for r in results]

    def resolve_message(self, user_input: str, k: int = 3, similarity_threshold: float = 0.4) -> str:
        """
        Hybrid resolver:
        1. Retrieve semantic context.
        2. If relevant context found, use LLM to rewrite message.
        3. Otherwise, return message unchanged.
        """
        retrieved = self.retrieve_semantic_context(user_input, k=k)
        history_text = self.get_recent_history()

        if not retrieved or not history_text:
            # fallback: just recent history
            if self.logger is not None:
                self.logger.info(f'No memory stored, returning original msg: {user_input}')
            return user_input
        
        # Optionally: could check similarity scores if using FAISS similarity_search_with_score
        context_text = "\n".join([f"{role}: {msg}" for (role, msg) in retrieved])

        # Use LLM to rewrite message
    
        resolved = self.llm.invoke(
            RESOLVE_PROMPT.format_messages(
                history=history_text,
                context=context_text,
                query=user_input
            )
        )
        if self.logger is not None:
            self.logger.info(f'RESOLVED MESSAGE BY MEMORY: ORIGINAL: {user_input},\t RESOLVED: {resolved.content.strip()}')
        return resolved.content.strip()


class MemoryGraph(HybridMemoryManager):
    def __init__(self, config: Dict, llm: BaseChatModel, embeddings: Embeddings = CustomEmbeddings(), k: int = 3, short_memory_size: int = 3, logger = None):
        super().__init__(llm, short_memory_size, embeddings, logger)
        self.graph = GraphBuilder(config)
        self.k = k

    def stream(self, inputs: dict, image_path: str = "", user_id: str = "1"):
        user_text = inputs.get('input')
        if not user_text:
            raise ValueError(f"Inputs must have key 'input': {inputs}")

        input_msg = self.resolve_message(user_input=user_text, k = self.k)
        inputs['input'] = input_msg

        for v in self.graph.stream(inputs, image_path, user_id):
            yield (v)

        self.add_message(role='user', content=user_text)
        if isinstance(v['response'], str):
            self.add_message(role='system', content=v["response"])
        elif isinstance(v['response'], AIMessage):
            self.add_message(role='system', content=v["response"].content)