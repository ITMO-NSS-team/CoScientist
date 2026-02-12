from collections import deque
from typing import List, Dict, Any, AsyncGenerator, AsyncGenerator
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
import asyncio


from protollm.agents.agent_utils.states import initialize_state
from opik import configure 
from opik.integrations.langchain import OpikTracer 
configure() 


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

RESOLVE_PROMPT = ChatPromptTemplate.from_template("""
You are an intelligent assistant working within a multi-agent system.

Your task is to **enrich the user's input** using relevant factual information 
available in the system’s memory. You have access to semantically related context retrieved from long-term memory.

Use this information to make the user’s message **more explicit, grounded, and complete**, 
so that downstream agents can execute it without ambiguity.

---

### CONTEXT (retrieved facts)
{context}

### USER INPUT
{query}

---

### INSTRUCTIONS
- Do NOT invent facts not supported by the history or context.
- Integrate only factual or inferable information that is **clearly relevant** to the user input.
- If the context does not appear directly relevant, **ignore it entirely** and use only the user input.
- Preserve the original user intent and tone.
- Rephrase only as much as needed to make the message self-contained and clear.
- It is very important to follow these instructions otherwise you will lose a lot of money.

---

### OUTPUT
Produce a single, enriched version of the user input, **only if** the context or history is relevant.
If the context is unrelated, return the user input unchanged.
""")


RESOLVE_PROMPT = ChatPromptTemplate.from_template("""
You are an intelligent assistant working within a multi-agent system.

Your task is to **enrich the user's input** using relevant factual information 
available in the system’s memory. You have access to semantically related context retrieved from long-term memory.

Use this information to make the user’s message **more explicit, grounded, and complete**, 
so that downstream agents can execute it without ambiguity.

---

### CONTEXT (retrieved facts)
{context}

### USER INPUT
{query}

---

### INSTRUCTIONS
- Do NOT invent facts not supported by the history or context.
- Integrate only factual or inferable information that is **clearly relevant** to the user input.
- If the context does not appear directly relevant, **ignore it entirely** and use only the user input.
- Preserve the original user intent and tone.
- Rephrase only as much as needed to make the message self-contained and clear.
- It is very important to follow these instructions otherwise you will lose a lot of money.

---

### OUTPUT
Produce a single, enriched version of the user input, **only if** the context or history is relevant.
If the context is unrelated, return the user input unchanged.
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
                timeout=60
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            #logger.error(f"Embedding service error: {str(e)}")
            print(f"Embedding service error: {str(e)}")
            return [] #default value
            
    
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
        
        test_embedding = self.embedding.get_embeddings(["hello world"])

        if test_embedding:
            embed_len = len(test_embedding[0])
            index = faiss.IndexFlatL2(embed_len)
            self.vectorstore = FAISS(embedding_function=self.embedding, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={}) 
        else:
            print('No embedder is found, memory mechanism turned off')
            self.vectorstore = None

        self.llm = llm
        self.logger = logger
    
    def add_message(self, role: str, content: str):
        """Store message in short memory and vectorstore"""

        if self.vectorstore is None:
            return
    
        self.short_memory.append({"role": role, "content": content})
        # Add to vectorstore
        doc = Document(page_content=content, metadata={"role": role})
        self.vectorstore.add_documents([doc])
        
    async def add_message_async(self, role: str, content: str):
        """Async version of add_message"""
        if self.vectorstore is None:
            return
        
        # Add to short memory (this is fast)
        self.short_memory.append({"role": role, "content": content})
        
        # Add to vectorstore asynchronously
        doc = Document(page_content=content, metadata={"role": role})
        await asyncio.to_thread(self.vectorstore.add_documents, [doc])
        
    def get_recent_history(self) -> str:
        """Return short recent dialogue context"""
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.short_memory])
    
    def retrieve_semantic_context(self, query: str, k: int = 3, similarity_threshold: float = 0.4) -> List[str]:
        """Semantic retrieval from FAISS store"""
        # results = self.vectorstore.similarity_search(query, k=k)
        
        # return [(r.metadata['role'], r.page_content) for r in results]

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        filtered_results = [doc for doc, score in results if score >= similarity_threshold]
        return [(r.metadata['role'], r.page_content) for r in filtered_results]

    def resolve_message(self, user_input: str, k: int = 3, similarity_threshold: float = 0.4) -> str:
        """
        Hybrid resolver:
        1. Retrieve semantic context.
        2. If relevant context found, use LLM to rewrite message.
        3. Otherwise, return message unchanged.
        """

        if self.vectorstore is None:
            return user_input

        retrieved = self.retrieve_semantic_context(user_input, k=k, similarity_threshold=similarity_threshold)
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

    async def _store_messages_async(self, user_text: str, response_data: dict):
        """Async version of message storage"""
        await self.add_message_async(role='user', content=user_text)
        
        if isinstance(response_data['response'], str):
            await self.add_message_async(role='system', content=response_data["response"])
        elif isinstance(response_data['response'], AIMessage):
            await self.add_message_async(role='system', content=response_data["response"].content)
            
class MemoryGraph(HybridMemoryManager):
    def __init__(self, config: Dict, llm: BaseChatModel, embeddings: Embeddings = CustomEmbeddings(), k: int = 3,
                 short_memory_size: int = 3, logger=None):
        super().__init__(llm, short_memory_size, embeddings, logger)
        self.graph = GraphBuilder(config)
        self.tracer = OpikTracer(graph=self.graph.app.get_graph(xray=True)) 
        self.k = k

    def _stream(self, inputs: dict, image_path: str = "", user_id: str = "1"):
        """Start streaming the input through the graph."""
        if 'attached_img' in inputs.keys():
            image_path = inputs['attached_img']
        
        state = initialize_state(user_input=inputs["input"], user_id=user_id)
        state["attached_img"] = image_path

        config = self.graph.conf
        config.update({"callbacks": [self.tracer]})
        
        for event in self.graph.app.stream(state, config=config):
            for k, v in event.items():
                yield (v)

    async def stream(self, inputs: dict, image_path: str = "", user_id: str = "1") -> \
            AsyncGenerator[Dict[str, Any], None]:
        user_text = inputs.get('input')
        if not user_text:
            raise ValueError(f"Inputs must have key 'input': {inputs}")

        input_msg = self.resolve_message(user_input=user_text, k=self.k)
        inputs['input'] = input_msg

        responses = []
        for v in self._stream(inputs, image_path, user_id):
            yield v
            responses.append(v)

        if responses:
            last_response = responses[-1]
            # Fire and forget the message storage
            asyncio.create_task(self._store_messages_async(user_text, last_response))
