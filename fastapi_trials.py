from fastapi import FastAPI, UploadFile, File, HTTPException
import logging
import sys
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from IPython.display import Markdown, display
from llama_index.prompts import PromptTemplate

app = FastAPI()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

@app.on_event("startup")
async def startup_event():
    global llm, callback_manager, service_context  # Define as global to access in other endpoints
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    llm = LlamaCPP(
        model_url="https://huggingface.co/TheBloke/Llama-2-7B-chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf",
        model_path=None,
        temperature=0.0,
        max_new_tokens=1024,
        context_window=3900,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 4},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )



    service_context = ServiceContext.from_defaults(
        llm=llm, chunk_size=100, embed_model="local", callback_manager=callback_manager
    )

@app.post("/create_index/")
async def create_index():
    global index  # Define as global to access in execute_query
    try:
        transcript_directory = r"C:\Users\14055\Desktop\SOURCE_DOCUMENTS"
        documents = SimpleDirectoryReader(transcript_directory).load_data()
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context, show_progress=True
        )
        storage_directory = r"C:\Users\14055\Desktop\db_folder"
        index.storage_context.persist(persist_dir=storage_directory)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "index created"}

@app.post("/query/")
async def execute_query(query: str):
    try:
        query_engine = index.as_query_engine(
            service_context=service_context, similarity_top_k=3
        )
        response = query_engine.query(query)
        # Assuming Markdown rendering will be handled client-side
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
        
@app.post("/upload/")
async def upload_file(file: UploadFile):
    try:
        file_location = f"C:\\Users\\14055\\Desktop\\SOURCE_DOCUMENTS\\{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        return {"status": "file uploaded and index updated"}  # Update index accordingly
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))