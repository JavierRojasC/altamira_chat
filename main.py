
from fastapi import FastAPI, Query, Body
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import openai
from openai import OpenAI
# Configuración de claves y API
os.environ['OPENAI_API_KEY'] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")

client_o = OpenAI()
# Inicialización de clientes y bases de datos
BD = "C:/Users/mrojas/Personal/ALTAMIRA/BD"
client = chromadb.PersistentClient(path=BD)
embeddings = OpenAIEmbeddings()
chroma_directory = BD
db = Chroma(persist_directory=chroma_directory, embedding_function=embeddings)

# Inicialización de la aplicación FastAPI
app = FastAPI()

@app.post("/chat_altamira")
async def chat_altamira(query: str = Body(..., description="La pregunta para buscar en la base de datos")):
        # Búsqueda de contexto en la base de datos
    context = db.similarity_search_with_score(query, k=1)
    
    complete = f"""
    Responde a esta pregunta: {query}, tienes este contexto: {context[0][0].page_content}
    Siempre contesta mencionando el nombre completo del estudiante, solo refiérete al estudiante mediante el nombre completo.
    Si te preguntan cosas generales que no tienen que ver con estudiantes, no menciones ningún estudiante, no saques información del contexto.
    """
    
    completion =  client_o.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres un asistente de inteligencia artificial diseñada por Javier Rojas, tu nombre es Iris, puedo ayudarte con información que el usuario necesite, tienes información general de los estudiantes de la Unidad Educativa Particular Altamira, solo te limitas a eso"},
            {"role": "user", "content": complete}
        ]
    )

    return {"response": completion.choices[0].message.content}

# Ejecución de la aplicación
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
