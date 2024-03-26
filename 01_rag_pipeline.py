import os

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder, SentenceTransformersDocumentEmbedder, \
    OpenAIDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import ComponentDevice, Secret
from langchain_community.document_loaders.wikipedia import WikipediaLoader
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

documents = [Document(content="My name is Wolfgang and I live in Berlin"),
             Document(content="I saw a black horse running"),
             Document(content="Germany has many big cities")]


docs=[]
for document in documents:
    docs.append(
        Document(content=document.content,meta={})
    )

print(docs)
document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")


document_embedder = OpenAIDocumentEmbedder()
document_writer = DocumentWriter(document_store)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("document_embedder", document_embedder)
indexing_pipeline.add_component("document_writer", document_writer)

indexing_pipeline.connect("document_embedder", "document_writer")

indexing_pipeline.run({"document_embedder": {"documents": documents}})
print(document_store)

template = """
Given the following information, answer the question.

Context: 
{% for document in documents %}
    {{ document.content }}
{% endfor %}
Question: {{query}}
"""
pipe = Pipeline()

pipe.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", OpenAIGenerator(model="gpt-4"))
pipe.connect("retriever", "prompt_builder")
pipe.connect("prompt_builder", "llm")

result = pipe.run({
    "documents":documents,
    "query":"What was the black horse doing?"
})
print(result)
