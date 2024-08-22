from FlagEmbedding import FlagModel
import json
from tqdm import tqdm
import faiss
import logging
import config
import numpy as np

args = config.get_args()

def load_embeddings(load_path):
    return np.load(load_path)

def save_embeddings(embeddings, save_path):
    np.save(save_path, embeddings)


def setup_logging():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# multilingual-e5-large是1024维度
def load_embedding_model(model_path):
    model = FlagModel(model_path,
                      query_instruction_for_retrieval="检索与问题相关的内容",
                      use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    logging.info(f"Model {model_path} Uploaded")
    return model


def create_index_knowledge_base(model_path, dimension):
    model = load_embedding_model(model_path)
    knowledge_sources = load_json_data(args.knowledge_source)

    embeddings = []
    for knowledge in tqdm(knowledge_sources, desc="Creating embeddings from knowledge source"):
        embedding = model.encode(knowledge)
        embeddings.append(embedding)

    embeddings = np.array(embeddings).astype('float32')
    save_embeddings(embeddings, args.knowledge_embedding)

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, args.knowledge_index)

    logging.info(f"Model {model_path} loaded and FAISS index created with {index.ntotal} vectors.")


def load_sentences(file_path):

    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentences.append(line.strip())

    return sentences

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data

if __name__ == "__main__":
    # create indexed knowledge base first
    create_index_knowledge_base(args.embedding_model, args.dimension)