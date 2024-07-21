#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import torch
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from typing import List, Dict, Tuple, Union
import torch
from PIL import Image
import pickle
from openai import OpenAI
import os
import torch
import time
class MemoryIndex:
    def __init__(self,number_of_neighbours,use_openai=False):
        self.documents = {}
        self.document_vectors = {}
        self.use_openai=use_openai
        if use_openai:
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')\
        # self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.number_of_neighbours=number_of_neighbours
        
    def load_documents_from_json(self, file_path,emdedding_path=""):

        with open(file_path, 'r') as file:
            data = json.load(file)
            for doc_id, doc_data in data.items():
                self.documents[doc_id] = doc_data
                self.document_vectors[doc_id] = self._compute_sentence_embedding(doc_data)
    
        # save self.documents and self.document_vectors to pkl file
        m=[self.documents,self.document_vectors]
        with open(emdedding_path, 'wb') as file:
            pickle.dump(m, file)
        return emdedding_path
    def load_embeddings_from_pkl(self, pkl_file_path):
        #read the pkl file 
        with open(pkl_file_path, 'rb') as file:
            data = pickle.load(file)
            self.documents=data[0]
            self.document_vectors=data[1]
            
        
    def load_data_from_pkl(self, pkl_file_path):
        with open(pkl_file_path, 'rb') as file:
            data = pickle.load(file)
            for doc_id, doc_data in data.items():
                self.documents[doc_id] = doc_data
                self.document_vectors[doc_id] = doc_data  
    def _compute_sentence_embedding(self, text: str) -> torch.Tensor:  
        if self.use_openai:      
            done=False
            while not done:
                try:
                    embedding=self.client.embeddings.create(input = [text], model="text-embedding-3-small").data[0].embedding
                    # Convert the list to a PyTorch tensor
                    embedding = torch.tensor(embedding)
                    done=True
                except Exception as e:
                    print("error",e)
                    print("text",text)
                    # sleep for 5 seconds and try again 
                    time.sleep(5)
                    continue
        else:
            return self.model.encode(text, convert_to_tensor=True).to(self.device)
            
        return embedding

    def search_by_similarity(self, query: str) -> List[str]:

        query_vector = self._compute_sentence_embedding(query)
        scores = {doc_id: torch.nn.functional.cosine_similarity(query_vector, doc_vector, dim=0).item()
                  for doc_id, doc_vector in self.document_vectors.items()}
        sorted_doc_ids = sorted(scores, key=scores.get, reverse=True)
        sorted_documents=[self.documents[doc_id] for doc_id in sorted_doc_ids]
        if self.number_of_neighbours == -1:
            return list(self.documents.values()), list(self.documents.keys())
        if self.number_of_neighbours > len(sorted_documents):
            return sorted_documents, sorted_doc_ids
        # if the retrieved document is the summary, return the summary and the next document to grauntee that always retieve clip name.
        if self.number_of_neighbours==1 and sorted_doc_ids[0]=='summary': 
            return sorted_documents[0:2], sorted_doc_ids[:2]
        return sorted_documents[:self.number_of_neighbours], sorted_doc_ids[:self.number_of_neighbours]

# # main function
# if __name__ == "__main__":
#     memory_index = MemoryIndex(-1,use_openai=True)
#     memory_index.load_documents_from_json('workspace/results/llama_vid/tt0035423.json')
#     print(memory_index.documents.keys())
#     docs,keys=memory_index.search_by_similarity('kerolos')