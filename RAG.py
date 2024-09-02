import os
import shutil
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from git import Repo
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import GitLoader
from openai import OpenAI

class GitHubGPT:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.embeddings = self.__initialize_embeddings()
        self.vector_db = self.__initialize_vector_db()
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)
        self.system_prompt = self.__initialize_system_prompt()
        self.thread = None
        self.thread_id = None
        self.messages = []  # List to store messages

    def __initialize_embeddings(self):
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.OPENAI_API_KEY
        )

    def __initialize_vector_db(self):
        if not os.path.exists("./vector_db"):
            os.makedirs("./vector_db", mode=0o777)
            
        return Milvus(
            embedding_function=self.embeddings,
            connection_args={"uri": "./vector_db/milvus_example.db"},
            auto_id=True,
            collection_name="github_gpt",
        )
        
    def __initialize_system_prompt(self):
        return '''
    What are you? A well informed, intelligent chatbot which can talk to a given codebase.
    What do you do? You are always given some file content from a codebase and a question/prompt. Your job is to generate a response.
    What should be the tone of your output? It should be friendly, helpful, confident, narrative.
    What outputs can we expect from you? You can be asked to generate documentations, code, or anything else only relevant to the given codebase content.
    '''

    @staticmethod
    def __clean_repo_name(name):
        return name.replace('-', '_')
    
    @staticmethod
    def __declean_repo_name(name):
        return name.replace('_', '-')
    
    def __add_repo_data_to_db(self):
        data = self.loader.load()
        print(f'Length of Data to Add: {len(data)}')
        print(f'Adding Data to Milvus Vector DB')
        self.vector_db.add_documents(documents=data)
        print(f'Done Adding Data to Milvus Vector DB')
    
    def add_repo(self, repo_url):
        repo_name = repo_url.split('/')[-1]
        repo_save_path = f"./Data/Repos"
        if not os.path.exists(repo_save_path):
            os.makedirs(repo_save_path)
        else:
            shutil.rmtree(repo_save_path)
            os.makedirs(repo_save_path)
        repo_save_path = repo_save_path + "/" + self.__clean_repo_name(repo_name)
        
        print(f'Cloning the repo from: {repo_url}')
        repo = Repo.clone_from(
            repo_url, 
            to_path=repo_save_path,
            branch="master"
        )
        print(f'Repo Cloned to: {repo_save_path}')
        self.repo_save_path = repo_save_path
        self.branch = repo.head.reference
        self.loader = GitLoader(repo_path=repo_save_path, branch=self.branch)
        self.__add_repo_data_to_db()

    def load_repo(self):
        repo_save_path = "./Data/Repos"
        repo_name = os.listdir(repo_save_path)[0]
        self.repo_save_path = repo_save_path + "/" + repo_name
        self.branch = "master"
        print(f'Loading repo: {repo_name}')
        print(f'Branch: {self.branch}')
        print(f'Repo path: {self.repo_save_path}')
        self.loader = GitLoader(repo_path=self.repo_save_path, branch=self.branch)
        self.__add_repo_data_to_db()

    def __retrieve_documents(self, prompt, k=3):
        retrieved_documents = self.vector_db.similarity_search(
            prompt,
            k=k
        )
        return retrieved_documents
    
    @staticmethod
    def __concatenate_documents(documents):
        print(f'Length of docs to concatenate: {len(documents)}')
        All_content = ''
        for idx, doc in enumerate(documents):
            print(f"Retrieved Document: {idx} --- [{doc.metadata}]")
            All_content += "Chunk:" + str(idx) + ":\n" + doc.page_content + "\n\n"
        print("\n\n")
        return All_content
    
    def query(self, prompt):
        retrieved_documents = self.__retrieve_documents(prompt)
        context = self.__concatenate_documents(retrieved_documents)

        # Add the user message to the messages list
        self.messages.append({"role": "user", "content": f"Context from codebase:{context}\nUser query prompt:{prompt}\nResponse:\n"})
        
        completion = self.client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"{self.system_prompt}"},
            {"role": "user", "content": f"Context from codebase:{context}\nUser query prompt:{prompt}\nResponse:\n"},
        ],
        stream=True
        )
        
        response_text = ''
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content
        
        self.messages.append({"role": "assistant", "content": response_text})
        print(f'\n\nMessage has been added to the list of messages: {self.messages}, which will be later on a part of the retrieved thread. To retrieve the thread, call retrieve_thread()')
    
    def retrieve_thread(self):
        thread = self.client.beta.threads.create(messages=self.messages)
        self.thread_id = thread.id
        self.thread = thread
        print(f'The thread ID is: {thread.id}')
        return thread