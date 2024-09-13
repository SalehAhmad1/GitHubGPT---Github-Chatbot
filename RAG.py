import os
import shutil
import time
import logging
from dotenv import load_dotenv
from git import Repo
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import GitLoader
from openai import OpenAI

class GitHubGPT:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.embeddings = self.__initialize_embeddings()
        self.vector_db = self.__initialize_vector_db()
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)
        self.system_prompt = self.__initialize_system_prompt()
        self.thread_id = None
        self.assistant_id = self.__create_assistant(name='Github GPT', instructions='Please address the user as Github GPT')
        self.thread_messages = []  # Store the conversation history

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
    What are you? A well-informed, intelligent chatbot that can interact with a codebase.
    What do you do? You are always provided with some file content from a codebase and a question/prompt. Your job is to generate a response.
    What should be the tone of your output? It should be friendly, helpful, confident, and narrative.
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        data = text_splitter.split_documents(data)
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

    def __create_assistant(self, name, instructions, model="gpt-3.5-turbo-16k"):
        assistant = self.client.beta.assistants.create(
            name=name,
            instructions=instructions,
            model=model,
        )
        print(f'Assistant created with ID: {assistant.id}')
        return assistant.id

    def __retrieve_documents(self, prompt, k=3):
        retrieved_documents = self.vector_db.similarity_search(
            prompt,
            k=k
        )
        return retrieved_documents
    
    @staticmethod
    def __concatenate_documents(documents):
        print(f'Length of docs to concatenate: {len(documents)}')
        all_content = ''
        for idx, doc in enumerate(documents):
            print(f"Retrieved Document: {idx} --- [{doc.metadata}]")
            all_content += "Chunk:" + str(idx) + ":\n" + doc.page_content + "\n\n"
        print("\n\n")
        return all_content

    def query(self, prompt, instructions="Please address the user as Github User"):
        # Step 1: Retrieve relevant documents based on the user's query
        retrieved_documents = self.__retrieve_documents(prompt)
        context = self.__concatenate_documents(retrieved_documents)

        # Step 2: Add the new user prompt and context to the conversation history
        user_query = f"Context from codebase: {context}\nUser query: {prompt}\n"
        self.thread_messages.append({
            "role": "user",
            "content": user_query,
        })

        # Step 3: If there's no existing thread, create a new one; otherwise, append to the existing thread
        if not self.thread_id:
            thread = self.client.beta.threads.create(
                messages=self.thread_messages
            )
            self.thread_id = thread.id
            print(f'Thread created with ID: {self.thread_id}')
        else:
            print(f'Using the existing thread ID: {self.thread_id}')
            # Add the new message to the existing thread
            self.client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role="user",
                content=user_query
            )

        Messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
        print(f'Count of messages(input prompt + generated response) in the thread:', len(Messages.data))

        # Step 4: Run the assistant on the created or updated thread
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
            instructions=instructions,
            stream=True,
        )
        
        text = ''
        for event in run:
            try:
                text = event.data.delta.content[0].text.value
                yield text
            except:
                continue
