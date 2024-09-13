
import os
import shutil
import time
import logging
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
        self.thread_id = None
        self.assistant_id = self.__create_assistant(name='Github GPT', instructions='Please address the user as Github GPT')

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
        All_content = ''
        for idx, doc in enumerate(documents):
            print(f"Retrieved Document: {idx} --- [{doc.metadata}]")
            All_content += "Chunk:" + str(idx) + ":\n" + doc.page_content + "\n\n"
        print("\n\n")
        return All_content

    def query(self, prompt, instructions="Please address the user as Github User"):
        # Step 1: Retrieve relevant documents based on the user's query
        retrieved_documents = self.__retrieve_documents(prompt)
        context = self.__concatenate_documents(retrieved_documents)

        # Step 2: Create a thread using the retrieved context and user prompt
        user_query = f"Context from codebase: {context}\nUser query: {prompt}\n"
        thread = self.client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": user_query,
                }
            ]
        )
        self.thread_id = thread.id
        print(f'Thread created with ID: {self.thread_id}')

        # Step 3: Run the assistant on the created thread
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
            instructions=instructions,
        )

        # Step 4: Wait for the assistant's response
        self.wait_for_run_completion(self.client, thread_id=self.thread_id, run_id=run.id)

    def wait_for_run_completion(self, client, thread_id, run_id, sleep_interval=5):
        while True:
            try:
                run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
                if run.completed_at:
                    elapsed_time = run.completed_at - run.created_at
                    formatted_elapsed_time = time.strftime(
                        "%H:%M:%S", time.gmtime(elapsed_time)
                    )
                    print(f"Run completed in {formatted_elapsed_time}")
                    logging.info(f"Run completed in {formatted_elapsed_time}")
                    # Get messages here once the run is completed!
                    messages = client.beta.threads.messages.list(thread_id=thread_id)
                    print(f'All messages: {messages.data}\n')
                    last_message = messages.data[0]
                    response = last_message.content[0].text.value
                    print(f"Assistant Response: {response}\n\n****\n\n")
                    break
            except Exception as e:
                logging.error(f"An error occurred while retrieving the run: {e}")
                break
            logging.info("Waiting for run to complete...")
            time.sleep(sleep_interval)
