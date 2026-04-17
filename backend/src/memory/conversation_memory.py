from langchain.memory import ConversationBufferMemory


class ConversationMemory:

    def __init__(self):

        self.memory = ConversationBufferMemory(
            return_messages=True
        )

    def add_user_message(self, message):

        self.memory.chat_memory.add_user_message(message)

    def add_ai_message(self, message):

        self.memory.chat_memory.add_ai_message(message)

    def get_history(self):

        return self.memory.load_memory_variables({})["history"]
