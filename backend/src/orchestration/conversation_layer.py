import json
import uuid

class ConversationLayer:

    def __init__(self, redis_client):
        self.redis = redis_client
        #self.local_store = {}  # fallback if Redis is not available

    # ---------------------------
    def create_chat(self, user_id):
        chat_id = str(uuid.uuid4())

        self.redis.lpush(f"user:{user_id}:chats", chat_id)

        self.redis.setex(
            f"chat:{chat_id}:meta",
            86400,
            json.dumps({"title": "New Chat"})
        )

        return chat_id

    # ---------------------------
    def get_user_chats(self, user_id):
        if not self.redis:
            return []
        return self.redis.lrange(f"user:{user_id}:chats", 0, -1)

    # ---------------------------
    def add_message(self, chat_id, role, content):
        if not self.redis:
            return []
        self.redis.rpush(
            f"chat:{chat_id}:messages",
            json.dumps({"role": role, "content": content})
        )

        self.redis.expire(f"chat:{chat_id}:messages", 86400)

    # ---------------------------
    def get_messages(self, chat_id):
        if not self.redis:
            return []
        msgs = self.redis.lrange(f"chat:{chat_id}:messages", 0, -1)

        return [json.loads(m) for m in msgs]

    # ---------------------------
    def set_title(self, chat_id, title):
        if not self.redis:
            return []
        self.redis.setex(
            f"chat:{chat_id}:meta",
            86400,
            json.dumps({"title": title})
        )

    def get_title(self, chat_id):
        if not self.redis:
            return []
        data = self.redis.get(f"chat:{chat_id}:meta")
        return json.loads(data)["title"] if data else "Chat"
    
    def set_last_chat(self, user_id, chat_id):
        if not self.redis:
            return []
        self.redis.set(f"user_last_chat:{user_id}", chat_id)


    def get_last_chat(self, user_id):
        if not self.redis:
            return []
        return self.redis.get(f"user_last_chat:{user_id}")

