from infer import BatchInference
from utils import pack_req
import aioredis

class BatchInferenceClient:

    async def __aenter__(self):
        self.redis = await aioredis.create_redis_pool('redis://localhost')
        print("Client Redis pool created")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        print("Client Redis pool closed")
        self.redis.close()
        await self.redis.wait_closed()

    async def infer(self, req_id, text):
        response_key = f'{BatchInference.reponse_key}:{req_id}'
        ch = await self.redis.subscribe(response_key)
        packed_binary = pack_req(req_id, text)
        await self.redis.rpush(BatchInference.request_key, packed_binary)
        response = await ch[0].get_json()
        await self.redis.unsubscribe(response_key)
        assert(req_id == response.get("req_id"))
        return response.get("response_text")