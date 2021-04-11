import torch
from utils import unpack_req
import aioredis
import asyncio
import msgpack
import secrets
import time
from aiorun import run

def gen_text(batch_size):
    text_len = 50
    text_padded = torch.randint(low=0, high=148,
                                size=(batch_size, text_len),
                                dtype=torch.long).cuda()
    input_lengths = torch.IntTensor([text_padded.size(1)]*
                                    batch_size).cuda().long()
    return (text_padded, input_lengths)

class BatchInference:

    request_key = "REQ"
    reponse_key = "RES"

    def __init__(self):
        self.model = torch.jit.load("/media/vigi99/ssd_1tb/Problems/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/tacotron2_jit.pt")
        self.model.eval().cuda().half()
        self._warmup()
        self.max_delay_in_ms = 50
        self.poll_time_in_ms = 5 
        self.max_batch_size = 32
        self.loop_times = int(self.max_delay_in_ms / self.poll_time_in_ms)

    async def __aenter__(self):
        print("Server Redis pool created")
        self.redis = await aioredis.create_redis_pool('redis://localhost')
        return self

    async def __aexit__(self, exc_type, exc, tb):
        print("Server Redis pool closed")
        self.redis.close()
        await self.redis.wait_closed()

    @torch.no_grad()
    def infer(self, text_input, padded_lengths):
        return self.model(text_input, padded_lengths)

    def _warmup(self):
        text_padded, input_lengths = gen_text(1)
        print(f"Model warming up")
        self.infer(text_padded, input_lengths)
        self.infer(text_padded, input_lengths)
        self.infer(text_padded, input_lengths)
        print(f"Model warmed up")

    async def forever_loop(self):
        while True:
            req_counts = await self.redis.llen(BatchInference.request_key)
            for _ in range(self.loop_times):
                if req_counts <= self.max_batch_size:
                    await asyncio.sleep(self.poll_time_in_ms * 1e-3)
                else:
                    break
            tr = self.redis.multi_exec()
            tr.lrange(BatchInference.request_key, 0, self.max_batch_size - 1)
            tr.ltrim(BatchInference.request_key, self.max_batch_size, -1)
            list_of_reqids, _ = await tr.execute()
            if len(list_of_reqids) > 0:
                req_ids, texts  = zip(*[unpack_req(req_bin) for req_bin in list_of_reqids])
                text_padded, input_lengths = gen_text(len(texts))
                audios = self.infer(text_padded, input_lengths)
                for i in range(len(req_ids)):
                    req_id = req_ids[i]
                    audio = audios[0][i, :, :]
                    reponse_text = self.audio_to_b64(audio)
                    await self.redis.publish_json(f'{BatchInference.reponse_key}:{req_id}', {
                        'req_id': req_id,
                        'response_text': reponse_text
                    })

    def audio_to_b64(self, audio):
        return secrets.token_urlsafe(64)

    @staticmethod
    async def main_loop():
        async with BatchInference() as inference_engine:
            await inference_engine.forever_loop()


if __name__ == "__main__":
    run(BatchInference.main_loop(), use_uvloop=True)