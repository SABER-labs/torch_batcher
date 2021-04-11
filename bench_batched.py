import time
from infer import gen_text
from itertools import count
from client import BatchInferenceClient
import asyncio
import uvloop

class Benchmarker:

    def __init__(self):
        self.request_counter = count()

    async def time_per_request(self, model):
        start = time.perf_counter()
        _ = await model.infer(next(self.request_counter), "some random text being sent")
        total_time = ((time.perf_counter() - start)*1000)
        return total_time

    async def benchmark(self, num_req=5):
        async with BatchInferenceClient() as model:
            start = time.perf_counter()
            time_taken_per_example = await asyncio.gather(*[self.time_per_request(model) for _ in range(num_req)])
            total_time = ((time.perf_counter() - start)*1000)
            print(f"Average time taken for num_req {num_req} is: {sum(time_taken_per_example)/num_req:.2f}ms")
            print(f"Total time taken for {num_req} was {total_time:.2f}ms")

async def main():
    benchmarker = Benchmarker()
    await benchmarker.benchmark(64)
    await benchmarker.benchmark(32)
    await benchmarker.benchmark(16)
    await benchmarker.benchmark(8)
    await benchmarker.benchmark(4)
    await benchmarker.benchmark(2)
    await benchmarker.benchmark(1)

if __name__ == "__main__":
    uvloop.install()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()