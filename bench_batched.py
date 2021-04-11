import time
from infer import gen_text
from itertools import count
from client import BatchInferenceClient
import asyncio
import uvloop
from beautifultable import BeautifulTable
from numpy import percentile


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
            return total_time, percentile(time_taken_per_example, 50), percentile(time_taken_per_example, 95), percentile(time_taken_per_example, 95)


async def main():
    benchmarker = Benchmarker()
    table = BeautifulTable()
    table.columns.header = ["Batch_Size", "Total time in ms", "p50", "p95", "p99"]
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        total_time, p_50, p_95, p_99 = await benchmarker.benchmark(batch_size)
        table.rows.append([batch_size, total_time, p_50, p_95, p_99])
        print(f"Processed {batch_size} req in {total_time}ms")
    print(table)

if __name__ == "__main__":
    uvloop.install()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
