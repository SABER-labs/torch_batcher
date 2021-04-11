# Torch Batcher
Serve batched requests using redis, can scale linearly by increasing the number of workers per device and along devices.

## Dependencies
* [Install Redis](https://redis.io/topics/quickstart)
* `pip3 install -r requriments.txt`

## Usage

* For Linear Scaling, start nvidia-cuda-mps-control, Check [Section 2.1.1 GPU utilization](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf) for details.
    ```bash
    nvidia-cuda-mps-control -d # To start

    # To exit mps after stoping the server do.
    nvidia-cuda-mps-control # Will enter the command prompt
    quit # enter command to quit
    ```

* Start Redis
    ```bash
    redis-server --save "" --appendonly no
    ```

* Start Batch-Serving
    ```bash
    supervisord -c supervisor.conf # Start 3 workers on a single gpu
    ```

* Start Batch benchmark
    ```bash
    python3 bench_batched.py
    ```