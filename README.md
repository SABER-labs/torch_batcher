# Torch Batcher
Serve batched requests using redis, can scale linearly along with 
nvidia-cuda-mps-control.

## Dependencies
* [Install Redis](https://redis.io/topics/quickstart)
* `pip3 install -r requriments.txt`

## Usage
* For Linear Scaling, start nvidia-cuda-mps-control
    ```bash
    nvidia-cuda-mps-control -d # To start
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