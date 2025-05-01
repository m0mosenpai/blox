import ray
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

@ray.remote
class InferenceActorClass:
    def __init__(self, model_name: str, max_batch_size: int = 8):
        engine_args = AsyncEngineArgs(model=model_name)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.max_batch_size = max_batch_size
        self.cancelled = set()
        print(f"[Actor] Loaded {model_name}, batch={max_batch_size}")

    async def generate(self, request_id: str, prompt: str,
                       params: dict = None, priority: str = "normal"):
        samp = SamplingParams(**(params or {}))
        prio = 1 if priority=="high" else 0
        async for output in self.engine.generate(prompt, samp, request_id, priority=prio):
            last = output
        if request_id in self.cancelled:
            self.cancelled.remove(request_id)
            return None
        return [last.prompt + o.text for o in last.outputs]

    def cancel_request(self, request_id: str):
        self.cancelled.add(request_id)
        return True
