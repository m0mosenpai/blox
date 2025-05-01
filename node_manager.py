import time
import grpc
import argparse
import ray
import logging
import asyncio # For handling async vLLM calls
from concurrent import futures
from typing import Tuple, Dict, Optional, Any

# Blox imports
import blox.deployment.grpc_server_nm as nm_serve # Use the updated server start function
import blox.deployment.grpc_client_rm as rm_client # Use the updated client
from local_scheduler import get_local_scheduler, LocalScheduler # Use the new local scheduler
from migration_manager import MigrationManager # Use the new migration manager
from vllm_wrapper import VLLMWrapper # Use the new vLLM wrapper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NodeManagerMain(object):
    """
    Main node manager class orchestrating local scheduling, vLLM interaction, and migration.
    """
    def __init__(
        self,
        ipaddr: str, # IP of this node
        node_manager_port: int, # Port this NM server listens on
        central_scheduler_addr: str, # Address (ip:port) of BloxManager RMServer
        model_name: str, # Model for vLLM
        num_gpus: int, # GPUs for vLLM tensor parallel
        local_scheduler_mode: str,
        max_queue_size: int,
        simulate: bool = False, # If true, skips Ray/vLLM init
        ray_address: Optional[str] = None
    ) -> None:
        self.ipaddr = ipaddr
        self.node_manager_port = node_manager_port
        self.central_scheduler_addr = central_scheduler_addr
        self.simulate = simulate
        self.num_gpus = num_gpus

        # 1. Communicator to talk back to BloxManager (RMServer)
        self.node_manager_comm = rm_client.ResourceManagerComm(self.central_scheduler_addr)

        # 2. vLLM Backend Wrapper (Initialize before Ray if needed by VLLMWrapper internals)
        self.vllm_backend: Optional[VLLMWrapper] = None
        if not self.simulate:
            # Initialize Ray connection first if VLLMWrapper needs it (e.g., for worker_use_ray=True)
            if not ray.is_initialized():
                 try:
                      logger.info(f"Connecting NodeManager to Ray at address: {ray_address or 'auto'}")
                      ray.init(address=ray_address, ignore_reinit_error=True, logging_level=logging.WARNING)
                 except Exception as e:
                      logger.error(f"Failed to connect to Ray: {e}", exc_info=True)
                      raise RuntimeError("Ray connection failed") from e
            logger.info(f"NodeManager connected to Ray at {ray.get_runtime_context().ray_address}")

            # Now initialize VLLMWrapper which might use Ray
            try:
                 # Pass relevant vLLM engine args here if needed
                 self.vllm_backend = VLLMWrapper(model=model_name, num_gpus=num_gpus)
            except Exception as e:
                 logger.error(f"Failed to initialize VLLMWrapper: {e}", exc_info=True)
                 # Should we attempt to proceed without vLLM? Probably not.
                 raise RuntimeError("VLLMWrapper initialization failed") from e
        else:
             logger.info("Simulation mode: Skipping Ray and vLLM initialization.")

        # 3. Local Scheduler
        self.local_scheduler: LocalScheduler = get_local_scheduler(
            mode=local_scheduler_mode,
            node_id=self.ipaddr,
            num_gpus=self.num_gpus,
            max_queue_size=max_queue_size
        )
        # Set the callback for actually launching jobs
        self.local_scheduler.set_launch_callback(self._launch_inference)

        # 4. Migration Manager (Needs refs to scheduler & backend)
        self.migration_manager: Optional[MigrationManager] = None
        if not self.simulate and self.vllm_backend:
            self.migration_manager = MigrationManager(
                node_id=self.ipaddr,
                local_scheduler=self.local_scheduler,
                vllm_backend=self.vllm_backend,
                rm_communicator=self.node_manager_comm # Pass RM client for notifications
            )
        elif not self.simulate:
             logger.warning("Cannot initialize MigrationManager because VLLMWrapper failed.")


        # 5. State for tracking running jobs (async tasks/futures from vLLM)
        # Key: job_id (str), Value: asyncio Task or Future from VLLMWrapper
        self.running_infer_tasks: Dict[str, asyncio.Task] = {}
        self.polling_interval_sec = 0.1 # How often to check futures


    async def poll_inference_jobs(self):
        """
        Periodically checks the status of running inference tasks managed by VLLMWrapper.
        MUST be run within an asyncio event loop.
        """
        if self.simulate or not self.running_infer_tasks:
            return # Nothing to poll

        done_job_ids = []
        # Iterate over a copy of items to allow modification during iteration
        for job_id, task in list(self.running_infer_tasks.items()):
            if task.done():
                success = False
                try:
                    # Get result or exception
                    result = task.result() # This will raise exception if task failed
                    # Define success criteria based on VLLMWrapper's return value
                    # Assuming it returns the vLLM RequestOutput or similar on success, None/Exception on failure
                    if result is not None and hasattr(result, 'finished') and result.finished:
                         logger.info(f"Inference job {job_id} completed successfully.")
                         # logger.debug(f"Job {job_id} output: {result.outputs[0].text[:100]}...") # Log snippet
                         success = True
                    else:
                         # Task finished but didn't return expected success indicator
                         logger.warning(f"Inference job {job_id} finished with unexpected result: {result}")
                         success = False # Treat as failure

                except asyncio.CancelledError:
                    logger.info(f"Inference job {job_id} was cancelled.")
                    success = False # Treat cancellation as non-success
                except Exception as e:
                    logger.error(f"Inference job {job_id} failed with exception: {e}", exc_info=False) # Avoid overly verbose logs
                    success = False # Treat exception as failure

                # Mark for cleanup and notify components
                done_job_ids.append(job_id)
                self.job_done(job_id, success)

        # Clean up completed tasks from tracking dict
        for job_id in done_job_ids:
            if job_id in self.running_infer_tasks:
                 del self.running_infer_tasks[job_id]


    def register_with_scheduler(self, interface: str) -> None:
        """Registers this node with the central BloxManager."""
        logger.info(f"Attempting to register with BloxManager at {self.central_scheduler_addr}")
        registered = False
        while not registered: # Retry loop
             registered = self.node_manager_comm.register_with_scheduler(
                  node_ipaddr=self.ipaddr,
                  node_interface=interface,
                  num_gpus=self.num_gpus,
                  node_manager_port=self.node_manager_port, # Tell RM how to reach back
                  # Add capability reporting if needed/implemented
                  # available_kv_memory_gb=...,
                  # supported_models=[self.vllm_backend.model_name],
             )
             if not registered:
                  logger.warning("Failed to register with scheduler. Retrying in 5 seconds...")
                  time.sleep(5)
             else:
                  logger.info("Successfully registered with scheduler.")


    def receive_job(self, job_id: str, launch_params: dict) -> bool:
        """
        Called by NMServer.LaunchJob. Hands off the job to the LocalScheduler.
        Returns True (ACK) if accepted locally, False (NAK) otherwise.
        """
        if self.simulate:
            logger.info(f"Simulate mode: Accepting job {job_id} locally.")
            # In simulation, we might just trigger completion after a delay
            return True

        if not self.local_scheduler:
            logger.error("Cannot receive job: LocalScheduler not initialized.")
            return False

        # The LocalScheduler's submit_request handles queuing and immediate scheduling attempts
        return self.local_scheduler.submit_request(launch_params)


    def _launch_inference(self, job_id: str, job_info: dict, gpu_id: int):
        """
        Callback function used by LocalScheduler.
        Submits the request to the VLLMWrapper and starts tracking the task.
        This function MUST run within the asyncio event loop.
        """
        if self.simulate or not self.vllm_backend:
            logger.warning(f"Skipping actual launch of job {job_id} (Simulate mode or no VLLM backend)")
            # TODO: Add simulation logic if needed (e.g., schedule a simulated completion event)
            return

        logger.info(f"Launching inference job {job_id} on assigned GPU {gpu_id} via VLLMWrapper.")
        try:
            # Extract necessary parameters
            prompt = job_info["prompt"]
            sampling_params = job_info.get("sampling_params", {})
            request_id = job_info.get("request_id", job_id) # Use original request ID if available

            # Ensure this runs in the event loop (NodeManager's main loop should manage this)
            # VLLMWrapper.generate should be an async function returning an asyncio Task
            async def run_generation():
                 task = await self.vllm_backend.generate(
                      request_id=job_id, # Use Blox job ID as the unique ID for vLLM tracking?
                      prompt=prompt,
                      sampling_params_dict=sampling_params
                 )
                 if task:
                     self.running_infer_tasks[job_id] = task
                     logger.debug(f"Started tracking asyncio task for job {job_id}")
                 else:
                      logger.error(f"VLLMWrapper failed to return a task for job {job_id}. Marking as failed.")
                      # Job failed to even start properly
                      self.job_done(job_id, False) # Notify failure

            # Schedule the async function to run in the loop
            asyncio.create_task(run_generation())

        except Exception as e:
            logger.error(f"Failed to dispatch job {job_id} to VLLMWrapper: {e}", exc_info=True)
            # If launch fails, treat job as failed immediately
            self.job_done(job_id, False)


    def job_done(self, job_id: str, success: bool):
        """
        Called internally when a job finishes (polled via poll_inference_jobs) or fails launch.
        Notifies the global scheduler (RM) and tells the local scheduler to free resources.
        """
        logger.info(f"Processing completion for job {job_id}, Success: {success}")

        # 1. Notify global scheduler (BloxManager/RMServer)
        notified = self.node_manager_comm.notify_job_done(job_id, success)
        if not notified:
            logger.warning(f"Failed to notify BloxManager about job {job_id} completion.")
            # TODO: Implement retry logic for RM notification?

        # 2. Update local scheduler (frees GPU, triggers schedule_next)
        if self.local_scheduler:
            self.local_scheduler.on_job_complete(job_id)
        else:
             logger.error("Cannot process job completion: LocalScheduler not initialized.")

        # Note: Cleanup of self.running_infer_tasks happens in poll_inference_jobs


    async def main_loop(self):
        """The main async loop for polling job statuses."""
        logger.info("Node Manager main async loop started.")
        while True:
             await self.poll_inference_jobs()
             await asyncio.sleep(self.polling_interval_sec)


# --- Argument Parsing ---
def parse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--ipaddr", required=True, help="IP address of this node manager")
    parser.add_argument("--interface", default="eth0", help="Network interface for registration")
    parser.add_argument("--node-manager-port", type=int, default=50051, help="Port for this Node Manager's gRPC server")
    parser.add_argument("--central-scheduler-addr", required=True, help="Address (ip:port) of the central BloxManager RMServer")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode (no Ray/vLLM)")

    # Ray and vLLM args
    parser.add_argument("--ray-address", type=str, default=None, help="Address of the Ray head node (e.g., 'ray://<ip>:10001' or 'auto')")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", help="HuggingFace model name or path for vLLM")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for vLLM Tensor Parallelism on this node")
    # Add other vLLM engine args if needed (e.g., --gpu-memory-utilization)

    # Local Scheduler args
    parser.add_argument("--local-scheduler-mode", choices=["fifo"], default="fifo", help="Local scheduling policy") # Add more choices later
    # --node-memory arg removed, using num_gpus and vLLM's own management mostly
    parser.add_argument("--max-queue-size", type=int, default=32, help="Maximum local job queue size")

    args = parser.parse_args()
    return args

# --- Main Execution ---
def main(args):
    """Initializes and runs the Node Manager."""
    try:
        node_manager_main = NodeManagerMain(
            ipaddr=args.ipaddr,
            node_manager_port=args.node_manager_port,
            central_scheduler_addr=args.central_scheduler_addr,
            model_name=args.model,
            num_gpus=args.num_gpus,
            local_scheduler_mode=args.local_scheduler_mode,
            max_queue_size=args.max_queue_size,
            simulate=args.simulate,
            ray_address=args.ray_address
        )
    except Exception as e:
        logger.error(f"Failed to initialize NodeManagerMain: {e}", exc_info=True)
        return # Exit if basic initialization fails

    # Start gRPC server (NMServer)
    grpc_server, nm_servicer_instance = nm_serve.start_server(args.node_manager_port)
    # Link the running server instance to the NodeManagerMain logic controller
    nm_servicer_instance.set_node_manager(node_manager_main)

    # Register with BloxManager (retry loop inside)
    node_manager_main.register_with_scheduler(args.interface)

    # Start the main async loop for polling vLLM tasks
    logger.info("Starting Node Manager async task polling loop.")
    loop = asyncio.get_event_loop()
    try:
        # Run the main polling loop
        loop.run_until_complete(node_manager_main.main_loop())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down Node Manager...")
    except Exception as e:
        logger.error(f"Node Manager main loop encountered an error: {e}", exc_info=True)
    finally:
        logger.info("Stopping gRPC server...")
        grpc_server.stop(grace=1).wait() # Wait for graceful shutdown
        logger.info("gRPC server stopped.")
        if not args.simulate and ray.is_initialized():
            logger.info("Shutting down Ray connection...")
            ray.shutdown()
            logger.info("Ray connection shut down.")
        # Close the asyncio loop? Only if we started it here.
        # loop.close()
        logger.info("Node Manager shutdown complete.")


if __name__ == "__main__":
    args = parse_args(
        argparse.ArgumentParser(description="Blox Node Manager for Inference")
    )
    main(args)