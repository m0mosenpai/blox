import os
import sys
import threading
import grpc
import logging
from typing import TYPE_CHECKING
from vllm_wrapper import VLLMWrapper
from local_scheduler import LocalScheduler
from blox.deployment.grpc_client_rm import ResourceManagerComm

sys.path.append(os.path.join(os.path.dirname(__file__), "./deployment/grpc_stubs"))
from blox.deployment.grpc_stubs import nm_pb2, nm_pb2_grpc, rm_pb2, rm_pb2_grpc


logger = logging.getLogger(__name__)

class MigrationManager:
    """Handles live migration of a single inference job."""
    def __init__(self,
                 node_id: str, # IP address of this node
                 local_scheduler: 'LocalScheduler',
                 vllm_backend: 'VLLMWrapper',
                 rm_communicator: 'ResourceManagerComm'): # To notify RM on completion
        self.node_id = node_id
        self.local_scheduler = local_scheduler
        self.vllm = vllm_backend
        self.rm_comm = rm_communicator # Client to talk to RMServer

    def initiate_migration(self, job_id: str, target_node_ip: str, target_nm_address: str):
        """
        Starts migration process: get state, send to target, cleanup locally.
        Runs in a background thread to avoid blocking the NMServer.
        """
        logger.info(f"Initiating migration of job {job_id} from {self.node_id} to {target_node_ip} ({target_nm_address})")

        def _do_migration():
            success = False
            try:
                # 1. Get state from local vLLM (blocking/async call)
                logger.debug(f"Getting state for job {job_id}...")
                # This needs to be implemented in VLLMWrapper and vLLM
                state_blob = self.vllm.get_request_state(job_id)
                if state_blob is None:
                    logger.error(f"Failed to get migration state for job {job_id}. Aborting migration.")
                    return # Cannot proceed without state

                logger.debug(f"Got state for job {job_id} (size: {len(state_blob)} bytes). Sending to {target_nm_address}")

                # 2. Send state to target Node Manager via gRPC
                req = nm_pb2.MigrationState(job_id=job_id, state_blob=state_blob)
                try:
                    with grpc.insecure_channel(target_nm_address) as channel:
                        stub = nm_pb2_grpc.NMServerStub(channel)
                        # Consider adding a timeout
                        resp = stub.ReceiveMigration(req, timeout=30) # 30 second timeout
                        target_accepted = resp.value
                except grpc.RpcError as e:
                    logger.error(f"gRPC error sending migration state for job {job_id} to {target_nm_address}: {e}")
                    target_accepted = False

                if target_accepted:
                    logger.info(f"Target node {target_node_ip} successfully received state for job {job_id}.")
                    # 3. If target accepted, cancel/cleanup locally
                    logger.debug(f"Cancelling local job {job_id} after successful state transfer.")
                    self.vllm.cancel_request(job_id) # Tell vLLM to stop processing
                    # Free up local resources via the scheduler
                    self.local_scheduler.on_job_complete(job_id)
                    success = True
                else:
                    logger.error(f"Target node {target_node_ip} failed to receive/accept state for job {job_id}. Migration failed.")
                    # Job remains running locally? Or should be marked as failed?
                    # For now, assume it continues locally if target fails.

            except Exception as e:
                logger.error(f"Unexpected error during migration initiation for job {job_id}: {e}", exc_info=True)
                success = False # Ensure failure is recorded

            finally:
                # 4. Notify ResourceManager about the outcome (only if successful?)
                if success:
                    logger.info(f"Notifying RM about successful migration completion of job {job_id} to {target_node_ip}.")
                    report = rm_pb2.MigrationReport(job_id=job_id, from_node=self.node_id, to_node=target_node_ip)
                    # This requires MigrationComplete RPC in RMServer
                    if hasattr(self.rm_comm, "migration_complete"): # Check if method exists
                         self.rm_comm.migration_complete(report)
                    else:
                         logger.warning("RM client does not have migration_complete method.")
                else:
                     logger.error(f"Migration failed for job {job_id}. No RM notification sent.")


        # Run the actual migration logic in a background thread
        thread = threading.Thread(target=_do_migration, daemon=True)
        thread.start()

    def receive_migration(self, job_id: str, state_blob: bytes) -> bool:
        """
        Called on the destination node by gRPC (ReceiveMigration).
        Resumes the job using the received state.
        """
        logger.info(f"Receiving migration state for job {job_id} (state size: {len(state_blob)} bytes).")
        try:
            # 1. Check if we can actually run this job (e.g., have free resources)
            # Simple check: are any GPUs free? More complex check might be needed.
            if not self.local_scheduler.free_gpus:
                logger.error(f"Cannot receive migration for job {job_id}: No free GPUs available locally.")
                return False # NAK - Cannot accept

            # 2. Allocate a GPU via the local scheduler
            # This bypasses the queue, directly assigning to a free GPU
            gpu_id = self.local_scheduler.free_gpus.pop()
            self.local_scheduler.running[job_id] = gpu_id # Mark as running locally
            logger.info(f"Allocated GPU {gpu_id} for incoming migrated job {job_id}.")

            # 3. Resume in vLLM using the state blob
            logger.debug(f"Resuming job {job_id} on GPU {gpu_id} using received state...")
            # This needs to be implemented in VLLMWrapper and vLLM
            resumed_ok = self.vllm.resume_request(job_id, state_blob, gpu_id=gpu_id)

            if resumed_ok:
                logger.info(f"Successfully resumed migrated job {job_id} on GPU {gpu_id}.")
                # Need to start polling this job's future now
                # This assumes resume_request returns a future or handles tracking internally
                # NodeManagerMain needs a way to add this job_id to its self.running_infer polling dict
                # We might need a callback here too, or return the future
                # TODO: Connect this back to NodeManagerMain's polling loop
                return True # ACK - Success
            else:
                logger.error(f"Failed to resume job {job_id} in vLLM backend.")
                # Rollback: Free the GPU and remove from running state
                self.local_scheduler.free_gpus.add(gpu_id)
                del self.local_scheduler.running[job_id]
                return False # NAK - Failed to resume

        except Exception as e:
            logger.error(f"Error during migration reception for job {job_id}: {e}", exc_info=True)
            # Clean up potential partial allocation
            if job_id in self.local_scheduler.running:
                 gpu_id = self.local_scheduler.running.pop(job_id)
                 self.local_scheduler.free_gpus.add(gpu_id)
            return False # NAK - Error
