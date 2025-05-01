import os
import sys
import grpc
import json
import uuid
import logging
from typing import Dict, Any, Tuple, Optional, List

# Import the generated stubs
sys.path.append(os.path.join(os.path.dirname(__file__), "grpc_stubs"))
import rm_pb2, rm_pb2_grpc

logger = logging.getLogger(__name__)

class ResourceManagerComm(object):
    """gRPC Client used by Node Managers and Frontend to communicate with BloxManager."""

    def __init__(self, rm_address: str): # Address of the RMServer (BloxManager)
        self.rm_addr = rm_address
        logger.info(f"ResourceManagerComm initialized for RM address: {self.rm_addr}")

    # --- Methods used by NodeManager ---

    def register_with_scheduler(
        self,
        node_ipaddr: str,
        node_interface: str,
        num_gpus: int,
        node_manager_port: int,
        available_kv_memory_gb: int = 0, # Optional reporting
        supported_models: Optional[List[str]] = None
    ) -> bool:
        """Register the node manager with the central scheduler (BloxManager)."""
        models_to_send = supported_models if supported_models else []
        request = rm_pb2.RegisterRequest(
            ipaddr=node_ipaddr,
            interface=node_interface,
            num_gpus=num_gpus,
            node_manager_port=node_manager_port, # Tell RM how to reach back
            available_kv_memory_gb=available_kv_memory_gb,
            supported_models=models_to_send
        )
        logger.debug(f"Sending RegisterWorker request for node {node_ipaddr} to {self.rm_addr}")
        try:
            with grpc.insecure_channel(self.rm_addr) as channel:
                stub = rm_pb2_grpc.RMServerStub(channel)
                response = stub.RegisterWorker(request, timeout=10)
            logger.info(f"RegisterWorker response from {self.rm_addr}: Success={response.value}")
            return response.value
        except grpc.RpcError as e:
            logger.error(f"gRPC error registering worker {node_ipaddr} with {self.rm_addr}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error registering worker {node_ipaddr}: {e}", exc_info=True)
            return False

    def notify_job_done(self, job_id: str, success: bool) -> bool:
        """Tell the scheduler (BloxManager) that an inference job has finished."""
        req = rm_pb2.JobDoneRequest(job_id=job_id, success=success)
        logger.debug(f"Sending JobCompleted notification for job {job_id} to {self.rm_addr} (Success: {success})")
        try:
            with grpc.insecure_channel(self.rm_addr) as ch:
                stub = rm_pb2_grpc.RMServerStub(ch)
                resp = stub.JobCompleted(req, timeout=10)
            # RMServer should always return true if it received the message
            return resp.value
        except grpc.RpcError as e:
            logger.error(f"gRPC error sending job completion for {job_id} to {self.rm_addr}: {e}")
            return False # Indicate notification failed
        except Exception as e:
             logger.error(f"Unexpected error sending job completion for {job_id}: {e}", exc_info=True)
             return False

    def migration_complete(self, report: rm_pb2.MigrationReport) -> bool:
        """Tell the scheduler (BloxManager) that a migration finished."""
        logger.debug(f"Sending MigrationComplete report for job {report.job_id} to {self.rm_addr}")
        try:
            with grpc.insecure_channel(self.rm_addr) as ch:
                 stub = rm_pb2_grpc.RMServerStub(ch)
                 resp = stub.MigrationComplete(report, timeout=10)
            return resp.value
        except grpc.RpcError as e:
            logger.error(f"gRPC error sending migration completion for {report.job_id}: {e}")
            return False
        except Exception as e:
             logger.error(f"Unexpected error sending migration completion for {report.job_id}: {e}", exc_info=True)
             return False


    # --- Methods used by Frontend ---

    def submit_inference(self,
                         prompt: str,
                         sampling_params: Dict[str, Any],
                         priority: str = "normal",
                         target_model: Optional[str] = None,
                         max_latency_ms: float = 0.0 # 0 means no specific SLO
                        ) -> Tuple[bool, Optional[str], str]:
        """
        Submits an inference request from the frontend to the BloxManager.
        Returns (accepted: bool, job_id: Optional[str], message: str)
        """
        # Generate a unique ID for this specific request attempt
        request_id = str(uuid.uuid4())
        # Serialize sampling params
        sampling_params_str_map = {str(k): json.dumps(v) for k, v in sampling_params.items()}

        req = rm_pb2.InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params_str_map,
            priority=priority,
            target_model=target_model if target_model else "",
            max_latency_ms=max_latency_ms
        )
        logger.debug(f"Sending SubmitInferenceRequest (ReqID: {request_id}) to {self.rm_addr}")
        try:
            with grpc.insecure_channel(self.rm_addr) as ch:
                stub = rm_pb2_grpc.RMServerStub(ch)
                resp = stub.SubmitInferenceRequest(req, timeout=10)
            logger.info(f"SubmitInferenceRequest response for ReqID {request_id}: Accepted={resp.accepted}, JobID={resp.job_id}, Msg={resp.message}")
            # Return acceptance status, the Blox job_id assigned, and any message
            return resp.accepted, resp.job_id if resp.accepted else None, resp.message
        except grpc.RpcError as e:
            logger.error(f"gRPC error submitting inference request {request_id} to {self.rm_addr}: {e}")
            return False, None, f"RPC Error: {e.details()}"
        except Exception as e:
            logger.error(f"Unexpected error submitting inference request {request_id}: {e}", exc_info=True)
            return False, None, f"Client-side Error: {e}"