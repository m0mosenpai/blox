import os
import sys
import grpc
import json
import uuid
import logging
import time
from concurrent import futures
from typing import TYPE_CHECKING

sys.path.append(os.path.join(os.path.dirname(__file__), "grpc_stubs"))
import rm_pb2,rm_pb2_grpc

if TYPE_CHECKING:
    from blox.blox_manager import BloxManager # Avoid circular import


logger = logging.getLogger(__name__)

class RMServer(rm_pb2_grpc.RMServerServicer):
    """
    gRPC Server running with the BloxManager, handling requests from Node Managers and Frontend.
    """
    def __init__(self, blox_manager_instance: 'BloxManager'):
        self.blox_manager = blox_manager_instance
        logger.info("RMServer initialized.")

    def RegisterWorker(self, request: rm_pb2.RegisterRequest, context) -> rm_pb2.BooleanResponse:
        """Handles worker registration calls from Node Managers."""
        ipaddr = request.ipaddr
        interface = request.interface
        num_gpus = request.num_gpus
        kv_memory = request.available_kv_memory_gb
        models = list(request.supported_models)
        nm_port = request.node_manager_port

        logger.info(f"Received RegisterWorker request from {ipaddr}:{nm_port} (Interface: {interface}, GPUs: {num_gpus})")

        # Pass registration details to BloxManager
        success = self.blox_manager.register_worker(
            ipaddr=ipaddr,
            interface=interface,
            num_gpus=num_gpus,
            nm_port=nm_port,
            available_kv_memory=kv_memory,
            supported_models=models
        )
        return rm_pb2.BooleanResponse(value=success)

    def JobCompleted(self, request: rm_pb2.JobDoneRequest, context) -> rm_pb2.BooleanResponse:
        """Handles job completion notifications from Node Managers."""
        job_id = request.job_id
        success = request.success
        logger.info(f"Received JobCompleted notification for JobID: {job_id}, Success: {success}")
        # Notify BloxManager to update job state
        self.blox_manager.handle_job_completion(job_id, success)
        # Acknowledge receipt
        return rm_pb2.BooleanResponse(value=True)

    def MigrationComplete(self, request: rm_pb2.MigrationReport, context) -> rm_pb2.BooleanResponse:
        """Handles migration completion notifications from the *source* Node Manager."""
        job_id = request.job_id
        from_node = request.from_node
        to_node = request.to_node
        logger.info(f"Received MigrationComplete report for JobID: {job_id} (From: {from_node}, To: {to_node})")
        # Notify BloxManager to update job location/state
        self.blox_manager.handle_migration_completion(job_id, from_node, to_node)
        # Acknowledge receipt
        return rm_pb2.BooleanResponse(value=True)

    def SubmitInferenceRequest(self, request: rm_pb2.InferenceRequest, context) -> rm_pb2.InferenceResponse:
        """Handles new inference requests submitted from the frontend."""
        req_id = request.request_id
        logger.info(f"Received SubmitInferenceRequest from frontend (ReqID: {req_id})")

        # Deserialize sampling params (expecting JSON strings)
        try:
            sampling_params = {k: json.loads(v) for k, v in request.sampling_params.items()}
        except Exception as e:
             logger.error(f"Failed to decode sampling_params for ReqID {req_id}: {e}")
             return rm_pb2.InferenceResponse(job_id="", accepted=False, message=f"Invalid sampling_params: {e}")

        # Prepare job details dictionary
        job_details = {
            "request_id": req_id,
            # Job ID will be assigned by BloxManager
            "prompt": request.prompt,
            "sampling_params": sampling_params, # Deserialized dict
            "priority": request.priority,
            "target_model": request.target_model,
            "max_latency_ms": request.max_latency_ms,
            "status": "pending",
            "submit_time": time.time(),
            "command_to_run": "inference" # Mark type
        }

        # Pass to BloxManager's queuing mechanism
        # This method should assign a job_id and return acceptance status
        accepted, assigned_job_id, message = self.blox_manager.add_inference_request(job_details)

        if accepted:
            logger.info(f"Accepted inference request {req_id} as JobID: {assigned_job_id}")
            return rm_pb2.InferenceResponse(job_id=assigned_job_id, accepted=True, message=message or "Request queued")
        else:
            logger.warning(f"Rejected inference request {req_id}: {message}")
            return rm_pb2.InferenceResponse(job_id="", accepted=False, message=message or "Request rejected")


# Function to start the server (called from blox_manager.py)
def start_server(blox_manager_instance: 'BloxManager', port: int) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20)) # Allow more workers for scheduler
    rm_servicer = RMServer(blox_manager_instance) # Instantiate the servicer
    rm_pb2_grpc.add_RMServerServicer_to_server(rm_servicer, server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    server.start()
    logger.info(f"BloxManager RMServer started listening on {listen_addr}")
    return server