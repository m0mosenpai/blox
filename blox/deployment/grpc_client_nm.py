# blox/deployment/grpc_client_nm.py

import os
import sys
import grpc
import json
import logging
from typing import Dict, List, Tuple, Any

# Import the generated stubs for the Node Manager service
sys.path.append(os.path.join(os.path.dirname(__file__), "grpc_stubs"))
import nm_pb2,nm_pb2_grpc, rm_pb2

logger = logging.getLogger(__name__)

class NodeManagerComm(object):
    """
    gRPC Client used by BloxManager to send commands to Node Managers (NMServer).
    Each method targets a specific Node Manager address.
    """

    # No __init__ needed, target address is provided per RPC call.

    def launch_job(
        self,
        target_nm_address: str, # e.g., "10.0.0.5:50051"
        job_id: str, # Blox Job ID (string)
        request_id: str, # Original frontend request ID (string)
        prompt: str,
        sampling_params: Dict[str, Any], # Will be serialized to map<string, string>
        priority: str
    ) -> bool:
        """
        Sends LaunchJob RPC to a specific Node Manager to start an inference job.

        Args:
            target_nm_address: The "ip:port" address of the target Node Manager's gRPC server.
            job_id: The unique ID assigned by BloxManager for this job.
            request_id: The unique ID from the original frontend request.
            prompt: The input prompt text.
            sampling_params: Dictionary of sampling parameters. Values will be JSON encoded.
            priority: Job priority string (e.g., "high", "normal").

        Returns:
            bool: True if the Node Manager ACKed the job (accepted it), False otherwise (NAKed or RPC error).
        """
        # Serialize complex sampling_params values to JSON strings
        try:
            sampling_params_str_map = {str(k): json.dumps(v) for k, v in sampling_params.items()}
        except TypeError as e:
            logger.error(f"Failed to JSON-serialize sampling_params for job {job_id}: {e} - Params: {sampling_params}")
            return False # Cannot proceed if params are not serializable

        request = nm_pb2.LaunchJobRequest(
            job_id=job_id,
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params_str_map,
            priority=priority,
        )

        logger.debug(f"Sending LaunchJob request for job {job_id} to {target_nm_address}")
        try:
            # Create a temporary channel for this specific call
            with grpc.insecure_channel(target_nm_address) as channel:
                stub = nm_pb2_grpc.NMServerStub(channel)
                # Set a reasonable timeout (e.g., 10 seconds)
                response = stub.LaunchJob(request, timeout=10)
                logger.info(f"LaunchJob response for job {job_id} from {target_nm_address}: ACK={response.value}")
                # Return the boolean ACK/NAK status from the Node Manager
                return response.value
        except grpc.RpcError as e:
            # Log specific gRPC errors (e.g., UNAVAILABLE, DEADLINE_EXCEEDED)
            status_code = e.code()
            logger.error(f"gRPC error launching job {job_id} on {target_nm_address}: Status={status_code}, Details={e.details()}")
            return False # Treat RPC error as NAK
        except Exception as e:
            # Catch other potential errors during the call
            logger.error(f"Unexpected error launching job {job_id} on {target_nm_address}: {e}", exc_info=True)
            return False

    def initiate_migration(
        self,
        source_nm_address: str, # Address of the NM currently holding the job
        job_id: str,            # Job to migrate
        target_node_ip: str,    # IP address of the destination node
        target_node_nm_address: str # Full "ip:port" of the destination NM server
        ) -> bool:
        """
        Sends InitiateMigration RPC to the source Node Manager to start migrating a job away.

        Args:
            source_nm_address: "ip:port" of the Node Manager currently running the job.
            job_id: The ID of the job to migrate.
            target_node_ip: The IP address of the node where the job should be migrated to.
            target_node_nm_address: The full "ip:port" of the Node Manager server on the target node.

        Returns:
            bool: True if the source Node Manager acknowledged the migration initiation request, False otherwise.
                  Note: This only acknowledges the request; actual migration happens asynchronously.
        """
        request = nm_pb2.MigrationRequest(
            job_id=job_id,
            target_node=target_node_ip,
            target_address=target_node_nm_address
        )
        logger.debug(f"Sending InitiateMigration for job {job_id} to source node {source_nm_address} (Target: {target_node_nm_address})")
        try:
            with grpc.insecure_channel(source_nm_address) as channel:
                stub = nm_pb2_grpc.NMServerStub(channel)
                # Timeout for acknowledging the initiation request
                response = stub.InitiateMigration(request, timeout=10)
                logger.info(f"InitiateMigration response for job {job_id} from {source_nm_address}: Success={response.value}")
                # Return whether the source node accepted the request to start migrating
                return response.value
        except grpc.RpcError as e:
            status_code = e.code()
            logger.error(f"gRPC error initiating migration for job {job_id} on {source_nm_address}: Status={status_code}, Details={e.details()}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error initiating migration for job {job_id} on {source_nm_address}: {e}", exc_info=True)
            return False

    def notify_terminate(self, target_nm_address: str, job_id: str) -> bool:
        """
        Sends NotifyTerminate RPC to a specific Node Manager to request cancellation
        of a running inference job.

        Args:
            target_nm_address: The "ip:port" address of the Node Manager running the job.
            job_id: The ID of the job to terminate/cancel.

        Returns:
            bool: True if the Node Manager acknowledged the termination request, False otherwise.
                  Note: This only acknowledges the request; actual cancellation might take time.
        """
        # Ensure you have defined TerminateJobRequest in your nm.proto
        request = nm_pb2.TerminateJobRequest(job_id=job_id)

        logger.debug(f"Sending NotifyTerminate request for job {job_id} to {target_nm_address}")
        try:
            with grpc.insecure_channel(target_nm_address) as channel:
                stub = nm_pb2_grpc.NMServerStub(channel)
                # Timeout for acknowledging the termination request
                response = stub.NotifyTerminate(request, timeout=10)
                logger.info(f"NotifyTerminate response for job {job_id} from {target_nm_address}: Acknowledged={response.value}")
                # Return whether the node acknowledged the cancellation request
                return response.value
        except grpc.RpcError as e:
            status_code = e.code()
            # Don't log error if the job is already gone (NOT_FOUND)
            if status_code == grpc.StatusCode.NOT_FOUND:
                 logger.warning(f"NotifyTerminate for job {job_id} failed on {target_nm_address}: Job not found (likely already completed).")
                 return False # Or True, as the goal (job gone) is achieved? False is safer.
            else:
                 logger.error(f"gRPC error terminating job {job_id} on {target_nm_address}: Status={status_code}, Details={e.details()}")
                 return False
        except Exception as e:
            logger.error(f"Unexpected error terminating job {job_id} on {target_nm_address}: {e}", exc_info=True)
            return False

    # --- Other potential methods (if needed) ---
    # e.g., querying node status, updating leases (if using training features)