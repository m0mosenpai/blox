import os
import sys
import grpc
import json
import logging
from typing import Dict, Any, Optional, Tuple
from concurrent import futures

# Import the generated stubs
sys.path.append(os.path.join(os.path.dirname(__file__), "grpc_stubs"))
import nm_pb2,nm_pb2_grpc, rm_pb2

# Type hint for NodeManagerMain to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from node_manager import NodeManagerMain

logger = logging.getLogger(__name__)

class NMServer(nm_pb2_grpc.NMServerServicer):
    """
    gRPC Server running on the Node Manager, handling requests from BloxManager.
    """
    def __init__(self):
        # This will be set by NodeManagerMain after instantiation
        self.node_manager: Optional['NodeManagerMain'] = None
        logger.info("NMServer initialized, waiting for NodeManagerMain reference.")

    def set_node_manager(self, node_manager_instance: 'NodeManagerMain'):
        """Allows NodeManagerMain to link itself to the server instance."""
        self.node_manager = node_manager_instance
        logger.info("NodeManagerMain reference set in NMServer.")

    def LaunchJob(self, request: nm_pb2.LaunchJobRequest, context) -> rm_pb2.BooleanResponse:
        """Handles inference job launch requests from the BloxManager."""
        if not self.node_manager:
            logger.error("LaunchJob called before NodeManagerMain reference was set!")
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Node Manager not fully initialized")
            return rm_pb2.BooleanResponse(value=False)

        job_id = request.job_id # Use string job_id
        logger.info(f"Received LaunchJob request for JobID: {job_id} (ReqID: {request.request_id})")

        # Deserialize sampling params (assuming JSON string values)
        try:
            sampling_params = {k: json.loads(v) for k, v in request.sampling_params.items()}
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode sampling_params for job {job_id}: {e}. Params: {request.sampling_params}")
             # Reject the job if params are invalid
             return rm_pb2.BooleanResponse(value=False) # NAK
        except Exception as e: # Catch other potential errors
             logger.error(f"Error processing sampling_params for job {job_id}: {e}", exc_info=True)
             return rm_pb2.BooleanResponse(value=False) # NAK


        # Prepare parameters for the local scheduler/node manager
        params = {
            "job_id": job_id, # Pass job_id within params too if LocalScheduler needs it
            "request_id": request.request_id,
            "prompt": request.prompt,
            "sampling_params": sampling_params, # The deserialized dict
            "priority": request.priority,
            # Add other details if needed by local scheduler/vLLM wrapper
        }

        # Pass the job to the NodeManagerMain's receive_job method
        # This method interacts with the LocalScheduler
        accepted = self.node_manager.receive_job(job_id, params)

        if accepted:
            logger.info(f"ACK LaunchJob for JobID: {job_id}")
            return rm_pb2.BooleanResponse(value=True)
        else:
            logger.warning(f"NAK LaunchJob for JobID: {job_id} (rejected by local scheduler)")
            return rm_pb2.BooleanResponse(value=False)


    def InitiateMigration(self, request: nm_pb2.MigrationRequest, context):
        """Handles request from BloxManager to start migrating a job *away* from this node."""
        if not self.node_manager or not self.node_manager.migration_manager:
            logger.error("InitiateMigration called before NodeManager or MigrationManager initialized!")
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Node Manager migration component not ready")
            return rm_pb2.BooleanResponse(value=False)

        job_id = request.job_id
        target_node_ip = request.target_node
        target_nm_address = request.target_address
        logger.info(f"Received InitiateMigration request for job {job_id} to target {target_node_ip} ({target_nm_address})")

        # Kick off the migration process in the background (non-blocking)
        # The MigrationManager handles the actual logic in a separate thread.
        self.node_manager.migration_manager.initiate_migration(job_id, target_node_ip, target_nm_address)

        # Return True immediately to acknowledge the request was received.
        # The actual success/failure is handled asynchronously.
        return rm_pb2.BooleanResponse(value=True)


    def ReceiveMigration(self, request: nm_pb2.MigrationState, context):
        """Handles receiving the state of a job being migrated *to* this node."""
        if not self.node_manager or not self.node_manager.migration_manager:
            logger.error("ReceiveMigration called before NodeManager or MigrationManager initialized!")
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Node Manager migration component not ready")
            return rm_pb2.BooleanResponse(value=False)

        job_id = request.job_id
        state_blob = request.state_blob
        logger.info(f"Received ReceiveMigration request for job {job_id} (state size: {len(state_blob)} bytes)")

        # Attempt to resume the job using the received state (blocking call here)
        # MigrationManager handles interaction with LocalScheduler and VLLMWrapper
        success = self.node_manager.migration_manager.receive_migration(job_id, state_blob)

        if success:
            logger.info(f"Successfully accepted and resumed migrated job {job_id}.")
            return rm_pb2.BooleanResponse(value=True) # ACK
        else:
            logger.error(f"Failed to accept/resume migrated job {job_id}.")
            return rm_pb2.BooleanResponse(value=False) # NAK


# Function to start the server (called from node_manager.py)
def start_server(port: int) -> Tuple[grpc.Server, NMServer]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nm_servicer = NMServer() # Instantiate the servicer
    nm_pb2_grpc.add_NMServerServicer_to_server(nm_servicer, server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    server.start()
    logger.info(f"Node Manager gRPC server started listening on {listen_addr}")
    # Return both server and servicer instance so NodeManagerMain can link to it
    return server, nm_servicer