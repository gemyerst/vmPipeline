import json
import logging
import queue
import subprocess
import time
from pathlib import Path
from typing import Final, Optional, Set, Sequence, Any
from concurrent.futures import ThreadPoolExecutor
import io


import typer
from google.auth.credentials import Credentials
from google.cloud.pubsub_v1 import SubscriberClient
from google.cloud.pubsub_v1.subscriber import futures
from google.cloud.pubsub_v1.subscriber.message import Message
from google.cloud.storage import Blob
from google.cloud.storage import Client as StorageClient
from google.oauth2 import service_account

from loggers import init_logging
from datamodels import PipelineRequest
from typealiases import Predicate, Consumer


LOGGER: Final[logging.Logger] = logging.getLogger()
LOGGER.setLevel(logging.INFO)


def initiate_message_pulling(
    subscriber: SubscriberClient, 
    consumer: Consumer[PipelineRequest], 
    project_id: str,
    subscription: str) -> futures.StreamingPullFuture:

    subscription_name = f'projects/{project_id}/subscriptions/{subscription}'
    
    LOGGER.info(f"PubSub | Attempting to subscribe to {subscription_name}")

    def callback(message: Message):
        json_data: Any
        try:
            json_data = json.loads(message.data.decode())
            request = PipelineRequest(**json_data)
            LOGGER.info(f"PubSub | Successfully decoded data {request}")
            consumer(request)
        except Exception as err:
            LOGGER.error(f"PubSub | Validation Error for message with bytes data {message.data}. Error = {err}")
        finally:
            message.ack()

    return subscriber.subscribe(subscription_name, callback)


def listen_until_messages_received(predicate: Predicate, sleep_secs: float) -> bool:
    LOGGER.info(f"PubSub | Checking for events in loop, sleeping for {sleep_secs} seconds between tries")
    while True:
        if predicate():
            LOGGER.info(f"PubSub | Found events to process")
            return True
        time.sleep(sleep_secs)


def download_files(gcs_client: StorageClient, request: PipelineRequest, out_folder: Path) -> None:
    LOGGER.info(f"{request.runId} | Cloud Storage | Downloading files to path {out_folder}")
    blobs = gcs_client.list_blobs(request.inputDataBucket, prefix=request.inputDataPrefix)

    out_folder.mkdir(parents=True, exist_ok=True)

    blob: Blob
    for blob in blobs:
        # Skip over the top-level folder itself
        if str(blob.name).endswith("/"):
            continue
        
        file_path = Path(blob.name.removeprefix(request.inputDataPrefix))
        LOGGER.info(f"{request.runId} | Cloud Storage | Saving blob {blob.name} to path {out_folder / file_path}")
        (out_folder / file_path).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(out_folder / file_path)
    
    LOGGER.info(f"{request.runId} | Cloud Storage | Request {request}: Completed download.")


def run_bash_command(log_prefix: str, args: Sequence[str]) -> int:
    process = subprocess.Popen(args=args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for line in io.TextIOWrapper(process.stdout, encoding="utf-8"):
            LOGGER.info(f"{log_prefix} | {line.strip()}")
    except Exception as e:
        LOGGER.error(f"{log_prefix} | {e}")

    return process.wait()


def handle_single_request(request: PipelineRequest, gcs_client: StorageClient, experiment_dir: Path, visionnerf_script: Path, nvdiffrec_script: Path) -> bool:
    visionnerf_data_dir = experiment_dir / "visionnerf" / "data" / "item1"

    # Download files to directory
    download_files(gcs_client=gcs_client, request=request, out_folder=visionnerf_data_dir)

    # Retrieve OpenGL JSON file
    json_files: Set[Path] = set(visionnerf_data_dir.glob("*.json"))
    if len(json_files) != 1:
        LOGGER.error(f"{request.runId} | Model Setup | Expected single JSON file containing OpenGL coords at top-level of downloaded folder {visionnerf_data_dir}, instead found {json_files}. Ending run.")
        return False
    
    openGL_coords_file: Path = next(iter(json_files)).resolve()

    LOGGER.info(f"{request.runId} | Model Setup | Using JSON file {openGL_coords_file} as OpenGL coords")


    ###########################
    ##### Run first model #####
    ###########################
    LOGGER.info(f"{request.runId} | VisionNerf | Running bash script '{visionnerf_script}'")

    vision_nerf_config = request.visionnerf_run_config().to_dict(experiment_name=request.runId, mount_prefix=Path("/mnt") / "visionnerf")
    with open(experiment_dir / "visionnerf" / "config.txt", "w") as configfile:
        configfile.write("\n".join(
            f"{key} = {value}"
            for (key, value) in vision_nerf_config.items()
        ))

    exit_code = run_bash_command(log_prefix=f"{request.runId} | Model VisionNerf", args=[str(visionnerf_script.resolve()), str(experiment_dir.resolve()), request.visionnerfWeights])
    if exit_code != 0:
        LOGGER.error(f"{request.runId} | VisionNerf | bash script returned a non-zero exit code of {exit_code}, ending run.")
        return False


    ###########################
    ##### Intermodel setup ####
    ###########################
    visionnerf_results_dir = experiment_dir / "visionnerf" / "results" / request.runId

    data_folders: Set[Path] = set(visionnerf_results_dir.glob("*"))
    if len(data_folders) != 1:
        LOGGER.error(f"{request.runId} | InterModel Setup | Expected single data folder containing NERFs to be generated by visionnerf. Instead found {data_folders}. Ending run.")
        return False

    # Rename visionnerf output data folder to 'train'
    data_folder = next(iter(data_folders))
    LOGGER.info(f"{request.runId} | InterModel Setup | Renaming {data_folder} to {visionnerf_results_dir / 'train'}")
    subprocess.call(args=["sudo", "mv", str(data_folder), str(visionnerf_results_dir / 'train')])

    # Copy OpenGL file to the top-level of the visionnerf output data folder
    LOGGER.info(f"{request.runId} | InterModel Setup | Copying {openGL_coords_file} to {visionnerf_results_dir / 'transforms_train.json'}")
    subprocess.call(args=["sudo", "cp", str(openGL_coords_file), str(visionnerf_results_dir / "transforms_train.json")])


    ###########################
    ####  Run second model ####
    ###########################
    LOGGER.info(f"{request.runId} | Nvdiffrec | Running bash script '{nvdiffrec_script}'")
    
    nvdiffrec_config = request.nvdiffrec_run_config().to_dict(experiment_name=request.runId)
    (experiment_dir / "nvdiffrec" / "results").mkdir(parents=True, exist_ok=True)

    with open(experiment_dir / "nvdiffrec" / "config.json", "w") as configfile:
        json.dump(nvdiffrec_config, configfile)

    exit_code = run_bash_command(log_prefix=f"{request.runId} | Model Nvdiffrec", args=[str(nvdiffrec_script.resolve()), str(experiment_dir.resolve())])
    if exit_code != 0:
        LOGGER.error(f"{request.runId} | Nvdiffrec | bash script returned a non-zero exit code of {exit_code}, ending run.")
        return False

    # All successful
    return True


def cli(
        google_application_credentials: str = typer.Option(..., help="Absolute path to GOOGLE_APPLICATION_CREDENTIALS json keys file"),
        project_id: str = typer.Option(..., help="Name of the GCP project"),
        pubsub_subscription_name: str = typer.Option(..., help="Name of the pubsub subscription to listen for messages"), 
        working_directory: str = typer.Option(..., help="Absolute Path to root LOCAL folder to download files to"),
        visionnerf_script: str = typer.Option(default=None, help="Absolute Path to bash script which runs vision nerf."),
        nvdiffrec_script: str = typer.Option(default=None, help="Absolute Path to bash script which runs nvdiffrec."),
        cleanup_before_each: Optional[bool] = typer.Option(default=False, help="Whether to clear out an existing experiment folder with same name (if it exists) before each experiment"),
        cleanup_after_each: Optional[bool] = typer.Option(default=False, help="Whether to cleanup experiment folder after each experiment")):
    
    creds: Credentials = service_account.Credentials.from_service_account_file(google_application_credentials)

    close_logging_fn = init_logging(logger=LOGGER, credentials=creds)

    requests_queue: queue.Queue[PipelineRequest] = queue.Queue()

    storage_client = StorageClient(credentials=creds)

    with SubscriberClient(credentials=creds) as subscriber_client, ThreadPoolExecutor(max_workers=1) as executor:
        initiate_message_pulling(
            subscriber=subscriber_client, 
            consumer=lambda req: requests_queue.put(req),
            project_id=project_id,
            subscription=pubsub_subscription_name)
        
        # Continously log that we are listening...
        def keep_alive_runnable() -> None:
            while True:
                LOGGER.info(f"ML Pipeline KeepAlive | Service is running and awaiting messages on Project {project_id} and PubSub subscription {pubsub_subscription_name}.")
                time.sleep(60)
        executor.submit(keep_alive_runnable)

        try:
            while True:
                listen_until_messages_received(lambda: not requests_queue.empty(), sleep_secs=10)

                request = requests_queue.get()
                experiment_dir = Path(working_directory) / request.runId

                try:
                    if cleanup_before_each:
                        subprocess.call(args=["sudo", "rm", "-r", str(experiment_dir.resolve())])

                    is_success = handle_single_request(
                        request=request, 
                        gcs_client=storage_client,
                        experiment_dir=experiment_dir,
                        visionnerf_script=Path(visionnerf_script),
                        nvdiffrec_script=Path(nvdiffrec_script))

                    if is_success:
                        visionnerf_results_dir: Path = experiment_dir / "visionnerf" / "results" / request.runId / "train"
                        LOGGER.info(f"{request.runId} | Results Output | Copying Visionnerf results from {visionnerf_results_dir} to GCS bucket {request.outputDataBucket}/{request.runId}/visionnerf")
                        run_bash_command(log_prefix=f"{request.runId} | GSUtil Visionnerf", args=["gsutil", "-m", "cp", "-R", str(visionnerf_results_dir.resolve()), f"gs://{request.outputDataBucket}/{request.runId}/visionnerf"])

                        nvdiffrec_results_dir: Path = experiment_dir / "nvdiffrec_eval" / "out"
                        LOGGER.info(f"{request.runId} | Results Output | Copying Nvdiffrec results from {nvdiffrec_results_dir} to GCS bucket {request.outputDataBucket}/{request.runId}/nvdiffrec")
                        run_bash_command(log_prefix=f"{request.runId} | GSUtil Nvdiffrec", args=["gsutil", "-m", "cp", "-R", str(nvdiffrec_results_dir.resolve()), f"gs://{request.outputDataBucket}/{request.runId}/nvdiffrec"])
                    else:
                        LOGGER.error(f"{request.runId} | Results Output | Pipeline failed for Experiment.")

                finally:
                    if cleanup_after_each:
                        subprocess.call(args=["sudo", "rm", "-r", str(experiment_dir.resolve())])

        finally:
            close_logging_fn()


def main():
    typer.run(cli)
