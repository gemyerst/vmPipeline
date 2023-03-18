import json
import logging
import queue
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Final, Optional, Sequence

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

LOGGER: Final[logging.Logger] = logging.getLogger()
LOGGER.setLevel(logging.INFO)


def initiate_message_pulling(
    subscriber: SubscriberClient, 
    consumer: Callable[[PipelineRequest], None], 
    project_id: str,
    subscription: str) -> futures.StreamingPullFuture:

    subscription_name = f'projects/{project_id}/subscriptions/{subscription}'

    LOGGER.info(f"Attempting to subscribe to {subscription_name}")

    def callback(message: Message):
        json_data = json.loads(message.data.decode())
        try:
            request = PipelineRequest(**json_data)
            LOGGER.info(f"Successfully decoded data {request}")
            consumer(request)
        except Exception as err:
            LOGGER.error(f"Validation Error for data {json_data}: {err}")

    return subscriber.subscribe(subscription_name, callback)


def check_on_a_loop(predicate: Callable[[], bool], sleep_secs: float = 1.0, tries: int = 3) -> bool:
    LOGGER.info(f"Checking in loop, trying {tries} times, sleeping {sleep_secs} between tries")
    for i in range(tries):
        if predicate():
            LOGGER.info(f"Succeeded!")
            return True
        LOGGER.info(f"Still did not succeed on try {i}, sleeping for {sleep_secs} seconds")
        time.sleep(sleep_secs)
    LOGGER.info(f"Checking in loop: exhausted wait period, finishing.")
    return False


def download_files(gcs_client: StorageClient, request: PipelineRequest, out_folder: Path) -> None:
    blobs = gcs_client.list_blobs(request.bucket, prefix=request.prefix)

    out_folder.mkdir(parents=True, exist_ok=True)

    blob: Blob
    for blob in blobs:
        if str(blob.name).endswith("/"):
            continue
        file_path = Path(blob.name.removeprefix(request.prefix))
        LOGGER.info(f"Saving blob {blob.name} to path {out_folder / file_path}")
        (out_folder / file_path).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(out_folder / file_path)
    
    LOGGER.info(f"Request {request}: Completed download.")


def run_shell_script(args: Sequence[Path]) -> int:
    process = subprocess.Popen(args=[str(a) for a in args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = process.communicate()
    LOGGER.info(out.decode())
    if err:
        LOGGER.error(err.decode())
    return process.wait()


def handle_single_request(request: PipelineRequest, gcs_client: StorageClient, experiment_dir: Path, visionnerf_script: Path, nvdiffrec_script: Path) -> None:
    visionnerf_data_dir = experiment_dir / "visionnerf" / "data"

    # Download files to directory
    download_files(gcs_client=gcs_client, request=request, out_folder=visionnerf_data_dir)

    # Retrieve OpenGL JSON file
    json_files: Sequence[Path] = list(visionnerf_data_dir.glob("*.json"))
    if len(json_files) != 1:
        LOGGER.error(f"Expected single JSON file containing OpenGL coords at top-level of downloaded folder {visionnerf_data_dir}, instead found {json_files}. Ending run.")
        return
    
    openGL_coords_file: Path = json_files[0].resolve()

    LOGGER.info(f"Using JSON file {openGL_coords_file} as OpenGL coords")

    ###########################
    ##### Run first model #####
    ###########################
    LOGGER.info(f"Running first model: script '{visionnerf_script}'")

    vision_nerf_config = request.vision_nerf_config().to_dict(experiment_name=request.runId, mount_prefix=Path("/mnt") / "visionnerf")
    with open(experiment_dir / "visionnerf" / "config.txt", "w") as configfile:
        configfile.write("\n".join(
            f"{key} = {value}"
            for (key, value) in vision_nerf_config.items()
        ))

    exit_code = run_shell_script(args=[visionnerf_script.resolve(), experiment_dir.resolve()])
    if exit_code != 0:
        LOGGER.error(f"Visionnerf shell script returned a non-zero exit code of {exit_code}, ending run.")
        return

    # Copy OpenGL file to the top-level of the vision nerf output dir
    vision_nerf_outdir: Path = experiment_dir / "visionnerf" / "results"
    vision_nerf_outdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(openGL_coords_file, vision_nerf_outdir / "transforms_train.json")

    ###########################
    ####  Run second model ####
    ###########################
    LOGGER.info(f"Running second model: script '{nvdiffrec_script}'")

    nvdiffrec_config = request.nvdiffrec_config().to_dict(visionnerf_mount_prefix=Path("/mnt") / "visionnerf", nvdiffrec_mount_prefix=Path("/mnt") / "nvdiffrec")
    (experiment_dir / "nvdiffrec").mkdir(parents=True, exist_ok=True)
    with open(experiment_dir / "nvdiffrec" / "config.txt", "w") as configfile:
        json.dump(nvdiffrec_config, configfile)

    exit_code = run_shell_script(args=[nvdiffrec_script.resolve(), experiment_dir.resolve()])
    if exit_code != 0:
        LOGGER.error(f"Nvdiffrec shell script returned a non-zero exit code of {exit_code}, ending run.")
        return


def cli(
        google_application_credentials: str = typer.Option(..., help="Absolute path to GOOGLE_APPLICATION_CREDENTIALS json keys file"),
        project_id: str = typer.Option(..., help="Name of the GCP project"),
        pubsub_subscription_name: str = typer.Option(..., help="Name of the pubsub subscription to check for messages"), 
        working_directory: str = typer.Option(..., help="Absolute Path to root LOCAL folder to download files to"),
        visionnerf_script: str = typer.Option(..., help="""Absolute Path to bash script which runs vision nerf."""),
        nvdiffrec_script: str = typer.Option(..., help="""Absolute Path to bash script which runs nvdiffrec."""),
        perform_cleanup: Optional[bool] = typer.Option(default=False, help="Whether to cleanup folders once done")):
    
    creds: Credentials = service_account.Credentials.from_service_account_file(google_application_credentials)

    close_logging_fn = init_logging(logger=LOGGER, credentials=creds)

    requests: queue.Queue[PipelineRequest] = queue.Queue()

    storage_client = StorageClient(credentials=creds)

    with SubscriberClient(credentials=creds) as subscriber_client:
        pull_future = initiate_message_pulling(
            subscriber=subscriber_client, 
            consumer=lambda req: requests.put(req),
            project_id=project_id,
            subscription=pubsub_subscription_name)
        
        try:
            while True:
                has_messages = check_on_a_loop(lambda: not requests.empty(), sleep_secs=1.0, tries=10)
                if not has_messages:
                    break

                next_request = requests.get()
                download_dir = Path(working_directory) / next_request.runId
                try:
                    handle_single_request(
                        next_request, 
                        storage_client,
                        experiment_dir=download_dir,
                        visionnerf_script=Path(visionnerf_script),
                        nvdiffrec_script=Path(nvdiffrec_script))
                    
                finally:
                    if perform_cleanup:
                        shutil.rmtree(download_dir)

        finally:
            pull_future.cancel()
            close_logging_fn()



def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
