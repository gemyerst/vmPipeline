

# Develop

Run pip install editable, with dev dependencies:

```
pip install -e .[dev]
```

# Run

Run pip install 

```
pip install .
```



# Data

```json
{
    "runId": "Experiment2",
    "bucket": "bucketName",
    "prefix": "prefix/"
}
```



# Running

```sh
ml_pipeline --google-application-credentials "$PWD/keys.json" --project-id "projectId" --pubsub-subscription-name "subName" --working-directory "$PWD/working" --visionnerf-script "$PWD/modelScripts/visionnerf.sh" --nvdiffrec-script "$PWD/modelScripts/nvdiffrec.sh"
```
