env:
  PROJECT_PREFIX: civic-index
  AWS_REGION: ap-south-1

  IMAGE_REPO: civic-indexing
  IMAGE_TAG: amd64-arm64-v2
  PLATFORMS: linux/amd64,linux/arm64
  LOCAL_PLATFORM: linux/amd64
  NO_CACHE: '1'

  PUSH: 'true'
  REGISTRY_TYPE: ecr
  ECR_REPO: 681802563986.dkr.ecr.ap-south-1.amazonaws.com/civic-index-repo-4dcfbb4
  PUSH_LATEST: 'false'

  BUILD_CONTEXT: indexing_pipeline
  DOCKERFILE_PATH: indexing_pipeline/Dockerfile

  OCR_LANGUAGES: eng,tam,hin
  INDIC_OCR_SIZE: best

  SMOKE_TEST: 'false'
  SMOKE_TIMEOUT: '60'
