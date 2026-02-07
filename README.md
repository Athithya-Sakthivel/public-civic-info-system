


# public-civic-info-system


```sh


export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID              # AWS access key used for programmatic authentication (assumed pre-exported)
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY      # AWS secret key paired with the access key (assumed pre-exported)

export AWS_REGION="ap-south-1"                            # AWS region for all deployed infrastructure and data residency
export S3_BUCKET="civic-data-raw-prod"     # S3 bucket for raw source documents, chunk storage, and citation artifacts (must be globally unique)
export PULUMI_STATE_BUCKET="pulumi-backend-670"          # S3 bucket backing Pulumi remote state for IaC deployments
export FRONTEND_UI_BUCKET="civic-bucket-for-ui"          # S3 bucket hosting the static frontend UI assets


make create-s3
make upload-force
export SKIP_WEB_SCRAPING=false && make ELT


# export SEED_URLS="https://www.india.gov.in/my-government/schemes,https://www.myscheme.gov.in,https://csc.gov.in"  # Authoritative seed entry points 
# export ALLOWED_DOMAINS="india.gov.in,myscheme.gov.in,csc.gov.in"  # Strict allowlist of domains permitted during crawling
# export SKIP_WEB_SCRAPING=false

```





