import json
import pulumi
from pulumi import Config, ResourceOptions, export
import pulumi_aws as aws
import pulumi_awsx as awsx

cfg = Config()
project_prefix = cfg.get("prefix") or "civic-index"
aws_region = aws.config.region or cfg.get("aws:region") or "ap-south-1"

# Naming and params (configurable via `pulumi config set`)
image_uri = cfg.require("image_uri")                    # e.g. 123456789012.dkr.ecr.ap-south-1.amazonaws.com/indexer:latest
schedule_expression = cfg.get("schedule") or "cron(0 3 * * ? *)"  # daily 03:00 UTC
cpu = cfg.get_int("cpu") or 1024
memory = cfg.get_int("memory") or 2048
fargate_platform_version = cfg.get("fargate_platform_version") or "LATEST"
container_name = cfg.get("container_name") or "indexer"
env_vars = cfg.get_object("env") or {}                  # optional dict of env vars
s3_prefix = cfg.get("s3_prefix") or f"{project_prefix}"
retain_buckets = cfg.get_bool("retain_buckets") or False

# S3 buckets (raw, chunks, metadata)
def make_bucket(name_suffix):
    name = f"{project_prefix}-{name_suffix}"
    b = aws.s3.Bucket(
        name,
        bucket=name,
        versioning=aws.s3.BucketVersioningArgs(enabled=True),
        server_side_encryption_configuration=aws.s3.BucketServerSideEncryptionConfigurationArgs(
            rule=aws.s3.BucketServerSideEncryptionConfigurationRuleArgs(
                apply_server_side_encryption_by_default=aws.s3.BucketServerSideEncryptionConfigurationRuleApplyServerSideEncryptionByDefaultArgs(
                    sse_algorithm="AES256"
                )
            )
        ),
        force_destroy=not retain_buckets,
        tags={"project": project_prefix, "component": name_suffix},
    )
    return b

raw_bucket = make_bucket("raw")
chunks_bucket = make_bucket("chunks")
meta_bucket = make_bucket("meta")

# VPC (awsx convenience; creates private subnets, no NAT by default unless specified)
vpc = awsx.ec2.Vpc(
    f"{project_prefix}-vpc",
    # default settings are production-sane; control nat_gateways, number of AZs via config if desired
    number_of_availability_zones=cfg.get_int("az_count") or 2,
    # If you want no NAT (private isolated subnets), set nat_gateways=0 in config
    nat_gateways=cfg.get_int("nat_gateways") or 0,
)

# Security Group for ECS tasks: allow outbound HTTPS only
task_sg = aws.ec2.SecurityGroup(
    f"{project_prefix}-task-sg",
    vpc_id=vpc.vpc_id,
    description="ECS task security group - only HTTPS outbound",
    egress=[aws.ec2.SecurityGroupEgressArgs(protocol="-1", from_port=0, to_port=0, cidr_blocks=["0.0.0.0/0"])],
    ingress=[],
    tags={"project": project_prefix, "component": "ecs-task-sg"},
)

# CloudWatch Log Group for the task
log_group = aws.cloudwatch.LogGroup(
    f"{project_prefix}-log",
    name=f"/{project_prefix}/indexer",
    retention_in_days=30,
)

# ECR repo (image should be built & pushed separately)
ecr = aws.ecr.Repository(
    f"{project_prefix}-repo",
    image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(scan_on_push=False),
    tags={"project": project_prefix},
)

# IAM role: task execution (managed policy required for pulling images and sending logs)
execution_role = aws.iam.Role(
    f"{project_prefix}-task-exec-role",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }),
    tags={"project": project_prefix, "component": "task-exec-role"},
)

aws.iam.RolePolicyAttachment(
    f"{project_prefix}-exec-role-attach",
    role=execution_role.name,
    policy_arn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
)

# IAM role: task application role (permissions to S3 + Bedrock invoke)
task_role = aws.iam.Role(
    f"{project_prefix}-task-role",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }),
    tags={"project": project_prefix, "component": "task-role"},
)

# Minimal inline policy for task role: S3 (only to our buckets) + CloudWatch logs + Bedrock invoke action
account_id = aws.get_caller_identity().account_id
bedrock_actions = [
    "bedrock:InvokeModel",
    "bedrock:InvokeModelWithResponseStream",
]
task_policy = aws.iam.RolePolicy(
    f"{project_prefix}-task-policy",
    role=task_role.id,
    policy=pulumi.Output.all(raw_bucket.arn, chunks_bucket.arn, meta_bucket.arn).apply(
        lambda arns: json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["s3:ListBucket"],
                    "Resource": [arns[0], arns[1], arns[2]]
                },
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                    "Resource": [f"{arns[0]}/*", f"{arns[1]}/*", f"{arns[2]}/*"]
                },
                {
                    "Effect": "Allow",
                    "Action": ["logs:CreateLogStream", "logs:PutLogEvents", "logs:CreateLogGroup"],
                    "Resource": ["*"]
                },
                {
                    "Effect": "Allow",
                    "Action": bedrock_actions,
                    "Resource": ["*"]
                }
            ]
        })
    ),
)

# ECS Cluster
cluster = aws.ecs.Cluster(
    f"{project_prefix}-cluster",
    name=f"{project_prefix}-cluster",
    settings=[aws.ecs.ClusterSettingArgs(name="containerInsights", value="enabled")],
    tags={"project": project_prefix},
)

# ECS Task Definition (Fargate)
# container definitions must be JSON string
container_def = pulumi.Output.all(log_group.name, ecr.repository_url).apply(
    lambda args: json.dumps([{
        "name": container_name,
        "image": image_uri if image_uri else f"{args[1]}:latest",
        "cpu": cpu,
        "memory": memory,
        "essential": True,
        "environment": [{"name": k, "value": str(v)} for k, v in (env_vars or {}).items()],
        "logConfiguration": {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": args[0],
                "awslogs-region": aws_region,
                "awslogs-stream-prefix": project_prefix,
            }
        }
    }])
)

task_def = aws.ecs.TaskDefinition(
    f"{project_prefix}-taskdef",
    family=f"{project_prefix}-taskdef",
    cpu=str(cpu),
    memory=str(memory),
    network_mode="awsvpc",
    requires_compatibilities=["FARGATE"],
    runtime_platform=aws.ecs.TaskDefinitionRuntimePlatformArgs(operating_system_family="LINUX"),
    container_definitions=container_def,
    task_role_arn=task_role.arn,
    execution_role_arn=execution_role.arn,
    tags={"project": project_prefix},
)

# IAM role that EventBridge needs to RunTask on ECS
events_role = aws.iam.Role(
    f"{project_prefix}-events-role",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "events.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }),
    tags={"project": project_prefix, "component": "events-role"},
)

# Policy for events role: allow ecs:RunTask and iam:PassRole (only for our two roles)
events_policy = aws.iam.RolePolicy(
    f"{project_prefix}-events-policy",
    role=events_role.id,
    policy=pulumi.Output.all(cluster.arn, task_def.arn, execution_role.arn, task_role.arn).apply(
        lambda vals: json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Action": ["ecs:RunTask"], "Resource": vals[1]},  # task definition
                {"Effect": "Allow", "Action": ["ecs:RunTask"], "Resource": vals[0]},  # cluster arn (RunTask requires access)
                {"Effect": "Allow", "Action": ["iam:PassRole"], "Resource": [vals[2], vals[3]]}
            ]
        })
    )
)

# EventBridge (CloudWatch) Rule for schedule
rule = aws.cloudwatch.EventRule(
    f"{project_prefix}-schedule",
    description="Schedule to run the indexing task",
    schedule_expression=schedule_expression,
)

# Event Target: instruct EventBridge to run the ECS task
# Use ecs_target block: clusterArn, taskDefinitionArn, taskCount, launchType, networkConfiguration (awsvpc)
# Pulumi `aws.cloudwatch.EventTarget` supports ecs_target property.
target = aws.cloudwatch.EventTarget(
    f"{project_prefix}-target",
    rule=rule.name,
    arn=cluster.arn,
    role_arn=events_role.arn,
    ecs_target=aws.cloudwatch.EventTargetEcsTargetArgs(  # Pulumi accepts ecs_target block
        task_definition_arn=task_def.arn,
        task_count=1,
        launch_type="FARGATE",
        network_configuration=aws.cloudwatch.EventTargetNetworkConfigurationArgs(
            # Use the VPC private subnets created by awsx; pick the subnet ids
            awsvpc_configuration=aws.cloudwatch.EventTargetNetworkConfigurationAwsvpcConfigurationArgs(
                subnets=vpc.private_subnet_ids,
                security_groups=[task_sg.id],
                assign_public_ip="DISABLED",
            )
        ),
    )
)

# Allow rule to be invoked (this is not always necessary; role above handles RunTask)
# No extra permission needed here in most cases.

# Exports for operational use
export("project_prefix", project_prefix)
export("region", aws_region)
export("raw_bucket_name", raw_bucket.id)
export("chunks_bucket_name", chunks_bucket.id)
export("meta_bucket_name", meta_bucket.id)
export("ecr_repo", ecr.repository_url)
export("ecs_cluster", cluster.arn)
export("task_definition_arn", task_def.arn)
export("schedule_rule_name", rule.name)
export("log_group", log_group.name)
