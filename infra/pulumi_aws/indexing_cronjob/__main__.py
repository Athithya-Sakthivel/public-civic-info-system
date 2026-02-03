import json
import pulumi
from pulumi import Config
import pulumi_aws as aws
import pulumi_awsx as awsx

cfg = Config()
project_prefix = cfg.get("prefix") or "civic-index"
aws_region = aws.config.region or cfg.get("aws:region") or "ap-south-1"

image_uri = cfg.require("image_uri")
schedule_expression = cfg.get("schedule") or "cron(0 3 * * ? *)"
cpu = cfg.get_int("cpu") or 1024
memory = cfg.get_int("memory") or 2048
container_name = cfg.get("container_name") or "indexer"
env_vars = cfg.get_object("env") or {}
retain_buckets = cfg.get_bool("retain_buckets") or False
az_count = cfg.get_int("az_count") or 2
nat_gateways = cfg.get_int("nat_gateways") if cfg.get("nat_gateways") is not None else 0

def make_bucket(suffix):
    name = f"{project_prefix}-{suffix}"
    b = aws.s3.Bucket(
        name,
        bucket=name,
        force_destroy=not retain_buckets,
        tags={"project": project_prefix, "component": suffix},
    )
    aws.s3.BucketVersioning(
        f"{name}-versioning",
        bucket=b.id,
        versioning_configuration={"status": "Enabled"},
        opts=pulumi.ResourceOptions(parent=b),
    )
    aws.s3.BucketServerSideEncryptionConfigurationV2(
        f"{name}-sse",
        bucket=b.id,
        server_side_encryption_configuration={
            "rule": {
                "apply_server_side_encryption_by_default": {"sse_algorithm": "AES256"}
            }
        },
        opts=pulumi.ResourceOptions(parent=b),
    )
    pulumi.log.info(f"created bucket {name}")
    return b

raw_bucket = make_bucket("raw")
chunks_bucket = make_bucket("chunks")
meta_bucket = make_bucket("meta")

vpc = awsx.ec2.Vpc(
    f"{project_prefix}-vpc",
    number_of_availability_zones=az_count,
    nat_gateways=nat_gateways,
)

task_sg = aws.ec2.SecurityGroup(
    f"{project_prefix}-task-sg",
    vpc_id=vpc.vpc_id,
    description="ECS task security group - allow outbound",
    egress=[aws.ec2.SecurityGroupEgressArgs(protocol="-1", from_port=0, to_port=0, cidr_blocks=["0.0.0.0/0"])],
    ingress=[],
    tags={"project": project_prefix, "component": "ecs-task-sg"},
)

log_group = aws.cloudwatch.LogGroup(
    f"{project_prefix}-log",
    name=f"/{project_prefix}/indexer",
    retention_in_days=30,
)

ecr = aws.ecr.Repository(
    f"{project_prefix}-repo",
    image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(scan_on_push=False),
    tags={"project": project_prefix},
)

execution_role = aws.iam.Role(
    f"{project_prefix}-task-exec-role",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}]
    }),
    tags={"project": project_prefix, "component": "task-exec-role"},
)

aws.iam.RolePolicyAttachment(
    f"{project_prefix}-exec-role-attach",
    role=execution_role.name,
    policy_arn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
)

task_role = aws.iam.Role(
    f"{project_prefix}-task-role",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}]
    }),
    tags={"project": project_prefix, "component": "task-role"},
)

bedrock_actions = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
task_policy = aws.iam.RolePolicy(
    f"{project_prefix}-task-policy",
    role=task_role.id,
    policy=pulumi.Output.all(raw_bucket.arn, chunks_bucket.arn, meta_bucket.arn).apply(
        lambda arns: json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": [arns[0], arns[1], arns[2]]},
                {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"], "Resource": [f"{arns[0]}/*", f"{arns[1]}/*", f"{arns[2]}/*"]},
                {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents", "logs:CreateLogGroup"], "Resource": ["*"]},
                {"Effect": "Allow", "Action": bedrock_actions, "Resource": ["*"]}
            ]
        })
    ),
)

cluster = aws.ecs.Cluster(
    f"{project_prefix}-cluster",
    name=f"{project_prefix}-cluster",
    settings=[aws.ecs.ClusterSettingArgs(name="containerInsights", value="enabled")],
    tags={"project": project_prefix},
)

container_definitions = pulumi.Output.all(log_group.name, ecr.repository_url).apply(
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
    container_definitions=container_definitions,
    task_role_arn=task_role.arn,
    execution_role_arn=execution_role.arn,
    tags={"project": project_prefix},
)

events_role = aws.iam.Role(
    f"{project_prefix}-events-role",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "events.amazonaws.com"}, "Action": "sts:AssumeRole"}]
    }),
    tags={"project": project_prefix, "component": "events-role"},
)

events_policy = aws.iam.RolePolicy(
    f"{project_prefix}-events-policy",
    role=events_role.id,
    policy=pulumi.Output.all(cluster.arn, task_def.arn, execution_role.arn, task_role.arn).apply(
        lambda vals: json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Action": ["ecs:RunTask"], "Resource": vals[1]},
                {"Effect": "Allow", "Action": ["ecs:RunTask"], "Resource": vals[0]},
                {"Effect": "Allow", "Action": ["iam:PassRole"], "Resource": [vals[2], vals[3]]}
            ]
        })
    ),
)

rule = aws.cloudwatch.EventRule(
    f"{project_prefix}-schedule",
    description="Schedule to run the indexing task",
    schedule_expression=schedule_expression,
)

target = aws.cloudwatch.EventTarget(
    f"{project_prefix}-target",
    rule=rule.name,
    arn=cluster.arn,
    role_arn=events_role.arn,
    ecs_target=aws.cloudwatch.EventTargetEcsTargetArgs(
        task_definition_arn=task_def.arn,
        task_count=1,
        launch_type="FARGATE",
        network_configuration=aws.cloudwatch.EventTargetEcsTargetNetworkConfigurationArgs(
            subnets=vpc.private_subnet_ids,
            assign_public_ip=False,
            security_groups=[task_sg.id],
        ),
    ),
    opts=pulumi.ResourceOptions(depends_on=[events_policy, task_def, cluster, events_role]),
)

pulumi.export("project_prefix", project_prefix)
pulumi.export("region", aws_region)
pulumi.export("raw_bucket_name", raw_bucket.id)
pulumi.export("chunks_bucket_name", chunks_bucket.id)
pulumi.export("meta_bucket_name", meta_bucket.id)
pulumi.export("ecr_repo", ecr.repository_url)
pulumi.export("ecs_cluster", cluster.arn)
pulumi.export("task_definition_arn", task_def.arn)
pulumi.export("schedule_rule_name", rule.name)
pulumi.export("log_group", log_group.name)
