# Terraform — AWS serverless serving infrastructure for the
# Predictive Maintenance Platform (Lambda container + API Gateway + S3 + ECR).
#
#   terraform init
#   terraform plan
#   terraform apply
#
# Credentials are supplied via terraform.tfvars (git-ignored) or the standard
# AWS provider chain. Everything here fits within the AWS Free Tier.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ----------------------------------------------------------------------------
# S3 bucket for model artifacts (champion weights, scalers, metadata).
# ----------------------------------------------------------------------------
resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.app_name}-artifacts-${var.environment}"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# ----------------------------------------------------------------------------
# ECR repository for the serving container image.
# ----------------------------------------------------------------------------
resource "aws_ecr_repository" "serving" {
  name                 = "${var.app_name}-serving"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ----------------------------------------------------------------------------
# IAM role for the Lambda function.
# ----------------------------------------------------------------------------
resource "aws_iam_role" "lambda" {
  name = "${var.app_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_s3" {
  name = "${var.app_name}-lambda-s3"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject"]
      Resource = "${aws_s3_bucket.artifacts.arn}/*"
    }]
  })
}

# ----------------------------------------------------------------------------
# Lambda function (container image) + API Gateway HTTP API.
# ----------------------------------------------------------------------------
resource "aws_lambda_function" "serving" {
  function_name = "${var.app_name}-serving"
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.serving.repository_url}:${var.image_tag}"
  timeout       = 60
  memory_size   = 1024

  environment {
    variables = {
      AWS_S3_BUCKET = aws_s3_bucket.artifacts.bucket
    }
  }
}

resource "aws_apigatewayv2_api" "http" {
  name          = "${var.app_name}-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.serving.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.serving.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}

output "api_endpoint" {
  value       = aws_apigatewayv2_stage.default.invoke_url
  description = "Public HTTPS endpoint of the predictive-maintenance API"
}

output "ecr_repository_url" {
  value       = aws_ecr_repository.serving.repository_url
  description = "Push the serving image here before deploying"
}
