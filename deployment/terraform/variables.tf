variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region for all resources"
}

variable "app_name" {
  type        = string
  default     = "predictive-maintenance"
  description = "Application name prefix (no spaces)"
}

variable "environment" {
  type        = string
  default     = "dev"
  description = "Deployment environment (dev/staging/prod)"
}

variable "image_tag" {
  type        = string
  default     = "latest"
  description = "Container image tag to deploy from ECR"
}
