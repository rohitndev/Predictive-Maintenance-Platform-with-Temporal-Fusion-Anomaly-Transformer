"""AWS Lambda entrypoint for serverless serving.

Wraps the FastAPI app with Mangum so the same application runs behind AWS
Lambda + API Gateway (free tier) without code changes. Deploy the container
image (see ``Dockerfile``) to ECR and point the Lambda function at this
handler: ``deployment.lambda_handler.handler``.
"""

from __future__ import annotations

try:
    from mangum import Mangum

    from api.main import app

    handler = Mangum(app)
except Exception as exc:  # pragma: no cover - mangum optional in local dev
    def handler(event, context):  # type: ignore
        return {
            "statusCode": 500,
            "body": f"Serverless adapter unavailable: {exc}",
        }
