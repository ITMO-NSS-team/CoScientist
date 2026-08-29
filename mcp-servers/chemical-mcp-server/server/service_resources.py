from .clients.chemical_client import ChemServiceClient
from .clients.retrosynthesis_client import RetrosynthesisServiceClient
from .config import get_settings
from .utils.s3_utils import S3BucketService

settings = get_settings()

s3_service = S3BucketService(
    endpoint=settings.s3_endpoint_url,
    access_key=settings.s3_access_key,
    secret_key=settings.s3_secret_key,
    bucket_name=settings.s3_bucket_name,
)

chem_service = ChemServiceClient(
    host=settings.chem_services_host,
    port=str(settings.chem_services_port),
    timeout=settings.chem_services_timeout,
    docking_timeout=settings.chem_services_docking_timeout,
)

retrosynthesis_service = RetrosynthesisServiceClient(
    host=settings.retrosynthesis_services_host,
    port=str(settings.retrosynthesis_services_port),
    timeout=settings.retrosynthesis_request_timeout,
)
