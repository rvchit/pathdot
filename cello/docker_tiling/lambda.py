import boto3
import os  
import pyvips
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set log level to INFO

s3_client = boto3.client('s3')

def download_from_s3(bucket, key, local_path):
    """Download the file from S3."""
    logger.info(f"Downloading {key} from S3 bucket {bucket} to {local_path}")
    try:
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Successfully downloaded {key} to {local_path}")
    except Exception as e:
        logger.error(f"Failed to download {key} from S3: {str(e)}")
        raise e

def create_tiles(svs_path, dzi_output_dir, tile_size=256, overlap=1):
    """Convert SVS to DZI format using pyvips."""
    logger.info(f"Converting {svs_path} to DZI tiles in {dzi_output_dir}")
    try:
        svs_image = pyvips.Image.new_from_file(svs_path)
        svs_image.dzsave(dzi_output_dir, tile_size=tile_size, overlap=overlap)
        logger.info(f"Successfully created DZI tiles in {dzi_output_dir}")
    except Exception as e:
        logger.error(f"Failed to create DZI tiles from {svs_path}: {str(e)}")
        raise e

import os
import logging
import boto3

s3_client = boto3.client('s3')

def upload_tiles_to_s3(dzi_output_dir, destination_bucket, image_name):
    """Upload the DZI tiles to S3."""
    logger.info(f"Uploading DZI tiles from {dzi_output_dir} to S3 bucket {destination_bucket}")
    
    # DZI file path and the directory for JPEG tiles
    dzi_path = f'/tmp/{image_name}.dzi'
    jpeg_dir = f'/tmp/{image_name}_files'
    s3_dzi_location = f"{image_name}/{image_name}.dzi"

    try:
        # Upload the DZI file
        logger.info("Uploading the DZI file")
        s3_client.upload_file(dzi_path, destination_bucket, s3_dzi_location)
        logger.info(f"Successfully uploaded DZI file to s3://{destination_bucket}/{s3_dzi_location}")
    except Exception as e:
        logger.error(f"Failed to upload DZI file: {str(e)}")
        raise e

    try:
        # Upload the JPEG tiles
        logger.info("Uploading JPEG tiles")
        for root, dirs, files in os.walk(jpeg_dir):
            for file in files:
                file_path = os.path.join(root, file)
                logger.info(f"File path: {file_path}")
                # Use the jpeg_dir to calculate the relative path of the tiles
                relative_path = os.path.relpath(file_path, jpeg_dir)
                s3_key = f'{image_name}/{relative_path}'  # S3 key for tile

                logger.info(f"Uploading {file_path} to s3://{destination_bucket}/{s3_key}")
                try:
                    s3_client.upload_file(file_path, destination_bucket, s3_key)
                    logger.info(f"Successfully uploaded {file_path} to s3://{destination_bucket}/{s3_key}")
                except Exception as upload_err:
                    logger.error(f"Failed to upload {file_path} to S3: {str(upload_err)}")
                    raise upload_err
        
        logger.info(f"Successfully uploaded all JPEG tiles for {image_name} to {destination_bucket}")
    
    except Exception as e:
        logger.error(f"Failed to upload JPEG tiles: {str(e)}")
        raise e

def handler(event, context):
    """Main Lambda function."""
    
    source_bucket = 'cello-bucket'
    destination_bucket = 'cello-tiles'

    key = event['Records'][0]['s3']['object']['key']

    # Log that the key has been extracted properly 
    logger.info(f"S3 key extracted from event: {key}")
    
    # gets name of the image using split 
    image_name = os.path.splitext(os.path.basename(key))[0]

    # sets local file location for download  
    local_file = f'/tmp/{os.path.basename(key)}'

    # output directory for image
    dzi_output_dir = f'/tmp/{image_name}'

    # Download, tile, and upload
    download_from_s3(source_bucket, key, local_file)
    create_tiles(local_file, dzi_output_dir)
    upload_tiles_to_s3(dzi_output_dir, destination_bucket, image_name)

    return {"statusCode": 200, "body": f"DZI tiles for {key} created and uploaded to {destination_bucket} successfully"}
