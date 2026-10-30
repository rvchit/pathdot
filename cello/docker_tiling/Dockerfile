
ARG FUNCTION_DIR="/function"

FROM python:3.9 AS build-image

ARG FUNCTION_DIR

# Copy function code
RUN mkdir -p ${FUNCTION_DIR}
COPY . ${FUNCTION_DIR}

# Install the function's dependencies
RUN pip install \
    --target ${FUNCTION_DIR} \
        awslambdaric


# Install dependencies for building and libvips
RUN apt-get update \
  && apt-get install -y \
	build-essential \
	pkg-config \
    libvips-dev \
    curl

# Install Python dependencies
RUN pip install --target ${FUNCTION_DIR} pyvips boto3

#sudo apt install libvips

# Use Python 3.9 slim image as the base
FROM python:3.9-slim

ARG FUNCTION_DIR

# Install dependencies for building and libvips
RUN apt-get update \
  && apt-get install -y \
	build-essential \
	pkg-config \
    libvips-dev \
    curl


# Install the AWS Lambda Runtime Interface Emulator
#RUN mkdir -p /aws-lambda && \
#    curl -Lo /aws-lambda/aws-lambda-rie https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie && \
#    chmod +x /aws-lambda/aws-lambda-rie

# Set the working directory to /data
WORKDIR ${FUNCTION_DIR}

# Copy in the built dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}



# set entry point
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]

# Set the default handler (replace 'lambda_function.lambda_handler' with your actual handler)
CMD ["lambda.handler"]
