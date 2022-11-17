To build and deploy a classification model using MLFLOW and AWS SAGEMAKER

Tracking the model experiments with MLFLOW (train.py)
Pushing the image to AWS ECR (aws command line)
Deploying the model on AWS Sagemaker (deploy.py)
Make predictions (predict.py)

______________________
1) Setup AWS IAM user
- Go inside IAM page
- Create a new user : An IAM user is an identity with long-term credentials that is used to interact with AWS in an account.
- Input a <b>username</b> and select Programmatic access to get <b>access key ID </b> and <b>secret access key</b>
- Go the Permissions -> Add user to group -> Create group
- Provide a group name
- Filter and select policies : 1) AmazonSagemakerFullaccess 2) AmazonEC2ContainerRegistryFullaccess
- Click on Create group
- Next Tag(Optional)
- Review -> Create User
- Copy and store 1) Access Key ID 2) Secret Access Key
- Click into your user(username) : Copy User ARN


2) AWS CLI Configuration
cmd :
    - sudo apt-get update
    - sudo apt-get install awscli
    - aws configure
        AWS Access Key ID
        AWS Secret Access Key
        Check default region at home page
        Default outpur format : json

________________________

Build a Docker Image and Push it to AWS ECR

cmd:
 - export AWS_ACCESS_KEY_ID=
 - export AWS_SECRET_ACCESS_KEY=

cmd(in another terminal):
  - mlflow ui
Copy required model path

Go to the required model path in export command terminal
cmd(from the same terminal where export of AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY was done):
  - mlflow sagemaker build-and-push-container

Now, all required images are pushed into AWS ECR and are also present as a part of local docker images
cmd:
- docker images

AWS Console:
Go to AWS ECR
Check IMAGE URI and image tags

__________________________

Deploy Image from AWS ECR to AWS Sagemaker
cmd:
To get the aws id
  - aws sts get-caller-identity --query Account --output text
  copy and store
 
__________________________________________

Create ARN Role for AWS Sagemaker Fullaccess

Roles : Create new role
Use cases for AWS Services -> Sagemaker -> Sagemaker Execution
Create Role
Copy and store ROLE ARN

______________________

Create permision for the user to access S3 bucket

Go to the recently created user
Click on Add inline Policies
Service -> S3
Actions -> All S3 Actions
Resources -> All Resources
Click on Review Policy
Provide a name to the policy
Click on Create Policy

______________________________

Model Deployment on AWS Sagemaker
python3 deploy.py

___________________
AWS Sagemaker inference

Go to AWS Sagemaker
Click Inference -> Models
