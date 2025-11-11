# ReviewSense

## Running the Script
- Note that the reviewsense_pipeline.py file takes less than 5 minutes to run, so there is no unit test file
- The Script contains several Unit Test Questions.
- GPU usage is recommended for faster inference. Cloud deployment is also recommended.
- For running the advanced RAG Agent with Lora Fine Tuned Model, use the reviewsense_frontend.py, where you can input questions and model at your choice, or if you want to compare them
- Just use [docker run -it reviewsense python reviewsense_frontend.py] command and the front end system will run

## Data
- The data is provided as [final_review_chunked_df.csv](https://drive.google.com/file/d/13I5BZIz7itCWmw5oiAx6PivQEM68FptB/view?usp=sharing)
- Download the data as in the name as final_review_chunked_df.csv
- If you change the name of the dataset, remember to change it as well in the dockerfile as well as in the code to avoid running into any issue.

## API KEY
- The projects require you to have an OPENAI API KEY to access OpenAI's GPT-3.5-turbo API and a Pinecone API Key. Thus, it is recommended to create an (.env) file in your root directory.
- Add you API Key In this plain text form
- OPENAI_API_KEY=your_api_key_here and PINECONE_API_KEY=your_api_key_here
- Remember to build your .env in the folder that you transfer to docker to ensure no error.
- Use load_dotenv() to read in these key.

## Dockerfile
- Remember that the data is included in the dockerfile as the line
- COPY final_review_chunked_df.csv /app/
- Note that there is a .env file in the dockerfile that is expected to have. You can remove that if you find it is not there
- If you change the name of the dataset, remember to change it as well in the dockerfile to avoid running into any issue.

# Docker Instructions

## Building the Docker Image

Build your Docker image:

```bash
docker build -t reviewsense .
```

## Verify the Docker Image Was Built

Check if the image was successfully created by running:

```bash
docker images
```
This command will list all available images. Look for your **image name** in the output.

## Run the Docker Image

To start a container from your image:

```bash
docker run -it reviewsense /bin/bash
```

This opens an interactive session inside the container.

If you want to run the whole code (reviewsense_pipeline.py)

```bash
docker run -it reviewsense
```

If you want to run the Front End (reviewsense_frontend.py)
```bash
docker run -it reviewsense python reviewsense_frontend.py
```

## Delete Docker Image (When No Longer Needed)

To remove an image you no longer need:

1. **Find the Image ID**  
   ```bash
   docker images
   ```

2. **Delete the Image**  
   ```bash
   docker rmi image_id
   ```

   If the image is in use, you may need to force delete it:

   ```bash
   docker rmi -f image_id
   ```
