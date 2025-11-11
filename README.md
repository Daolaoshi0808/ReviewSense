# Final Stiching Project

## Running the Script
- Note that the Stiching_Hongkai.py file takes less than 5 minutes to wrong, so there is no unit test file
- The Script contains several Unit Test Questions.
- The Docker image is at deepdish4 server with the image name pin2118/stiching_project
- For running the advanced RAG Agent with Lora Fine Tuned Model, use the Stiching_Hongkai_frontend.py, where you can input questions and model at your choice, or if you want to compare them
- Just use [docker run -it pin2118/stiching_project python Stiching_Hongkai_frontend.py] command and the front end system will run

- 
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

# Docker Instruction

## Running the Project on DeepDish Server

To run this project on a **DeepDish** server at **Northwestern University**, follow the steps below:

### 1. Copy Your Files to DeepDish

You need to transfer your project files to DeepDish. You have two options:

#### **Option A: Using SFTP (FileZilla)**
- Connect to **DeepDish** using **FileZilla** or any other SFTP software.
- Upload your project directory to your home folder (`~`).

#### **Option B: Using SCP (from Terminal)**
If you prefer the command line, use `scp` to copy files over:

```bash
scp -r local_project_directory netid@mlds-deepdish2.ads.northwestern.edu:~/
```

Replace `netid` with your **Northwestern NetID**.

---

### 2. Connect to a DeepDish Server

Use SSH to connect to a DeepDish server:

```bash
ssh netid@mlds-deepdishX.ads.northwestern.edu
```

> Replace **`X`** with a number between **1 and 4** (e.g., `mlds-deepdish2`).

---

### 3. Navigate to Your Project Directory

Once logged in, navigate to the directory where you uploaded your Docker files:

```bash
cd ~/your_project_directory
```

---

### 4. Build the Docker Image

Run the following command to build your Docker image:

```bash
docker build -t name_of_the_image /path/to/directory
```

If you're inside the project directory, you can simply use:

```bash
docker build -t pin2118/stiching_project .
```

Replace **`netid`** with your **Northwestern NetID**.

---

### 5. Verify the Docker Image Was Built

Check if the image was successfully created by running:

```bash
docker images
```
This command will list all available images. Look for your **image name** in the output.

---

### 6. Run the Docker Image

To start a container from your image:

```bash
docker run -it pin2118/stiching_project /bin/bash
```

This opens an interactive session inside the container.

If you want to run the whole code (Stiching_Hongkai.py)

```bash
docker run -it pin2118/stiching_project
```

If you want to run the Front End (Stiching_Hongkai_frontend.py
```bash
docker run -it pin2118/stiching_project python Stiching_Hongkai_frontend.py
```

---

### 7. Delete Docker Image (When No Longer Needed)

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


