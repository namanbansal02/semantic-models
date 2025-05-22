# 🧠 Sentence Embedding using Transformers: Analysis and Reflection

## Overview

In this exercise, I worked with the HuggingFace `transformers` library to generate **sentence embeddings** using the pre-trained model `sentence-transformers/paraphrase-mpnet-base-v2`. Sentence embeddings are numerical representations of text that capture semantic information and can be used for tasks like semantic search, clustering, or sentence similarity.

The code performs the following steps:

1. **Tokenization**  
   I used `AutoTokenizer` to tokenize a list of sentences. Tokenization converts text into token IDs while ensuring proper padding and truncation for batch processing.

2. **Model Loading**  
   I loaded the pre-trained `paraphrase-mpnet-base-v2` model from HuggingFace, which is fine-tuned for generating semantically meaningful sentence embeddings.

3. **Model Inference**  
   The tokenized input is passed through the transformer model using `torch.no_grad()` to prevent gradient computations, which speeds up inference and reduces memory usage.

4. **Mean Pooling**  
   After obtaining token-level embeddings from the model, I applied **mean pooling**, which averages the embeddings across all tokens in the sentence, taking the attention mask into account. This provides a fixed-size vector representation per sentence.

5. **Output**  
   Finally, I printed the sentence embeddings, which are `768`-dimensional vectors capturing the semantic meaning of the input sentences.

---

## 🧮 Pooling Strategy: Mean Pooling

The `mean_pooling` function is crucial here. It ensures that only the actual tokens (not padding) contribute to the final sentence embedding by applying the attention mask. This gives a more accurate representation of the sentence meaning, as padding tokens are excluded from the average.

---

## 🛠️ If I Had to Build This From Scratch

If I were to build a sentence embedding model from scratch (without using a pre-trained transformer), I would approach it in the following way:

1. **Data Collection**  
   I would first gather a large dataset of sentence pairs with semantic similarity scores, such as the SNLI, STS-B, or Quora Question Pairs datasets.

2. **Model Architecture**  
   I would implement a transformer-based architecture similar to BERT or MPNet using PyTorch or TensorFlow. This would involve:
   - Token and positional embeddings  
   - Multi-head self-attention layers  
   - Feed-forward layers  
   - Layer normalization and residual connections

3. **Training Objective**  
   I would train the model using a **contrastive learning objective**, such as **triplet loss** or **cosine similarity loss**, to ensure that semantically similar sentences are closer in vector space than dissimilar ones.

4. **Pooling Layer**  
   After obtaining the final hidden states, I would implement a pooling strategy — mean pooling, max pooling, or using the `[CLS]` token embedding — to generate fixed-size sentence vectors.

5. **Evaluation**  
   Finally, I would evaluate the model on downstream tasks like sentence similarity, clustering, or classification using standard benchmarks (e.g., STS Benchmark).

---

## ✍️ Reflection

Using pre-trained models like `paraphrase-mpnet-base-v2` significantly accelerates NLP experimentation and deployment by providing high-quality semantic representations out of the box. However, understanding the internals — like attention, pooling, and training objectives — gives me the confidence to customize or build models for specialized applications when needed.
---


# 🚀 Part B: Deployment of Sentence Similarity Model as a Server API Endpoint

## Core Approach

In Part B, my goal was to **deploy the sentence similarity algorithm developed in Part A as a RESTful API** on a cloud service provider. This API allows external clients to send two input sentences and receive a similarity score in response.

### Steps Taken:

1. **API Development with FastAPI **  
   I wrapped the sentence embedding and similarity computation logic from Part A inside a lightweight web framework:
   - The API exposes a POST endpoint `/similarity` that accepts a JSON body with keys `"text1"` and `"text2"`.
   - Upon receiving the request, the API:
     - Tokenizes and embeds both input sentences using the pre-trained transformer model.
     - Computes the cosine similarity between the two embeddings.
     - Returns the similarity score in JSON format with the key `"similarity score"`.

2. ### **Model Loading and Caching**
   - The `SentenceTransformer` model (`paraphrase-mpnet-base-v2`) is either loaded from disk using `joblib` or downloaded and cached on first run.
   - This drastically improves performance by preventing the model from reloading with every request.
   - The model is used in CPU mode for broad compatibility across environments.


3. ### **Hosting**
   - The API was deployed on a **self-hosted Linux server** with public IP access.
   - It runs on port `8005` and can be accessed globally.
   - Deployment was managed using `uvicorn` as the ASGI server.

4. **Request-Response Format**  
   The API strictly follows the prescribed request-response format:

   - **Request JSON**:  
     ```json
     {
       "text1": "nuclear body seeks new tech",
       "text2": "terror suspects face arrest"
     }
     ```
   
   - **Response JSON**:  
     ```json
     {
       "similarity score": 0.2
     }
     ```
5. ### **Similarity Computation Logic**
   - Both input sentences are encoded into dense vector embeddings using the loaded transformer.
   - Cosine similarity is calculated using `sklearn.metrics.pairwise.cosine_similarity`.
   - The score, initially in the range `[-1, 1]`, is normalized to `[0, 1]` using the formula:  
     \[
     \text{normalized\_score} = \frac{\text{cosine\_score} + 1}{2}
     \]
   - The output is rounded to 4 decimal places for consistency.

6. ### **Error Handling**
   - If either `"text1"` or `"text2"` is missing or empty, a `400 Bad Request` is returned.
   - This prevents unnecessary computation and ensures input quality.

7. **Testing**  
   I tested the API using HTTP clients such as `curl` or Postman to ensure the endpoint correctly processes inputs and returns the expected output.

---

## Submission Contents

- **Live API Endpoint:**  
  The deployed API is accessible at:  
  `http://207.148.78.17:8005/similarity`  

- **Complete Code:**  
  Both Part A (model and embedding code) and Part B (API and deployment scripts) are provided as `.py` files.

- **Report:**  
  This report explains the main approach for both parts in a concise manner.

---

## Reflection

Deploying this sentence similarity model as a cloud-accessible API made it immediately usable for real-world applications. Key strengths of this deployment include:

Performance: Using cached model loading and CPU-friendly inference ensured fast response times.

Scalability: FastAPI’s non-blocking async design and uvicorn server allow handling multiple requests concurrently.

Portability: The deployment is cloud-agnostic and can easily be ported to AWS, Azure, or Docker environments.

However, there are also some limitations to consider:
- **Limited Model Flexibility**: Using a pre-trained model limits the model’s ability to adapt to new data.
- **Single Endpoint**: The API only supports a single similarity computation endpoint.
- **No Authentication**: The API is open to the public without any authentication or authorization.
- **Time Constraints**: The API is not optimized for high-frequency or low-latency use cases and complete model coudn't built in given 2 days timeframe. 
In summary, this deployment showcases the power of FastAPI and Hugging Face Transformers in building performant, scalable, and portable cloud services.
---

