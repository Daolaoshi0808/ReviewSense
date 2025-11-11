from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import pandas as pd
import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_openai import ChatOpenAI
import kagglehub
import numpy as np
from pprint import pprint
import random
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from dotenv import load_dotenv
from langchain import hub
import time
from typing import List, Dict, Any
from langchain_pinecone import PineconeVectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pandas import DataFrame
from langchain.docstore.document import Document
import torch
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
load_dotenv()

# Assuming the dataframe has been upserted to Pinecone
final_review_chunked_df = pd.read_csv('final_review_chunked_df.csv')

pc = Pinecone()

# Define your index name
index_name = 'food-review-info'

# Connect to the index
index = pc.Index(index_name)

class EmbeddingModel:
    def __init__(self, model):
        self.model = TextEmbedding(model_name=model)
        
    def embed_documents(self, splits):
        # Use self.model instead of embedding_model
        # Also, batch processing if your TextEmbedding supports it would be more efficient
        return [list(self.model.embed(split.page_content))[0].tolist() for split in splits]
        
    def embed_query(self, query):
        # This method needs to accept a query string directly, not a Document object
        return list(self.model.embed(query))[0].tolist()

embeddings = EmbeddingModel("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
text_field = "raw_text_index"
vectorstore = PineconeVectorStore(  
    index, embeddings, text_field
)  
time.sleep(10)

class CustomRetriever(BaseRetriever):
    vectorstore: PineconeVectorStore
    df: DataFrame

    def _get_relevant_documents(self, query):
        docs = self.vectorstore.similarity_search(query, k=5)
        outputs = []
        for doc in docs:
        # Retrieve the original text from your DataFrame
            raw_text = self.df.loc[int(doc.id), 'Text']
        # Turn that into a Document
            outputs.append(Document(page_content=raw_text))
        return outputs

retriever = CustomRetriever(vectorstore=vectorstore, df=final_review_chunked_df)
template_prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model="gpt-3.5-turbo", seed=0)
rag_chain = template_prompt | llm | StrOutputParser()

checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"
base_model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Load your LoRA adapter
peft_model_path = "./my_model"  # Path to your saved LoRA adapter
model = PeftModel.from_pretrained(base_model, peft_model_path)

# Set the model to evaluation mode
model.eval()

generator_params = {
    "max_new_tokens": 10, 
    "temperature": 0.1,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id
}

def score_review(review_text):
    # Format the input for the fine-tuned model
    question_template = '''
    You will be provided Amazon Fine Food Review from a user, and you need to guess the score that user rates from the review. There are 5 possible scores: 1,2,3,4, and 5, where 1 is the 
    lowest score and 5 is the highest score. You can Identify key descriptive words and phrases, determine the tone or emotion conveyed by these words or phrases,
    and summarize how these descriptors combine to reflect an overall sentiment.
    Place the score at the end of the response with a space and then the final answer. Like if the score if 4, then give it as:  4 with extra space before 4. 
    '''
    
    prompt = question_template + review_text
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Tokenize input
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate attention mask
    attention_mask = torch.ones_like(inputs)

    with torch.no_grad():
        outputs = model.generate(inputs, attention_mask=attention_mask, **generator_params)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the numeric score (1-5)
    try:
        # Look for the last digit in the result
        for char in reversed(result):
            if char.isdigit() and int(char) in [1, 2, 3, 4, 5]:
                return int(char)
        # Default to 3 if no valid score found
        return 3
    except:
        return 3

lora_scorer = score_review

base_model = AutoModelForCausalLM.from_pretrained(checkpoint)
def score_review_base(review_text):
    # Format the input for the fine-tuned model
    question_template = '''
    You will be provided Amazon Fine Food Review from a user, and you need to guess the score that user rates from the review. There are 5 possible scores: 1,2,3,4, and 5, where 1 is the 
    lowest score and 5 is the highest score. You can Identify key descriptive words and phrases, determine the tone or emotion conveyed by these words or phrases,
    and summarize how these descriptors combine to reflect an overall sentiment.
    Place the score at the end of the response with a space and then the final answer. Like if the score if 4, then give it as:  4 with extra space before 4. 
    '''
    
    prompt = question_template + review_text
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Tokenize input
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate attention mask
    attention_mask = torch.ones_like(inputs)

    with torch.no_grad():
        outputs = base_model.generate(inputs, attention_mask=attention_mask, **generator_params)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the numeric score (1-5)
    try:
        # Look for the last digit in the result
        for char in reversed(result):
            if char.isdigit() and int(char) in [1, 2, 3, 4, 5]:
                return int(char)
        # Default to 3 if no valid score found
        return 3
    except:
        return 3

lora_scorer_base = score_review_base

positive_system = """You are an expert at analyzing food product reviews to identify advantages and positive aspects.
Extract specific benefits, advantages, and positive qualities mentioned in the review.
Focus on aspects like taste, quality, value, convenience, nutritional benefits, or unique selling points.
Be specific and detail-oriented in your analysis."""

positive_prompt = ChatPromptTemplate.from_messages([
    ("system", positive_system),
    ("human", "Review: {review}\n\nWhat specific advantages or positive aspects are mentioned in this review?")
])

# Prompt for analyzing disadvantages/negative aspects
negative_system = """You are an expert at analyzing food product reviews to identify disadvantages and drawbacks.
Extract specific issues, complaints, or negative qualities mentioned in the review.
Focus on aspects like taste problems, quality issues, value concerns, inconvenience factors, or health drawbacks.
Be specific and detail-oriented in your analysis."""

negative_prompt = ChatPromptTemplate.from_messages([
    ("system", negative_system),
    ("human", "Review: {review}\n\nWhat specific disadvantages or drawbacks are mentioned in this review?")
])

positive_analyzer = positive_prompt | llm | StrOutputParser()
negative_analyzer = negative_prompt | llm | StrOutputParser()

system = """You are an expert food product analyst tasked with creating a comprehensive, balanced summary of a product.
Based on the advantages and disadvantages provided, create a well-organized summary that helps consumers make an informed decision.
Structure your response with clear sections for:
1. Product Overview (brief description of what the product is)
2. Key Advantages (organized by themes like taste, quality, value, etc.)
3. Notable Disadvantages (organized by themes like taste, quality, value, etc.)
4. Overall Assessment (balanced conclusion about the product)

Be specific, balanced, and factual in your assessment."""

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Product: {product}\n\nAdvantages identified from positive reviews:\n{advantages}\n\nDisadvantages identified from negative reviews:\n{disadvantages}\n\nPlease create a comprehensive summary.")
])

summary_generator = summary_prompt | llm | StrOutputParser()

# Define the Graph
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    product: str
    documents: List[Document]
    review_scores: List[Dict[str, Any]]  # Each dict contains review text and its score
    positive_reviews: List[str]
    negative_reviews: List[str]
    advantages: List[str]
    disadvantages: List[str]
    summary: str

# Adding Nodes
def extract_product(state):
    """Extract product name from question"""
    print("---EXTRACTING PRODUCT NAME---")
    question = state["question"]
    
    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that extracts the specific food product name from a user question."),
        ("human", "From the following question, what is the specific food product being asked about? Return only the product name.\n\nQuestion: {question}")
    ])
    
    extract_chain = extract_prompt | llm | StrOutputParser()
    product = extract_chain.invoke({"question": question})
    
    print(f"Extracted product: {product}")
    return {"question": question, "product": product}

# Node 2: Retrieve reviews
def retrieve_reviews(state):
    """Retrieve reviews about the product"""
    print("---RETRIEVING REVIEWS---")
    question = state["question"]
    product = state["product"]
    
    # Create a search query focused on the product
    search_query = f"reviews about {product}"
    
    # Retrieve documents
    documents = retriever.invoke(search_query)
    print(f"Retrieved {len(documents)} reviews")
    
    return {"question": question, "product": product, "documents": documents}

# Node 3: Score reviews
def score_reviews(state):
    """Score each review using the LoRA model"""
    print("---SCORING REVIEWS---")
    documents = state["documents"]
    product = state["product"]
    
    review_scores = []
    
    for doc in documents:
        review_text = doc.page_content
        score = lora_scorer(review_text)
        
        review_scores.append({
            "review": review_text,
            "score": score
        })
        print(f"Review scored: {score}/5")
    
    # Separate positive and negative reviews
    positive_reviews = [r["review"] for r in review_scores if r["score"] >= 4]
    negative_reviews = [r["review"] for r in review_scores if r["score"] <= 3]
    
    print(f"Found {len(positive_reviews)} positive reviews and {len(negative_reviews)} negative reviews")
    
    return {
        "question": state["question"],
        "product": product,
        "documents": documents,
        "review_scores": review_scores,
        "positive_reviews": positive_reviews,
        "negative_reviews": negative_reviews
    }

# Node 4: Analyze advantages
def analyze_advantages(state):
    """Analyze positive reviews to extract advantages"""
    print("---ANALYZING ADVANTAGES---")
    positive_reviews = state["positive_reviews"]
    product = state["product"]
    
    advantages = []
    
    for review in positive_reviews:
        advantage = positive_analyzer.invoke({"review": review})
        advantages.append(advantage)
        print(f"Extracted advantage: {advantage[:50]}...")
    
    return {
        "question": state["question"],
        "product": product,
        "documents": state["documents"],
        "review_scores": state["review_scores"],
        "positive_reviews": positive_reviews,
        "negative_reviews": state["negative_reviews"],
        "advantages": advantages
    }

# Node 5: Analyze disadvantages
def analyze_disadvantages(state):
    """Analyze negative reviews to extract disadvantages"""
    print("---ANALYZING DISADVANTAGES---")
    negative_reviews = state["negative_reviews"]
    
    disadvantages = []
    
    for review in negative_reviews:
        disadvantage = negative_analyzer.invoke({"review": review})
        disadvantages.append(disadvantage)
        print(f"Extracted disadvantage: {disadvantage[:50]}...")
    
    return {
        "question": state["question"],
        "product": state["product"],
        "documents": state["documents"],
        "review_scores": state["review_scores"],
        "positive_reviews": state["positive_reviews"],
        "negative_reviews": negative_reviews,
        "advantages": state["advantages"],
        "disadvantages": disadvantages
    }

# Node 6: Generate summary
def generate_summary(state):
    """Generate a comprehensive summary of advantages and disadvantages"""
    print("---GENERATING SUMMARY---")
    product = state["product"]
    advantages = state["advantages"]
    disadvantages = state["disadvantages"]
    
    advantages_text = "\n".join([f"- {adv}" for adv in advantages])
    disadvantages_text = "\n".join([f"- {dis}" for dis in disadvantages])
    
    summary = summary_generator.invoke({
        "product": product,
        "advantages": advantages_text,
        "disadvantages": disadvantages_text
    })
    
    print(f"Summary generated: {summary[:100]}...")
    
    return {
        "question": state["question"],
        "product": state["product"],
        "documents": state["documents"],
        "review_scores": state["review_scores"],
        "positive_reviews": state["positive_reviews"],
        "negative_reviews": state["negative_reviews"],
        "advantages": state["advantages"],
        "disadvantages": state["disadvantages"],
        "summary": summary
    }

def reformulate_query(state):
    """Reformulate the query to get better search results"""
    print("---REFORMULATING QUERY---")
    question = state["question"]
    product = state["product"]
    
    reformulate_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at reformulating search queries to find more relevant information."),
        ("human", "I'm trying to find reviews about {product} but haven't found enough. Please reformulate this search query to find more relevant reviews: {question}")
    ])
    
    reformulate_chain = reformulate_prompt | llm | StrOutputParser()
    new_query = reformulate_chain.invoke({"product": product, "question": question})
    
    print(f"Reformulated query: {new_query}")
    
    return {
        "question": new_query,  # Update the question with the reformulated query
        "product": state["product"]
    }

def analyze_beverage(state):
    """Specialized analysis for beverage products"""
    print("---SPECIALIZED BEVERAGE ANALYSIS---")
    positive_reviews = state["positive_reviews"]
    product = state["product"]
    
    beverage_system = """You are an expert beverage analyst specialized in analyzing coffee, tea, and other drink products.
    Extract specific benefits and advantages mentioned in beverage reviews, with special attention to:
    - Flavor profile and taste notes
    - Brewing characteristics
    - Aroma
    - Aftertaste
    - Consistency between cups/bottles
    - Packaging quality
    - Value for money
    Be specific and detail-oriented in your analysis."""
    
    beverage_prompt = ChatPromptTemplate.from_messages([
        ("system", beverage_system),
        ("human", "Product: {product}\n\nReview: {review}\n\nWhat specific advantages or positive aspects are mentioned in this beverage review?")
    ])
    
    beverage_chain = beverage_prompt | llm | StrOutputParser()
    
    advantages = []
    for review in positive_reviews:
        advantage = beverage_chain.invoke({"product": product, "review": review})
        advantages.append(advantage)
        print(f"Extracted beverage advantage: {advantage[:50]}...")
    
    return {
        "question": state["question"],
        "product": product,
        "documents": state["documents"],
        "review_scores": state["review_scores"],
        "positive_reviews": positive_reviews,
        "negative_reviews": state["negative_reviews"],
        "advantages": advantages  # Store the advantages for later use
    }

def analyze_snack(state):
    """Specialized analysis for snack food products"""
    print("---SPECIALIZED SNACK FOOD ANALYSIS---")
    positive_reviews = state["positive_reviews"]
    product = state["product"]
    
    snack_system = """You are an expert snack food analyst specialized in analyzing bars, chips, crackers, and other snack products.
    Extract specific benefits and advantages mentioned in snack food reviews, with special attention to:
    - Taste and flavor profile
    - Texture and mouthfeel
    - Freshness
    - Portion size and packaging
    - Nutritional benefits
    - Convenience factors
    - Value for money
    Be specific and detail-oriented in your analysis."""
    
    snack_prompt = ChatPromptTemplate.from_messages([
        ("system", snack_system),
        ("human", "Product: {product}\n\nReview: {review}\n\nWhat specific advantages or positive aspects are mentioned in this snack food review?")
    ])
    
    snack_chain = snack_prompt | llm | StrOutputParser()
    
    advantages = []
    for review in positive_reviews:
        advantage = snack_chain.invoke({"product": product, "review": review})
        advantages.append(advantage)
        print(f"Extracted snack advantage: {advantage[:50]}...")
    
    return {
        "question": state["question"],
        "product": product,
        "documents": state["documents"],
        "review_scores": state["review_scores"],
        "positive_reviews": positive_reviews,
        "negative_reviews": state["negative_reviews"],
        "advantages": advantages  # Store the advantages for later use
    }

# Node 3: Score reviews with base model (no LoRA)
def score_reviews_basic(state):
    """Score each review using the base model without LoRA fine-tuning"""
    print("---SCORING REVIEWS---")
    documents = state["documents"]
    product = state["product"]
    
    review_scores = []
    
    for doc in documents:
        review_text = doc.page_content
        score = lora_scorer_base(review_text)
        
        review_scores.append({
            "review": review_text,
            "score": score
        })
        print(f"Review scored: {score}/5")
    
    # Separate positive and negative reviews
    positive_reviews = [r["review"] for r in review_scores if r["score"] >= 4]
    negative_reviews = [r["review"] for r in review_scores if r["score"] <= 2]
    
    print(f"Found {len(positive_reviews)} positive reviews and {len(negative_reviews)} negative reviews")
    
    return {
        "question": state["question"],
        "product": product,
        "documents": documents,
        "review_scores": review_scores,
        "positive_reviews": positive_reviews,
        "negative_reviews": negative_reviews
    }

# Edge Functions
def check_review_count(state):
    """Determines whether enough reviews were found"""
    if len(state["documents"]) < 3:
        print("---INSUFFICIENT REVIEWS FOUND, REFORMULATING QUERY---")
        return "insufficient_reviews"
    else:
        print(f"---SUFFICIENT REVIEWS FOUND: {len(state['documents'])}---")
        return "sufficient_reviews"

def determine_product_category(state):
    """Determines product category for specialized analysis"""
    product = state["product"].lower()
    if "coffee" in product or "tea" in product or "drink" in product or "water" in product or "juice" in product:
        print(f"---PRODUCT CATEGORY: BEVERAGE---")
        return "beverage"
    elif "bar" in product or "snack" in product or "chip" in product or "cracker" in product:
        print(f"---PRODUCT CATEGORY: SNACK FOOD---")
        return "snack_food"
    else:
        print(f"---PRODUCT CATEGORY: GENERAL FOOD---")
        return "general_food"

# Build workflow with base model (no LoRA fine-tuning)
print("Building base model workflow...")
workflow_base = StateGraph(GraphState)

workflow_base.add_node("extract_product", extract_product)
workflow_base.add_node("retrieve_reviews", retrieve_reviews)
workflow_base.add_node("score_reviews_basic", score_reviews_basic)
workflow_base.add_node("analyze_advantages", analyze_advantages)
workflow_base.add_node("analyze_disadvantages", analyze_disadvantages)
workflow_base.add_node("generate_summary", generate_summary)
workflow_base.add_node("reformulate_query", reformulate_query)
workflow_base.add_node("analyze_beverage", analyze_beverage)
workflow_base.add_node("analyze_snack", analyze_snack)

# Add edges with conditions
workflow_base.add_edge(START, "extract_product")
workflow_base.add_edge("extract_product", "retrieve_reviews")

# Conditional edge: Check if we have enough reviews
workflow_base.add_conditional_edges(
    "retrieve_reviews",
    check_review_count,
    {
        "insufficient_reviews": "reformulate_query",
        "sufficient_reviews": "score_reviews_basic",
    },
)

# Edge from reformulate_query back to retrieve_reviews
workflow_base.add_edge("reformulate_query", "retrieve_reviews")

# Conditional edge: Route based on product category
workflow_base.add_conditional_edges(
    "score_reviews_basic",
    determine_product_category,
    {
        "beverage": "analyze_beverage",
        "snack_food": "analyze_snack",
        "general_food": "analyze_advantages",
    },
)
workflow_base.add_edge("analyze_beverage", "analyze_disadvantages")
workflow_base.add_edge("analyze_snack", "analyze_disadvantages")
workflow_base.add_edge("analyze_advantages", "analyze_disadvantages")

# Conditional edge: Check if we have enough data for a meaningful summary
workflow_base.add_conditional_edges(
    "analyze_disadvantages",
    lambda state: "sufficient_data" 
                 if (len(state.get("advantages", [])) > 0 or len(state.get("disadvantages", [])) > 0) 
                 else "insufficient_data",
    {
        "sufficient_data": "generate_summary",
        "insufficient_data": END,  # End with a default message if not enough data
    },
)

workflow_base.add_edge("generate_summary", END)

# Compile
app_base = workflow_base.compile()

# Build workflow with LoRA fine-tuned model
print("Building LoRA fine-tuned model workflow...")
workflow = StateGraph(GraphState)

workflow.add_node("extract_product", extract_product)
workflow.add_node("retrieve_reviews", retrieve_reviews)
workflow.add_node("score_reviews", score_reviews)
workflow.add_node("analyze_advantages", analyze_advantages)
workflow.add_node("analyze_disadvantages", analyze_disadvantages)
workflow.add_node("generate_summary", generate_summary)
workflow.add_node("reformulate_query", reformulate_query)
workflow.add_node("analyze_beverage", analyze_beverage)
workflow.add_node("analyze_snack", analyze_snack)

# Add edges with conditions
workflow.add_edge(START, "extract_product")
workflow.add_edge("extract_product", "retrieve_reviews")

# Conditional edge: Check if we have enough reviews
workflow.add_conditional_edges(
    "retrieve_reviews",
    check_review_count,
    {
        "insufficient_reviews": "reformulate_query",
        "sufficient_reviews": "score_reviews",
    },
)

# Edge from reformulate_query back to retrieve_reviews
workflow.add_edge("reformulate_query", "retrieve_reviews")

# Conditional edge: Route based on product category
workflow.add_conditional_edges(
    "score_reviews",
    determine_product_category,
    {
        "beverage": "analyze_beverage",
        "snack_food": "analyze_snack",
        "general_food": "analyze_advantages",
    },
)
workflow.add_edge("analyze_beverage", "analyze_disadvantages")
workflow.add_edge("analyze_snack", "analyze_disadvantages")
workflow.add_edge("analyze_advantages", "analyze_disadvantages")

# Conditional edge: Check if we have enough data for a meaningful summary
workflow.add_conditional_edges(
    "analyze_disadvantages",
    lambda state: "sufficient_data" 
                 if (len(state.get("advantages", [])) > 0 or len(state.get("disadvantages", [])) > 0) 
                 else "insufficient_data",
    {
        "sufficient_data": "generate_summary",
        "insufficient_data": END,  # End with a default message if not enough data
    },
)

workflow.add_edge("generate_summary", END)

# Compile
app = workflow.compile()

# Simple CLI interface
def process_question(question, use_lora=True):
    print("\n\n" + "="*80)
    print(f"PROCESSING QUESTION: {question}")
    print("="*80 + "\n")
    
    runconfig = {"recursion_limit": 50}
    inputs = {"question": question}
    
    # Choose which workflow to use
    if use_lora:
        print("Using advanced RAG with LoRA fine-tuned model...")
        result = app.invoke(inputs, config=runconfig)
    else:
        print("Using advanced RAG with base model (no LoRA)...")
        result = app_base.invoke(inputs, config=runconfig)
    
    # Return the final summary
    return result.get("summary", "No summary generated.")

def main():
    print("\nAdvanced RAG System with LoRA Fine-tuning")
    print("----------------------------------------\n")
    
    while True:
        print("\nOptions:")
        print("1. Ask a question with LoRA fine-tuned model")
        print("2. Ask a question with base model (no LoRA)")
        print("3. Compare both models on same question")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            question = input("\nEnter your question about food products: ")
            result = process_question(question, use_lora=True)
            print("\nRESULT:")
            print("-"*80)
            print(result)
            print("-"*80)
            
        elif choice == '2':
            question = input("\nEnter your question about food products: ")
            result = process_question(question, use_lora=False)
            print("\nRESULT:")
            print("-"*80)
            print(result)
            print("-"*80)
            
        elif choice == '3':
            question = input("\nEnter your question about food products: ")
            
            print("\nProcessing with base model...")
            base_result = process_question(question, use_lora=False)
            
            print("\nProcessing with LoRA fine-tuned model...")
            lora_result = process_question(question, use_lora=True)
            
            print("\nBASE MODEL RESULT:")
            print("-"*80)
            print(base_result)
            print("-"*80)
            
            print("\nLoRA FINE-TUNED MODEL RESULT:")
            print("-"*80)
            print(lora_result)
            print("-"*80)
            
        elif choice == '4':
            print("\nExiting the program. Goodbye!")
            break
            
        else:
            print("\nInvalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()