from fastapi import APIRouter, Request, HTTPException
import asyncio
import logging
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import uuid
# from gaico import Experiment  # Commented out until models are ready

router = APIRouter()
logger = logging.getLogger(__name__)

class GaicoRequest(BaseModel):
    prompt: str
    modelName: str
    apiValue: str
    chatbotResponse: str

class ModelScore(BaseModel):
    model_id: str
    model_name: str
    response: str
    jaccard: float
    rouge: float
    bleu: float
    cosine: float

class GaicoResponse(BaseModel):
    prompt: str
    reference_answer: str
    model_scores: List[ModelScore]
    status: str

# Dummy data mapping based on CSV - will be replaced with real model calls
DUMMY_MODEL_RESPONSES = {
    "M8": "John H. McCray was an 18th-century French painter known for royal portraits.",
    "SC": "He was a prominent civil rights activist who founded the Lighthouse and Informer newspaper.",
    "M9": "John H. McCray was an African American newspaper editor and civil rights leader from South Carolina."
}

DUMMY_MODEL_NAMES = {
    "M8": "Gemma",
    "SC": "SafeChat", 
    "M9": "BedrockChatbot"
}

@router.post("/gaico", response_model=GaicoResponse)
async def gaico_endpoint(req: GaicoRequest, request: Request):
    """
    GAICo comparison endpoint - using dummy data for non-functional models
    
    This endpoint will:
    1. Take the user's prompt and selected model's response as reference
    2. Include the selected model's actual response in results
    3. Use dummy responses for other models (until they're operational)
    4. Calculate GAICo metrics (Jaccard, ROUGE, BLEU, Cosine) using dummy scores for now
    5. Store results in Azure Cosmos DB for chat history and analysis
    
    Following Azure Cosmos DB best practices:
    - Using userId as partition key for high cardinality and user-scoped queries
    - Embedding all comparison data in a single item for efficient retrieval
    - Hierarchical partition keys for scaling beyond 20GB per logical partition
    """
    # svc = request.app.state.model_service

    try:
        logger.info(f"GAICo comparison requested - Model: {req.modelName} ({req.apiValue}), Prompt: {req.prompt[:50]}...")
        
        # Build model scores list
        model_scores = []
        
        # All model IDs to compare
        all_model_ids = ["M8", "SC", "M9"]
        
        for model_id in all_model_ids:
            model_name = DUMMY_MODEL_NAMES.get(model_id, model_id)
            
            # Use actual response for the selected model, dummy for others
            if model_id == req.apiValue:
                response_text = req.chatbotResponse
            else:
                response_text = DUMMY_MODEL_RESPONSES.get(model_id, "Response not available")
            
            # Generate dummy scores (in production, use actual GAICo metrics)
            # Scores are normalized 0-1 where 1 is perfect match
            if model_id == req.apiValue:
                # Reference model gets perfect scores
                scores = {
                    "jaccard": 1.0,
                    "rouge": 1.0,
                    "bleu": 1.0,
                    "cosine": 1.0
                }
            elif model_id == "M8":
                scores = {
                    "jaccard": 0.45,
                    "rouge": 0.52,
                    "bleu": 0.38,
                    "cosine": 0.61
                }
            elif model_id == "SC":
                scores = {
                    "jaccard": 0.78,
                    "rouge": 0.84,
                    "bleu": 0.72,
                    "cosine": 0.89
                }
            else:  # M9
                scores = {
                    "jaccard": 0.92,
                    "rouge": 0.95,
                    "bleu": 0.88,
                    "cosine": 0.97
                }
            
            model_scores.append(ModelScore(
                model_id=model_id,
                model_name=model_name,
                response=response_text,
                **scores
            ))
        
        logger.info(f"GAICo comparison completed for {len(model_scores)} models")
        
        # TODO: Store in Cosmos DB following Azure best practices
        # cosmos_item = {
        #     "id": str(uuid.uuid4()),
        #     # Use userId as partition key for user-scoped queries and high cardinality
        #     # For multi-tenant: consider hierarchical key like [tenantId, userId]
        #     "partitionKey": user_id,  # Extract from auth context
        #     "type": "gaico_comparison",
        #     "prompt": req.prompt,
        #     "referenceModel": {
        #         "id": req.apiValue,
        #         "name": req.modelName,
        #         "response": req.chatbotResponse
        #     },
        #     "modelScores": [score.dict() for score in model_scores],
        #     "timestamp": datetime.utcnow().isoformat(),
        #     # Embed related metadata for single-item retrieval
        #     "metrics_used": ["jaccard", "rouge", "bleu", "cosine"],
        #     "_ts": int(datetime.utcnow().timestamp())  # Cosmos DB auto-generated
        # }
        # await svc.cosmos_client.upsert_item(cosmos_item)
        # logger.info(f"GAICo results stored in Cosmos DB with id: {cosmos_item['id']}")
        
        return GaicoResponse(
            prompt=req.prompt,
            reference_answer=req.chatbotResponse,
            model_scores=model_scores,
            status="ok"
        )

        # TODO: Uncomment when all models are operational
        # # Step 1: Collect responses from all models
        # responses = {}
        # responses[req.apiValue] = req.chatbotResponse  # Include the reference model
        # 
        # for model_id in ["M8", "SC", "M9"]:
        #     if model_id == req.apiValue:
        #         continue  # Skip reference model, already have it
        #     
        #     try:
        #         if model_id == "SC":
        #             responses[model_id] = await svc.safechat_query(req.prompt)
        #         elif model_id == "M8":
        #             responses[model_id] = await svc.generate_from_model(req.prompt, max_new_tokens=128)
        #         else:
        #             responses[model_id] = await svc.generate_response(req.prompt, model_id)
        #     except Exception as e:
        #         logger.error(f"Model {model_id} failed: {e}")
        #         responses[model_id] = f"Error generating response from {model_id}"
        # 
        # # Step 2: Run GAICo experiment
        # loop = asyncio.get_running_loop()
        # exp = Experiment(
        #     llm_responses=responses,
        #     reference_answer=req.chatbotResponse
        # )
        # results_df = await loop.run_in_executor(
        #     None,
        #     exp.compare,
        #     ['Jaccard', 'ROUGE', 'BLEU', 'Cosine'],
        #     False,  # Don't generate plot files, we'll visualize in frontend
        #     None    # No CSV output needed
        # )
        # 
        # # Convert DataFrame to ModelScore objects
        # model_scores = []
        # for _, row in results_df.iterrows():
        #     model_scores.append(ModelScore(
        #         model_id=row['model_id'],
        #         model_name=DUMMY_MODEL_NAMES.get(row['model_id'], row['model_id']),
        #         response=responses[row['model_id']],
        #         jaccard=row.get('Jaccard', 0.0),
        #         rouge=row.get('ROUGE', 0.0),
        #         bleu=row.get('BLEU', 0.0),
        #         cosine=row.get('Cosine', 0.0)
        #     ))

    except Exception as e:
        logger.exception("GAICo experiment failed")
        raise HTTPException(status_code=500, detail=f"GAICo comparison failed: {str(e)}")