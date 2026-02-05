from fastapi import APIRouter, Request, HTTPException
import asyncio
import logging
from pydantic import BaseModel
from gaico import Experiement
from typing import List

router = APIRouter()
logger = logging.getLogger(__name__)


class GaicoRequest(BaseModel):
    input: str
    ideal: str
    chatbot_responses = List[str]

class GaicoResponse(BaseModel):



@router.post("/gaico", response_model=GaicoResponse)
async def gaico_endpoint(req: GaicoRequest, request: Request):
    svc = request.app.state.model_service

    # Ensure every model has a response (call generation if empty)
    responses = dict(req.responses)  # shallow copy
    for model_name, resp in responses.items():
        if resp in (None, ""):
            if model_name == "SC":
                responses[model_name] = await svc.safchat_query(req.input)
            elif model_name == "M8":
                # default max tokens; could be passed in schema if needed
                responses[model_name] = await svc.generate_from_model(
                    req.input, max_new_tokens=128
                )
            else:
                responses[model_name] = f"<no generator for {model_name}>"

    # Build and run Experiment (this may be CPU-bound; run in executor if needed)
    try:
        loop = asyncio.get_running_loop()
        # run compare() in executor because it might be CPU/IO heavy
        exp = Experiment(llm_responses=responses, reference_answer=req.ideal)
        results_df = await loop.run_in_executor(None, exp.compare, True, "bin/out.csv")

        logger.info("Experiment compare complete")
        return {"status": "ok", "output_csv": "bin/out.csv"}
    except Exception as e:
        logger.exception("Gaico experiment failed")
        raise HTTPException(status_code=500, detail=str(e))
