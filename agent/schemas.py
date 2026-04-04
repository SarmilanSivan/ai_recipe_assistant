"""
Defining the structured output

"""

from pydantic import BaseModel, Field
from typing import TypedDict, List, Dict, Any, Literal
from typing import List, Optional

class ExtractionResult(BaseModel):
    """The structured output the LLM must generate"""
    ingredients: List[str] = Field(default_factory=list, description="List of ingredients the user wants to use.")
    avoid_ingredients: List[str] = Field(default_factory=list, description="Ingredients to strictly avoid.")
    max_calories: Optional[int] = Field(default=None, description="Maximum calories per serving.")
    max_prep_time: Optional[int] = Field(default=None, description="Maximum prep time in minutes.")
    # mode_of_prep: Optional[str] = Field(default=None, description="E.g., Stovetop, Oven, Microwave, Slow cooker.")
    is_sugar_free: bool = Field(default=False, description="True if the user requests a sugar-free meal.")
    
    is_complete: bool = Field(
        default=False, 
        description="Set to True ONLY if you have gathered at least ingredients, max_prep_time, and max_calories. Otherwise, False."
    )
    
    ai_message: str = Field(
        description="Your conversational response to the user. If is_complete is False, ask a natural question to gather the missing fields. If True, confirm you are searching."
    )

class VerificationResult(BaseModel):
    is_valid: bool = Field(
        description="True ONLY if the draft matches all user constraints and accurately reflects the database results. False if there are hallucinations."
    )
    feedback: str = Field(
        description="If is_valid is False, provide a strict, 1-sentence instruction on what needs to be fixed. If True, output 'Looks good'."
    )