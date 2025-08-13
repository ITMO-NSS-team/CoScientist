from pydantic import BaseModel, Field
from typing import Union, List, Optional


class Response(BaseModel):
    """
    Response to user.
    """

    response: str

class Plan(BaseModel):
    """
    Plan to follow in future
    """

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

class Act(BaseModel):
    """
    Action to perform.
    """

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

class Worker(BaseModel):
    """
    Worker to call in future
    """

    next: str = Field(
        description="Next worker to call"
    )


class Chat(BaseModel):
    """
    Action to perform
    """


    action: Union[Response, Worker] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Next."
    )

class Translation(BaseModel):
    """
    Action to perform
    """


    language: Optional[str] = Field(
        description="language to translate",
        default='English'
    )
    translation: Optional[str] = Field(
        default=None,
        description="translation from English"
    )
