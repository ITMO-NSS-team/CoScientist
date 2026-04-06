"""Human-in-the-Loop (HITL) module for CoScientist agents."""

from CoScientist.hitl.models import HITLAction, HITLRequest, HITLResponse
from CoScientist.hitl.handler import AbstractHITLHandler, ConsoleHITLHandler, CallbackHITLHandler
from CoScientist.hitl.tool import HITLToolset

# Variant B (callbacks) is commented out in callbacks.py.
# Uncomment and import when needed:
# from CoScientist.hitl.callbacks import make_hitl_after_callback, make_hitl_before_callback

__all__ = [
    "HITLAction",
    "HITLRequest",
    "HITLResponse",
    "AbstractHITLHandler",
    "ConsoleHITLHandler",
    "CallbackHITLHandler",
    "HITLToolset",
]
