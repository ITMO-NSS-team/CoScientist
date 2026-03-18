import os
import time


def update_activity(session_folder: str) -> None:
    """
    Updates the last activity timestamp for a session.
    
    This method records the time of the latest interaction with a session,
    allowing the system to track session usage and potentially manage resources
    or provide context-aware features. It writes the current timestamp to a 
    hidden file within the session directory.
    
    Args:
        session_folder: The path to the session folder.
    
    Returns:
        None
    """
    with open(os.path.join(session_folder, ".last_activity"), "w") as f:
        f.write(str(time.time()))


