from ..state import GraphState
from ..api.user import get_user_and_rewards_data_async
from ..utils import logger

async def fetch_user_data_node(state: GraphState):
    """Fetch user data from MongoDB."""
    user_id = state.get("user_id")
    if not user_id:
        return {
            "user_data": None,
            "error_message": "User ID missing."
        }
        
    data = await get_user_and_rewards_data_async(user_id)
    
    if data.get("error"):
        logger.error(f"User data fetch failed: {data['error']}")
        return {
            "user_data": None,
            "error_message": data["error"]
        }
    
    return {
        "user_data": data,
        "error_message": None
    }
