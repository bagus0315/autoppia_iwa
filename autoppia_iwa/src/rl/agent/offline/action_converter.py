#!/usr/bin/env python3
"""
Action format converter for BC training.

Converts actions from leaderboard dataset format to BaseAction format.
"""

from __future__ import annotations

from loguru import logger


def convert_leaderboard_action_to_base_action_dict(action_data: dict) -> dict | None:
    """
    Convert leaderboard action format to BaseAction-compatible dict.
    
    Leaderboard format:
    {
      "type": "input",
      "attributes": {
        "text": "...",
        "selector": {...}
      }
    }
    
    BaseAction format:
    {
      "type": "TypeAction",
      "text": "...",
      "selector": {...}
    }
    
    Args:
        action_data: Action dict from leaderboard dataset
        
    Returns:
        Converted action dict, or None if conversion fails
    """
    try:
        action_type = action_data.get("type", "").lower()
        attributes = action_data.get("attributes", {})
        
        # Map action types (input -> type, others stay the same)
        type_mapping = {
            "navigate": "navigate",
            "click": "click",
            "type": "type",
            "input": "type",  # "input" is actually "type" action
            "wait": "wait",
            "scroll": "scroll",
            "hover": "hover",
            "select": "select",
        }
        
        base_action_type = type_mapping.get(action_type)
        if not base_action_type:
            logger.debug(f"Unknown action type: {action_type}")
            return None
        
        # Start with the base type
        converted = {"type": base_action_type}
        
        # Flatten attributes into main dict
        for key, value in attributes.items():
            converted[key] = value
        
        # Handle specific action types
        if base_action_type == "navigate":
            # navigate needs url or go_back/go_forward
            if not converted.get("url") and not converted.get("go_back") and not converted.get("go_forward"):
                # Default to no navigation
                return None
                
        elif base_action_type == "click":
            # click needs either selector or x/y coordinates
            if not converted.get("selector") and ("x" not in converted or "y" not in converted):
                # No valid click target
                return None
                
        elif base_action_type == "type":
            # type needs text and selector
            if not converted.get("text") or not converted.get("selector"):
                return None
                
        elif base_action_type == "wait":
            # wait needs either selector or time_seconds
            if not converted.get("selector") and not converted.get("time_seconds"):
                # Try to use timeout_seconds as time_seconds
                if "timeout_seconds" in converted:
                    converted["time_seconds"] = converted.pop("timeout_seconds")
                else:
                    return None
        
        return converted
        
    except Exception as e:
        logger.debug(f"Failed to convert action: {e}")
        return None


def convert_actions_batch(actions_data: list[dict]) -> list[dict]:
    """
    Convert a batch of leaderboard actions to BaseAction-compatible dicts.
    
    Args:
        actions_data: List of action dicts from leaderboard dataset
        
    Returns:
        List of converted action dicts (skips failed conversions)
    """
    converted = []
    for action_data in actions_data:
        result = convert_leaderboard_action_to_base_action_dict(action_data)
        if result:
            converted.append(result)
    return converted

