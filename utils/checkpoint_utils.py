import json
from pathlib import Path
import torch
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from algorithms.base_agent import BaseAgent # Use string for type hint to avoid circular import at runtime

def save_checkpoint(agent: 'BaseAgent', dir_path: Path, step: int) -> None:
    """
    Saves the agent's state (networks, optimizers) to the specified directory.

    Args:
        agent: The agent instance whose state is to be saved.
        dir_path: The directory Path object to save the checkpoint to.
        step: The current training step number.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    save_data = {"step": step}

    # Actor network and optimizer
    if agent.actor is not None:
        torch.save(agent.actor.state_dict(), dir_path / "actor.pt")
    if agent.actor_optimizer is not None:
        torch.save(agent.actor_optimizer.state_dict(), dir_path / "opt_actor.pt")

    # Critic network and optimizer (optional, e.g., not for GRPO_NoCritic)
    if hasattr(agent, "critic") and agent.critic is not None:
        torch.save(agent.critic.state_dict(), dir_path / "critic.pt")
    if hasattr(agent, "critic_optimizer") and agent.critic_optimizer is not None:
        torch.save(agent.critic_optimizer.state_dict(), dir_path / "opt_critic.pt")
    
    # Save reference actor if it exists (for GRPO)
    if hasattr(agent, "actor_ref") and agent.actor_ref is not None:
        torch.save(agent.actor_ref.state_dict(), dir_path / "actor_ref.pt")


    with open(dir_path / "checkpoint_info.json", "w") as f:
        json.dump(save_data, f, indent=4)

    agent.logger.debug(f"Checkpoint saved at step {step} to {dir_path}")


def load_checkpoint(agent: 'BaseAgent', dir_path: Path) -> int:
    """
    Loads the agent's state from the specified directory.

    Args:
        agent: The agent instance to load the state into.
        dir_path: The directory Path object to load the checkpoint from.

    Returns:
        The training step number to resume from (0 if no checkpoint found).
    """
    start_step = 0
    info_path = dir_path / "checkpoint_info.json"
    actor_path = dir_path / "actor.pt"
    opt_actor_path = dir_path / "opt_actor.pt"
    critic_path = dir_path / "critic.pt"
    opt_critic_path = dir_path / "opt_critic.pt"
    actor_ref_path = dir_path / "actor_ref.pt"


    if not actor_path.is_file(): # Check for actor, as it's fundamental
        agent.logger.info(f"No actor checkpoint found at {actor_path}, starting fresh.")
        return start_step

    if info_path.is_file():
        with open(info_path, "r") as f:
            try:
                info = json.load(f)
                start_step = info.get("step", 0)
            except json.JSONDecodeError:
                agent.logger.warning(f"Could not decode checkpoint_info.json, step count might be incorrect.")
    else:
        agent.logger.warning(f"checkpoint_info.json not found in {dir_path}, assuming step 0.")


    agent.logger.info(f"Loading checkpoint from {dir_path}, resuming at step {start_step}")

    # Load actor
    if agent.actor is not None:
        agent.actor.load_state_dict(torch.load(actor_path, map_location=agent.device))
    if agent.actor_optimizer is not None and opt_actor_path.is_file():
        agent.actor_optimizer.load_state_dict(torch.load(opt_actor_path, map_location=agent.device))
    elif agent.actor_optimizer is not None and not opt_actor_path.is_file():
        agent.logger.warning(f"Actor optimizer checkpoint {opt_actor_path} not found.")


    # Load critic (optional)
    if hasattr(agent, "critic") and agent.critic is not None:
        if critic_path.is_file():
            agent.critic.load_state_dict(torch.load(critic_path, map_location=agent.device))
        else:
            agent.logger.warning(f"Critic checkpoint {critic_path} not found, but critic object exists.")
    
    if hasattr(agent, "critic_optimizer") and agent.critic_optimizer is not None:
        if opt_critic_path.is_file():
            agent.critic_optimizer.load_state_dict(torch.load(opt_critic_path, map_location=agent.device))
        elif agent.critic is not None: # Only warn if critic exists but optimizer checkpoint doesn't
             agent.logger.warning(f"Critic optimizer checkpoint {opt_critic_path} not found.")

    # Load reference actor if it exists (for GRPO)
    if hasattr(agent, "actor_ref") and agent.actor_ref is not None:
        if actor_ref_path.is_file():
            # Make sure agent.actor_ref is properly initialized before loading state dict
            if agent.actor is not None and agent.actor_ref is None: # Common case if _setup_networks isn't called before load
                import copy
                agent.actor_ref = copy.deepcopy(agent.actor).eval() # Initialize if missing
            
            if agent.actor_ref is not None: # Re-check after potential init
                 agent.actor_ref.load_state_dict(torch.load(actor_ref_path, map_location=agent.device))
                 for param in agent.actor_ref.parameters(): param.requires_grad = False # Ensure it's frozen
            else:
                agent.logger.error("agent.actor_ref is None, cannot load reference actor state.")
        else:
            agent.logger.warning(f"Reference actor checkpoint {actor_ref_path} not found, but actor_ref attribute exists.")


    # Ensure networks are in train mode after loading (BaseAgent.train will handle this if called after)
    if agent.actor: agent.actor.train()
    if hasattr(agent, "critic") and agent.critic: agent.critic.train()
    # Also ensure CNN parts are in train mode if they exist
    if hasattr(agent, 'cnn_feature_extractor') and agent.cnn_feature_extractor:
        agent.cnn_feature_extractor.train()
    if hasattr(agent, 'actor_head') and agent.actor_head:
        agent.actor_head.train()
    if hasattr(agent, 'critic_head') and agent.critic_head:
        agent.critic_head.train()

    return start_step