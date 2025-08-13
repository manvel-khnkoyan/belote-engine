import os
import argparse
import time
import numpy as np
from src.stages.player.network import BeloteNetwork
from src.stages.player.ppo.belote_agent import PPOBeloteAgent
from src.stages.player.helper_agents.human import Human
from src.stages.player.helper_agents.randomer import Randomer
from src.stages.player.history import History
from src.deck import Deck
from src.states.trump import Trump
from src.stages.player.env import BeloteEnv
from src.stages.player.simulator import simulate
from src.stages.player.actions import ActionCardPlay


def parse_args():
    """Parse command line arguments"""
    root = os.environ.get("PROJECT_ROOT")
    
    history_dir = os.path.join(root, 'histories')
    os.makedirs(history_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Play Belote against trained AI agents")
    parser.add_argument("--model-path", type=str, default=os.path.join(root, 'models', 'belote_agent.pt'), help="Model path for the agent")
    parser.add_argument("--mode", type=str, default='play', choices=['play', 'observe', 'record', 'replay', 'test'], help="Type of run: play, observe, record, replay, test")
    parser.add_argument("--repeat", type=int, default=500, help="Repeat the game n times")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to play")
    parser.add_argument("--history-dir", type=str, default=history_dir, help="Histories root directory")
    parser.add_argument("--history-file", type=str, default="history-20250807-085153", help="History file name")
    
    return parser.parse_args()


def create_env():
    """Create a new game environment with random setup"""
    trump = Trump()
    trump.set_random_trump()

    deck = Deck()
    deck.reset()
    deck.deal_cards(8)
    deck.reorder_hands(trump)

    # Choose random starting player
    rng = np.random.default_rng(None)
    next_player = int(rng.integers(0, 4))
    
    return BeloteEnv(trump, deck, next_player)


def load_ai_agent(model_path):
    """Load an AI agent from model file"""
    network = BeloteNetwork()
    agent = PPOBeloteAgent(network=network)
    agent.load(model_path)
    return agent


def load_human_agent(model_path):
    """Load a human-controlled agent"""
    network = BeloteNetwork()
    agent = Human(network=network)
    agent.load(model_path)
    return agent

def load_randomer_agent():
    """Load a random agent"""
    agent = Randomer()
    return agent


def get_history_files(history_dir):
    """Get all history files from directory"""
    items = os.listdir(history_dir)
    return [item for item in items if item.startswith('history-') and os.path.isfile(os.path.join(history_dir, item))]


def fn_play(args):
    """Human plays against AI agents"""
    env = create_env()

    agents = []
    for i in range(4):
        agent = load_human_agent(args.model_path) if i == 0 else load_ai_agent(args.model_path)
        agents.append(agent)

    simulate(env, agents, display=0)
    gain, lose = env.total_scores[0], env.total_scores[1]
    
    # Display final result
    result = "You won!" if gain > lose else "You lost!"
    print(f"\n{result} Final score: {gain} - {lose}")


def fn_observe(args):
    """Watch AI agents play against each other"""
    agents = [
        load_ai_agent(args.model_path),
        load_randomer_agent(),
        load_ai_agent(args.model_path), 
        load_randomer_agent()
    ]
    
    wins = 0
    for _ in range(args.episodes):
        env = create_env()
        simulate(env, agents, display=False)
        gain, lose = env.total_scores[0], env.total_scores[1]
        wins += (gain > lose)
    
    print(f"Total played {args.episodes} | Wins: {wins} / Losses: {args.episodes - wins}")


def fn_record(args):
    """Record humans playing against each other"""
    env = create_env()
    agents = [load_human_agent(args.model_path) for _ in range(4)]
    
    history = History.create(env.trump, env.deck, env.next_player)

    # Action selector that records moves
    def choose_action(env, agent, history=history):
        action = agent.choose_action(env)
        history.record_action(agent.env_index, action)
        return action

    simulate(env, agents, action_selector=choose_action, display=True)
    
    # Save history
    history_path = os.path.join(args.history_dir, f"history-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}")
    history.save(history_path)
    print(f"Saved history to {history_path}")


def fn_replay(args):
    """Replay a recorded game from history"""
    # Load history
    history = History()
    history_path = os.path.join(args.history_dir, args.history_file)
    history.load(history_path)
    history.reset()
    
    env = history.create_env()
    agents = [load_ai_agent(args.model_path) for _ in range(4)]

    print(f"Replaying game from: {args.history_file}")

    def choose_action(env, agent, history=history):
        action, _ = history.get_next_action()
        action_made = agent.choose_action(env)
        match = "✓" if actions_equal(action, action_made) else "✗"
        print()
        print(f"Await:  {action} vs Played: {action_made} [{match}]")
        return action

    simulate(env, agents, action_selector=choose_action, display=True)


def fn_test(args):
    """Test AI accuracy against recorded histories"""
    agents = [load_ai_agent(args.model_path) for _ in range(4)]
    
    history_files = get_history_files(args.history_dir)
    if not history_files:
        print("No history files found.")
        return
    
    total_correct = 0
    total_moves = 0
    
    for history_file in history_files:
        # Load history
        history = History()
        history_path = os.path.join(args.history_dir, history_file)
        history.load(history_path)
        print(f"Testing: {history_file}")

        def choose_action(env, agent, history=history):
            nonlocal total_moves
            nonlocal total_correct
            
            total_moves += 1
            action, _ = history.get_next_action()
            action_made = agent.choose_action(env)
            total_correct += 1 if action == action_made else 0
    
            return action
        
        for _ in range(args.repeat):
            env = history.create_env()
            history.reset()            
            simulate(env, agents, action_selector=choose_action, display=False)

    
    accuracy = (total_correct / total_moves * 100) if total_moves > 0 else 0
    print(f"\nOverall accuracy: {accuracy:.2f}% ({total_correct}/{total_moves})")


if __name__ == "__main__":
    args = parse_args()
    
    # Mode-specific validation
    if args.mode == 'replay' and not args.history_file:
        print("Error: --history-file required for replay mode")
        exit(1)
    
    # Execute selected mode
    modes = {
        'play': fn_play,
        'observe': fn_observe,
        'record': fn_record,
        'replay': fn_replay,
        'test': fn_test
    }
    
    modes[args.mode](args)