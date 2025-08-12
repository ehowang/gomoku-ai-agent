# Import necessary modules for regular expressions, JSON parsing, and the Gomoku framework
import os
import re
import json
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player
from dotenv import load_dotenv
load_dotenv()

class MyExampleAgent(Agent):
    """
    A Gomoku AI agent that uses a language model to make strategic moves.
    Inherits from the base Agent class provided by the Gomoku framework.
    """

    def _setup(self):
        """
        Initialize the agent by setting up the language model client.
        This method is called once when the agent is created.
        """
        # Create an OpenAI-compatible client using the Gemma2 model for move generation
        self.llm = OpenAIGomokuClient(
            model="qwen/qwen3-8b",
            api_key=os.getenv("PROF_API_KEY"),
            endpoint=os.getenv("PROF_BASE_URL"),
        )
    def _create_system_prompt(self,player,rival) -> str:
        """Create the system prompt that teaches the LLM how to play Gomoku."""
        return """
You are a Gomoku move selector.
Think silently and do not reveal your reasoning.
Return only a single JSON object with two integer fields: "row" and "col".
Never include any other text, explanations, tags, or code fences.
""".strip()
    

    async def get_move(self, game_state):
        """
        Generate the next move for the current game state using an LLM.

        Args:
            game_state: Current state of the Gomoku game board

        Returns:
            tuple: (row, col) coordinates of the chosen move
        """
        # Get the current player's symbol (e.g., 'X' or 'O')
        player = self.player.value

        # Determine the opponent's symbol by checking which player we are
        rival = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

        # Convert the game board to JSON string and gather legal moves
        board_str = game_state.format_board("json")
        board_size = game_state.board_size
        legal_moves = game_state.get_legal_moves()
        # Ensure JSON-serializable list of [row, col]
        legal_moves_list = [[r, c] for (r, c) in legal_moves]

        # Prepare the conversation messages for the language model
        messages = [
            {
                "role": "system",
                "content": self._create_system_prompt(player,rival),
            },
            {
                "role": "user",
                "content": f"""You are {player}. Opponent is {rival}. Grid is {board_size}x{board_size} with 0-based indices.

Board (JSON):
{board_str}

Legal moves (choose exactly one pair from this list):
{json.dumps(legal_moves_list)}

Priorities:
1) If you can win in one move this turn, play that move.
2) Else, if the opponent can win on their next turn, block that move.
3) Else, extend/defend your longest line, preferring central positions and adjacency to your stones.

Output exactly one line, JSON only, no other text:
{{"row": <int>, "col": <int>}}""",
            },
        ]

        # Send the messages to the language model and get the response
        content = await self.llm.complete(messages)

        # Helpers
        def parse_last_row_col_pair(text: str):
            pattern = r'\{\s*"row"\s*:\s*-?\d+\s*,\s*"col"\s*:\s*-?\d+\s*\}'
            matches = re.findall(pattern, text, flags=re.DOTALL)
            if not matches:
                return None
            try:
                obj = json.loads(matches[-1])
                return int(obj["row"]), int(obj["col"])
            except Exception:
                return None

        legal_moves_set = {(r, c) for (r, c) in legal_moves}

        # Try to parse and validate the first response
        parsed = parse_last_row_col_pair(content)
        if parsed is not None:
            row, col = parsed
            if (row, col) in legal_moves_set and game_state.is_valid_move(row, col):
                return (row, col)

        # One repair attempt if invalid or unparsable
        repair_messages = [
            {
                "role": "system",
                "content": self._create_system_prompt(player, rival),
            },
            {
                "role": "user",
                "content": f"""Your previous selection was invalid.
Choose exactly one pair from these legal moves and return JSON only:
{json.dumps(legal_moves_list)}

Output exactly: {{"row": <int>, "col": <int>}}""",
            },
        ]
        content2 = await self.llm.complete(repair_messages)
        parsed2 = parse_last_row_col_pair(content2)
        if parsed2 is not None:
            row2, col2 = parsed2
            if (row2, col2) in legal_moves_set and game_state.is_valid_move(row2, col2):
                return (row2, col2)

        # Smarter fallback: choose center-most legal move
        if not legal_moves:
            # No legal moves available (shouldn't happen during a move request)
            return game_state.get_legal_moves()[0]

        center_row = (board_size - 1) / 2.0
        center_col = (board_size - 1) / 2.0
        def center_distance(move):
            r, c = move
            return abs(r - center_row) + abs(c - center_col)

        best_move = min(legal_moves, key=center_distance)
        return best_move
