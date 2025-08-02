import re
import json
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player


class MyExampleAgent(Agent):

    def _setup(self):
        self.llm = OpenAIGomokuClient(model="gemma2-9b-it")

    async def get_move(self, game_state):
        player = self.player.value
        rival = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

        board_str = game_state.format_board("standard")

        messages = [
            {
                "role": "system",
                "content": f"You are a professional Gomoku (5-in-a-row) player. You are playing as {player}, and your opponent is {rival}. Your task is to examine the current 8x8 board and select the best next move to increase your chances of winning. Aim for a strategic advantage or to block your opponent if necessary.",
            },
            {
                "role": "user",
                "content": f"""Here is the current board. The grid is 8x8, with row and column indices labeled. Cells contain:
- "." for empty
- "{player}" for your stones
- "{rival}" for opponent's stones

{board_str}

Respond with the best next move using this exact JSON format (no explanation):

{{ "row": <row_number>, "col": <col_number> }}""",
            },
        ]

        content = await self.llm.complete(messages)

        try:
            if m := re.search(r"{[^}]+}", content, re.DOTALL):
                move = json.loads(m.group(0))
                row, col = (move["row"], move["col"])
                if game_state.is_valid_move(row, col):
                    return (row, col)
        except json.JSONDecodeError as e:
            pass

        return game_state.get_legal_moves()[0]
