import html
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import List

class HtmlQLLogger:
    def __init__(self, 
                 path: str | Path, 
                 title: str = "Q-Learning Logger", 
                 append: bool = False, 
                 max_games_to_write: int = 1000, 
                 game_write_interval: int = 1):
        assert append==False, "Append mode not yet supported."
        self.path = Path(path)
        self.title = title
        self.mode = "a" if append and self.path.exists() else "w"
        self._f = None
        self._opened = False
        self.max_games_to_write = max_games_to_write
        self.game_write_interval = game_write_interval
        self.num_games_written = 0
        self.num_games_played = 0
        self.writing_current_game = False
        self.num_columns = 5 

    def __enter__(self):
        self._f = self.path.open(self.mode, encoding="utf-8")
        if self.mode == "w":
            self._write_header()
        # If appending to an existing file, assume it already has header/table open.
        self._opened = True
        return self

    def __exit__(self, exc_type, exc, tb):
        # Always write footer if we created the document
        if self._opened and self.mode == "w":
            self._write_footer()
        if self._f:
            self._f.close()
        # Do not suppress exceptions
        return False
    
    def _board_as_text_lines(self, board: np.ndarray) -> List[str]:
        if board.shape == (2, 6, 7):
            me, opp = board[0], board[1]
        elif board.shape == (6, 7, 2):
            me, opp = board[...,0], board[...,1]
        else:
            raise ValueError(f"Unexpected obs shape  {board.shape}")

        grid = me.astype(np.int8) - opp.astype(np.int8)  # +1 me, -1 opp
        sym = {0: ".", 1: "O", -1: "X"}
        lines = []
        for r in range(6):  # show top row first
            lines.append("| " + "".join(sym[int(grid[r, c])] for c in range(7)) + " |")
        return lines
    
    # ---- Public API ---------------------------------------------------------

    def start_game(self):
        if self.num_games_written >= self.max_games_to_write:
            return

        if (self.num_games_played % self.game_write_interval) != 0:
            return

        if not self._f:
            raise RuntimeError("Logger not opened. Use 'with HtmlQLLogger(...) as log:'")
                
        self.writing_current_game = True

        self._f.write(f"<tr><td colspan='{self.num_columns}'></td></tr>\n")
        self._f.flush()

    def end_game(self):
        self.num_games_played += 1

        if self.writing_current_game:
            self.num_games_written += 1
            self.writing_current_game = False
            self._f.write(f"<tr><td colspan='{self.num_columns}' class='meta'></td></tr>\n")
            self._f.flush()

        if self.num_games_written >= self.max_games_to_write:
            return
        
        if not self._f:
            raise RuntimeError("Logger not opened. Use 'with HtmlQLLogger(...) as log:'")

    def add_row(self, 
                step: int, 
                q_scores : np.ndarray, 
                mask : np.ndarray, 
                board: np.ndarray,
                chosen_move: int):
        
        if not self.writing_current_game:
            return
        
        if not self._f:
            raise RuntimeError("Logger not opened. Use 'with HtmlQLLogger(...) as log:'")

        q_str = np.array2string(q_scores, formatter={'float_kind':lambda x: f"{x:8.3f}"})
        m_str = np.array2string(mask.astype(int), formatter={'int':lambda x: "." if x==1 else "x"})

        board_lines = self._board_as_text_lines(board)
        board_html = "<br>".join(html.escape(b) for b in board_lines)
     
        q_html = html.escape(q_str)
        m_html = html.escape(m_str)
        move_html = html.escape(str(chosen_move))

        self._f.write(
            f"<tr>"
            f"<td>{step}</td>"
            f"<td class='board'>{board_html}</td>"
            f"<td>{q_html}</td>"
            f"<td>{m_html}</td>"
            f"<td>{move_html}</td>"
            f"</tr>\n"
        )
        self._f.flush()  # helpful for long runs

    # ---- Internals ----------------------------------------------------------
    def _write_header(self):
        t = html.escape(self.title)
        now = html.escape(datetime.now().isoformat(timespec="seconds"))
        self._f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{t}</title>
  <style>
    body {{ font-family: sans-serif; }}
    table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
    th, td {{ border: 1px solid #ccc; padding: 4px 8px; vertical-align: top; white-space: nowrap; width: 1%; }}
    td:nth-child(1) {{ width: 6ch; }}
    td:nth-child(2), td:nth-child(3) {{ width: 24ch; }}
    .board {{ font-family: monospace; white-space: pre; line-height: 1.1; }}
    .meta {{ color: #666; font-size: 0.9em; margin: 0.5rem 0 1rem; }}
  </style>
</head>
<body>
<h1>{t}</h1>
<p class="meta">Started: {now}</p>
<table>
  <tr>
    <th>Step</th>
    <th>Board</th>
    <th>Q-scores</th>
    <th>Mask</th>
    <th>Move</th>
  </tr>
""")

    def _write_footer(self):
        self._f.write("</table>\n</body>\n</html>\n")
        self._f.flush()