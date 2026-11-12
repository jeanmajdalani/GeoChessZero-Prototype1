
from pathlib import Path
from importlib import import_module

import chess
import chess.svg
import streamlit as st
import streamlit.components.v1 as components
import torch

from shared.chess_utils import encode_board_for_transformer
from shared.move_encoder import legal_move_mask, move_to_index


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ChessPolicyModel = import_module(
    "prototypes.01_geochesszero.policy_value"
).ChessPolicyModel

CHECKPOINTS = {
    "GeoChessZero": {
        "filename": "geochesszero_lichess_best.pt",
        "use_geometry": True,
    },
    "Standard Transformer": {
        "filename": "standard_transformer_lichess_best.pt",
        "use_geometry": False,
    },
}


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model(model_name):
    config = CHECKPOINTS[model_name]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = (
        ROOT / "checkpoints" / config["filename"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = ChessPolicyModel(
        use_geometry=config["use_geometry"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, device, checkpoint


# ============================================================
# 3. AI MOVE PREDICTION
# ============================================================

@torch.no_grad()
def predict_ai_move(model, board, device):
    if board.is_game_over(claim_draw=True):
        return None, 0.0

    position = encode_board_for_transformer(board)

    X = torch.as_tensor(
        position,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    mask = torch.as_tensor(
        legal_move_mask(board),
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)

    logits = model(
        X,
        legal_mask=mask,
    )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )[0]

    legal_moves = list(board.legal_moves)

    best_move = max(
        legal_moves,
        key=lambda move: probabilities[
            move_to_index(move)
        ].item(),
    )

    confidence = probabilities[
        move_to_index(best_move)
    ].item()

    return best_move, confidence


# ============================================================
# 4. GAME STATE
# ============================================================

def reset_game():
    st.session_state.board = chess.Board()
    st.session_state.history = []
    st.session_state.last_ai_move = None
    st.session_state.last_ai_confidence = None


def initialize_game():
    if "board" not in st.session_state:
        reset_game()


# ============================================================
# 5. CHESSBOARD DISPLAY
# ============================================================

def display_board(board):
    last_move = (
        board.peek()
        if board.move_stack
        else None
    )

    board_svg = chess.svg.board(
        board=board,
        size=520,
        lastmove=last_move,
        coordinates=True,
        colors={
            "square light": "#f0d9b5",
            "square dark": "#b58863",
            "margin": "#ffffff",
        },
    )

    # Render the SVG inside an HTML iframe.
    # This fixes the blank chessboard problem.
    components.html(
        board_svg,
        height=550,
        scrolling=False,
    )


# ============================================================
# 6. HUMAN MOVE
# ============================================================

def play_human_move(board, move):
    san = board.san(move)

    st.session_state.history.append(
        {
            "player": "White",
            "move": san,
            "uci": move.uci(),
        }
    )

    board.push(move)


# ============================================================
# 7. AI MOVE
# ============================================================

def play_ai_move(model, board, device):
    move, confidence = predict_ai_move(
        model,
        board,
        device,
    )

    if move is None:
        return

    san = board.san(move)

    st.session_state.history.append(
        {
            "player": "AI",
            "move": san,
            "uci": move.uci(),
            "confidence": confidence,
        }
    )

    st.session_state.last_ai_move = san
    st.session_state.last_ai_confidence = confidence

    board.push(move)


# ============================================================
# 8. GAME HISTORY
# ============================================================

def display_history():
    history = st.session_state.history

    if not history:
        st.write("No moves played yet.")
        return

    for index, item in enumerate(history):
        move_number = index // 2 + 1

        if item["player"] == "White":
            st.write(
                f"**{move_number}. White:** "
                f"{item['move']}"
            )
        else:
            confidence = item.get(
                "confidence",
                0.0,
            )

            st.write(
                f"**{move_number}... AI:** "
                f"{item['move']} "
                f"({confidence:.2%})"
            )


# ============================================================
# 9. MAIN STREAMLIT APPLICATION
# ============================================================

def main():
    st.set_page_config(
        page_title="GeoChessZero - Play AI",
        page_icon="♟️",
        layout="wide",
    )

    initialize_game()

    st.title("♟️ GeoChessZero - Play Against AI")

    st.write(
        "Interactive chess interface comparing "
        "a Geometry-Aware Transformer with "
        "a Standard Transformer."
    )

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    with st.sidebar:
        st.header("AI Configuration")

        model_name = st.selectbox(
            "Choose AI model",
            list(CHECKPOINTS.keys()),
        )

        if st.button(
            "Reset Game",
            use_container_width=True,
        ):
            reset_game()
            st.rerun()

        st.info(
            "You play White.\n\n"
            "The AI plays Black."
        )

        st.divider()

        st.subheader("Research Information")

        st.write("Project: GeoChessZero")
        st.write("Architecture: Chess Transformer")
        st.write("Dataset: Lichess")
        st.write("Actions: 4272")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:
        model, device, checkpoint = load_model(
            model_name
        )

    except Exception as error:
        st.error(
            f"Error loading model: {error}"
        )
        st.stop()

    board = st.session_state.board

    # --------------------------------------------------------
    # MAIN LAYOUT
    # --------------------------------------------------------

    left, right = st.columns(
        [1.3, 1],
        gap="large",
    )

    # --------------------------------------------------------
    # LEFT: CHESS BOARD
    # --------------------------------------------------------

    with left:
        st.subheader("Chess Board")

        display_board(board)

        st.caption(
            "Highlighted squares show the last move."
        )

    # --------------------------------------------------------
    # RIGHT: GAME CONTROLS
    # --------------------------------------------------------

    with right:
        st.subheader("Game Controls")

        st.write(f"**Model:** {model_name}")
        st.write(
            f"**Checkpoint epoch:** "
            f"{checkpoint['epoch']}"
        )
        st.write(f"**Device:** {device}")

        st.divider()

        # GAME OVER
        if board.is_game_over(claim_draw=True):
            result = board.result(
                claim_draw=True
            )

            st.success(
                f"Game finished! Result: {result}"
            )

        # HUMAN TURN
        elif board.turn == chess.WHITE:
            st.write("### Your turn (White)")

            legal_moves = list(
                board.legal_moves
            )

            move_options = {
                f"{board.san(move)} ({move.uci()})": move
                for move in legal_moves
            }

            selected_move = st.selectbox(
                "Choose your move",
                list(move_options.keys()),
            )

            if st.button(
                "Play Move",
                type="primary",
                use_container_width=True,
            ):
                move = move_options[
                    selected_move
                ]

                play_human_move(
                    board,
                    move,
                )

                st.rerun()

        # AI TURN
        else:
            st.write("### AI turn (Black)")

            if st.button(
                "AI Play Move",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner(
                    "AI is thinking..."
                ):
                    play_ai_move(
                        model,
                        board,
                        device,
                    )

                st.rerun()

        # ----------------------------------------------------
        # LAST AI MOVE
        # ----------------------------------------------------

        if st.session_state.last_ai_move:
            st.success(
                "Last AI move: "
                + st.session_state.last_ai_move
            )

            confidence = (
                st.session_state.last_ai_confidence
            )

            if confidence is not None:
                st.write(
                    f"**Model probability:** "
                    f"{confidence:.2%}"
                )

        st.divider()

        # ----------------------------------------------------
        # GAME STATUS
        # ----------------------------------------------------

        st.subheader("Game Status")

        st.write(
            "**Turn:** "
            + (
                "White"
                if board.turn == chess.WHITE
                else "Black"
            )
        )

        st.write(
            f"**Legal moves:** "
            f"{board.legal_moves.count()}"
        )

        st.write(
            f"**Moves played:** "
            f"{len(board.move_stack)}"
        )

        st.divider()

        # ----------------------------------------------------
        # MOVE HISTORY
        # ----------------------------------------------------

        st.subheader("Move History")

        display_history()

        st.divider()

        st.caption(
            "This is a research prototype trained "
            "on human chess games. "
            "It predicts moves but is not optimized "
            "for winning against strong chess engines."
        )


# ============================================================
# 10. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
