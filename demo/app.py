
import streamlit as st
import chess
import chess.svg

import streamlit.components.v1 as components


st.set_page_config(
    page_title="GeoChessZero",
    page_icon="♟️",
    layout="wide"
)


# Initialize chess board
if "board" not in st.session_state:
    st.session_state.board = chess.Board()


board = st.session_state.board


st.title("♟️ GeoChessZero")
st.subheader("Geometry-Aware Chess Transformer")

st.write(
    "Interactive chess interface for our "
    "Generative AI research project."
)


# Display chess board
col1, col2 = st.columns([2, 1])

with col1:
    svg_board = chess.svg.board(
        board=board,
        size=500
    )

    components.html(
        svg_board,
        height=520
    )


with col2:
    st.header("Game Controls")

    st.write(
        "Turn:",
        "White" if board.turn == chess.WHITE else "Black"
    )

    legal_moves = list(board.legal_moves)

    st.write("Legal moves:", len(legal_moves))

    if not board.is_game_over():

        move_options = {
            board.san(move): move
            for move in legal_moves
        }

        selected_move = st.selectbox(
            "Choose a move",
            options=list(move_options.keys())
        )

        if st.button("Play Move"):
            board.push(move_options[selected_move])
            st.rerun()

    if st.button("Reset Game"):
        st.session_state.board = chess.Board()
        st.rerun()

    st.divider()

    st.header("AI Models")

    st.write("Standard Transformer: Not trained")
    st.write("GeoChessZero: Not trained")

    st.info(
        "AI move prediction will be connected "
        "in the next steps."
    )


st.divider()

st.write("### Current Position (FEN)")
st.code(board.fen())
