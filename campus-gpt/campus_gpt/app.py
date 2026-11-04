"""Gradio chat UI around :class:`CampusBot`."""
from __future__ import annotations

from .bot import CampusBot


def launch(bot: CampusBot, share: bool = True) -> None:
    """Start the chat UI (``share=True`` gives a public Colab link)."""
    import gradio as gr

    def respond(message, history):
        return bot.answer(message, history)

    kwargs = dict(
        title="Campus GPT - VTU AI&ML assistant",
        description=(
            "Answers from verified VTU rules, the AIML 2022 scheme and ingested course "
            "material; says so when it does not know."
        ),
    )
    try:
        demo = gr.ChatInterface(fn=respond, type="messages", **kwargs)
    except TypeError:  # newer Gradio: messages is the only format
        demo = gr.ChatInterface(fn=respond, **kwargs)
    demo.queue().launch(share=share)
