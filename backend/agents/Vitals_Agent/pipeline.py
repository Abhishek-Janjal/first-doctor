from langgraph.graph import StateGraph, END
from state import VitalsState
from .nodes import ppg_interpreter, audio_classifier, clinical_summarizer


def build_vitals_graph():
    graph = StateGraph(VitalsState)

    graph.add_node("ppg", ppg_interpreter)
    graph.add_node("audio", audio_classifier)
    graph.add_node("summary", clinical_summarizer)

    graph.set_entry_point("ppg")
    graph.add_edge("ppg", "audio")
    graph.add_edge("audio", "summary")
    graph.add_edge("summary", END)

    return graph.compile()


vitals_graph = build_vitals_graph()


def run_vitals_pipeline(ppg_signal, ppg_fps, audio_transcript, audio_duration):
    state = {
        "ppg_signal": ppg_signal,
        "ppg_fps": ppg_fps,
        "audio_transcript": audio_transcript,
        "audio_duration": audio_duration,
    }
    return vitals_graph.invoke(state)