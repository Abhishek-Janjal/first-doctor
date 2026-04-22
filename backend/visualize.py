# visualize.py
from backend.agents.PredDoc import medical_assessment_graph

graph = medical_assessment_graph

try:
    with open("graph_viz.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
    print("✅ Saved as graph_viz.png")
except Exception as e:
    print(f"Mermaid failed ({e}), trying Graphviz...")
    try:
        with open("graph_viz.png", "wb") as f:
            f.write(graph.get_graph().draw_png())
        print("✅ Saved as graph_viz.png (Graphviz)")
    except Exception as e2:
        print(f"❌ Both renderers failed: {e2}")
        print("\nFalling back to ASCII:\n")
        graph.get_graph().print_ascii()
