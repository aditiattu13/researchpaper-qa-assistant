import gradio as gr

from rag import answer, build_index, parse_pdf, summarize


def load_paper(file_path):
    if not file_path:
        return None, "Please upload a PDF first."

    try:
        chunks = parse_pdf(file_path)

        if not chunks:
            return None, (
                "No extractable text was found. The PDF may be scanned or image-only; "
                "OCR is not currently implemented."
            )

        index = build_index(chunks)
        state = {"chunks": chunks, "index": index}
        return state, f"Paper indexed successfully: **{len(chunks)} text chunks**."

    except Exception as exc:
        return None, f"Could not process this PDF: {type(exc).__name__}: {exc}"


def ask_question(question, state):
    if not state:
        return "Upload and index a paper first.", ""

    if not question or not question.strip():
        return "Please enter a question.", ""

    try:
        response, hits = answer(
            question, state["index"], state["chunks"]
        )

        sources = "\n\n".join(
            f"**[{i + 1}] {hit['section']}, page {hit['page']}**\n\n"
            f"{hit['text'][:500]}{'...' if len(hit['text']) > 500 else ''}"
            for i, hit in enumerate(hits)
        )

        return response, sources or "No source passages were retrieved."

    except Exception as exc:
        return (
            f"An error occurred: {type(exc).__name__}: {exc}",
            ""
        )


def summarize_paper(state):
    if not state:
        return "Upload and index a paper first."

    try:
        return summarize(state["index"], state["chunks"])
    except Exception as exc:
        return f"Could not summarize the paper: {type(exc).__name__}: {exc}"


with gr.Blocks(title="Research Paper Assistant") as demo:
    gr.Markdown(
        """
        # Research Paper Assistant
        Upload a scientific paper, ask questions about its contents, and generate
        a structured summary. Answers are grounded in retrieved passages and
        include source references.
        """
    )

    paper_state = gr.State(value=None)

    with gr.Row():
        pdf_file = gr.File(
            label="Upload research paper",
            file_types=[".pdf"],
            type="filepath"
        )
        load_button = gr.Button("Process PDF", variant="primary")

    status = gr.Markdown("Upload a PDF and select **Process PDF**.")

    load_button.click(
        fn=load_paper,
        inputs=pdf_file,
        outputs=[paper_state, status]
    )

    with gr.Tabs():
        with gr.Tab("Ask a question"):
            question = gr.Textbox(
                label="Your question",
                placeholder="Ask something about the uploaded paper...",
                lines=2
            )
            ask_button = gr.Button("Ask", variant="primary")
            answer_output = gr.Markdown(label="Answer")
            source_output = gr.Markdown(label="Retrieved sources")

            ask_button.click(
                fn=ask_question,
                inputs=[question, paper_state],
                outputs=[answer_output, source_output]
            )
            question.submit(
                fn=ask_question,
                inputs=[question, paper_state],
                outputs=[answer_output, source_output]
            )

        with gr.Tab("Paper summary"):
            summary_button = gr.Button("Summarize paper", variant="primary")
            summary_output = gr.Markdown()

            summary_button.click(
                fn=summarize_paper,
                inputs=paper_state,
                outputs=summary_output
            )

    gr.Markdown(
        """
        ---
        **Note:** This application works with text-based PDFs. Scanned PDFs
        requiring OCR are not currently supported.
        """
    )


if __name__ == "__main__":
    demo.queue().launch()
