"""
test_live_all_functions.py
==========================
Comprehensive verification script that invokes and tests EVERY function
in the RAG Document Assistant codebase with real live data and real API calls.
"""

import os
import sys
import tempfile
import numpy as np
from io import BytesIO

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import modules
import config
import utils
import document_processor
import embeddings
import vector_store
import document_manager
import chat_manager
import llm_service
import rag_engine

passed = []
failed = []

def record(test_name: str, success: bool, detail: str = ""):
    status = "PASS" if success else "FAIL"
    print(f"[{status}] {test_name}" + (f" - {detail}" if detail else ""))
    if success:
        passed.append(test_name)
    else:
        failed.append((test_name, detail))

print("=" * 60)
print("TESTING UTILS MODULE")
print("=" * 60)

try:
    h = utils.compute_file_hash(b"test data for hash")
    assert len(h) == 64
    record("utils.compute_file_hash", True, f"Hash: {h[:16]}...")
except Exception as e:
    record("utils.compute_file_hash", False, str(e))

try:
    cleaned = utils.clean_text("  line 1  \n\n\n\nline 2   spaces   ")
    assert "line 1\n\nline 2 spaces" in cleaned
    record("utils.clean_text", True)
except Exception as e:
    record("utils.clean_text", False, str(e))

try:
    s = utils.format_file_size(1024 * 1024 * 3)
    assert s == "3.0 MB"
    record("utils.format_file_size", True, s)
except Exception as e:
    record("utils.format_file_size", False, str(e))

try:
    t = utils.truncate_text("The quick brown fox jumps over the lazy dog", max_chars=20)
    assert t.endswith("...")
    record("utils.truncate_text", True, t)
except Exception as e:
    record("utils.truncate_text", False, str(e))

print("=" * 60)
print("TESTING DOCUMENT PROCESSOR MODULE")
print("=" * 60)

sample_text = (
    "Internet of Things (IoT) protocols enable communication between sensors, "
    "actuators, and cloud gateways. Common IoT protocols include MQTT, CoAP, and HTTP. "
    "MQTT uses a publish-subscribe architecture over TCP/IP, which is lightweight and "
    "well-suited for low-bandwidth networks. CoAP uses UDP and is designed for constrained devices. "
) * 8

try:
    p_txt = document_processor.extract_text_from_txt(sample_text.encode("utf-8"))
    assert len(p_txt) == 1 and p_txt[0]["page"] == 1
    record("document_processor.extract_text_from_txt", True, f"{len(p_txt)} page extracted")
except Exception as e:
    record("document_processor.extract_text_from_txt", False, str(e))

try:
    import docx
    d = docx.Document()
    d.add_paragraph("IoT Architecture Overview.")
    d.add_paragraph("Protocols include MQTT and CoAP.")
    b = BytesIO()
    d.save(b)
    p_docx = document_processor.extract_text_from_docx(b.getvalue())
    assert len(p_docx) == 1
    record("document_processor.extract_text_from_docx", True)
except Exception as e:
    record("document_processor.extract_text_from_docx", False, str(e))

try:
    from pypdf import PdfWriter
    pw = PdfWriter()
    pw.add_blank_page(width=100, height=100)
    b_pdf = BytesIO()
    pw.write(b_pdf)
    p_pdf = document_processor.extract_text_from_pdf(b_pdf.getvalue())
    assert isinstance(p_pdf, list)
    record("document_processor.extract_text_from_pdf", True)
except Exception as e:
    record("document_processor.extract_text_from_pdf", False, str(e))

try:
    pages = [{"page": 1, "text": sample_text}]
    chunks = document_processor.split_into_chunks(pages, "iot_doc.txt", chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 2
    assert chunks[0]["document"] == "iot_doc.txt"
    record("document_processor.split_into_chunks", True, f"Generated {len(chunks)} chunks")
except Exception as e:
    record("document_processor.split_into_chunks", False, str(e))

try:
    res = document_processor.process_document(sample_text.encode("utf-8"), "iot_doc.txt")
    assert res["success"] is True
    assert len(res["chunks"]) > 0
    record("document_processor.process_document", True, f"{res['pages']} pages, {len(res['chunks'])} chunks")
except Exception as e:
    record("document_processor.process_document", False, str(e))

print("=" * 60)
print("TESTING EMBEDDINGS MODULE")
print("=" * 60)

try:
    model = embeddings.load_embedding_model()
    assert model is not None
    record("embeddings.load_embedding_model", True, f"Loaded {config.EMBEDDING_MODEL}")
except Exception as e:
    record("embeddings.load_embedding_model", False, str(e))

try:
    n_test = min(3, len(res["chunks"]))
    vecs = embeddings.generate_embeddings(res["chunks"][:n_test])
    assert vecs.shape == (n_test, 384)
    assert vecs.dtype == np.float32
    record("embeddings.generate_embeddings", True, f"Shape: {vecs.shape}")
except Exception as e:
    record("embeddings.generate_embeddings", False, str(e))

try:
    q_vec = embeddings.generate_query_embedding("What is MQTT architecture?")
    assert q_vec.shape == (1, 384)
    record("embeddings.generate_query_embedding", True, f"Shape: {q_vec.shape}")
except Exception as e:
    record("embeddings.generate_query_embedding", False, str(e))

print("=" * 60)
print("TESTING VECTOR STORE MODULE")
print("=" * 60)

tmp_dir = tempfile.TemporaryDirectory()
tmp_idx = os.path.join(tmp_dir.name, "live_test.faiss")
tmp_meta = os.path.join(tmp_dir.name, "live_test_meta.pkl")

try:
    vstore = vector_store.VectorStore(index_path=tmp_idx, metadata_path=tmp_meta)
    vstore.create_index()
    assert vstore.total_chunks == 0
    record("vector_store.create_index", True)
except Exception as e:
    record("vector_store.create_index", False, str(e))

try:
    all_vecs = embeddings.generate_embeddings(res["chunks"])
    vstore.add_documents(res["chunks"], all_vecs)
    assert vstore.total_chunks == len(res["chunks"])
    record("vector_store.add_documents", True, f"Indexed {vstore.total_chunks} chunks")
except Exception as e:
    record("vector_store.add_documents", False, str(e))

try:
    search_res = vstore.search(q_vec, top_k=3)
    assert len(search_res) == min(3, vstore.total_chunks)
    top_chunk, score = search_res[0]
    assert score > 0.3
    record("vector_store.search", True, f"Top score: {score:.3f}")
except Exception as e:
    record("vector_store.search", False, str(e))

try:
    vstore.save_index()
    assert os.path.exists(tmp_idx) and os.path.exists(tmp_meta)
    record("vector_store.save_index", True)
except Exception as e:
    record("vector_store.save_index", False, str(e))

try:
    reload_store = vector_store.VectorStore(index_path=tmp_idx, metadata_path=tmp_meta)
    loaded = reload_store.load_index()
    assert loaded is True
    assert reload_store.total_chunks == vstore.total_chunks
    record("vector_store.load_index", True, f"Reloaded {reload_store.total_chunks} chunks")
except Exception as e:
    record("vector_store.load_index", False, str(e))

try:
    cfd = vstore.chunks_for_document("iot_doc.txt")
    assert len(cfd) == vstore.total_chunks
    record("vector_store.chunks_for_document", True, f"Found {len(cfd)} chunks")
except Exception as e:
    record("vector_store.chunks_for_document", False, str(e))

try:
    vstore.clear_index()
    assert vstore.total_chunks == 0
    assert not os.path.exists(tmp_idx)
    record("vector_store.clear_index", True)
except Exception as e:
    record("vector_store.clear_index", False, str(e))

tmp_dir.cleanup()

print("=" * 60)
print("TESTING DOCUMENT MANAGER MODULE")
print("=" * 60)

try:
    v_err = document_manager.validate_file("valid.pdf", b"pdf_bytes_here")
    assert v_err == ""
    inv_err = document_manager.validate_file("bad.exe", b"exe_bytes")
    assert "Unsupported file type" in inv_err
    record("document_manager.validate_file", True)
except Exception as e:
    record("document_manager.validate_file", False, str(e))

try:
    # Test document addition in isolated store
    add_res = document_manager.add_document("iot_doc.txt", sample_text.encode("utf-8"))
    assert add_res["success"] is True
    record("document_manager.add_document", True, add_res["message"])
except Exception as e:
    record("document_manager.add_document", False, str(e))

try:
    reg = document_manager.get_document_registry()
    assert len(reg) >= 1
    record("document_manager.get_document_registry", True, f"Registered docs: {len(reg)}")
except Exception as e:
    record("document_manager.get_document_registry", False, str(e))

try:
    names = document_manager.list_document_names()
    assert "iot_doc.txt" in names
    record("document_manager.list_document_names", True, f"Docs: {names}")
except Exception as e:
    record("document_manager.list_document_names", False, str(e))

print("=" * 60)
print("TESTING CHAT MANAGER MODULE")
print("=" * 60)

try:
    chat_manager.init_chat_history()
    chat_manager.clear_history()
    assert chat_manager.get_history() == []
    chat_manager.add_message("user", "What is MQTT?")
    chat_manager.add_message("assistant", "MQTT is a pub/sub protocol.", sources=[{"document": "iot_doc.txt", "page": 1, "score": 0.9}])
    hist = chat_manager.get_history()
    assert len(hist) == 2
    chat_manager.clear_history()
    assert len(chat_manager.get_history()) == 0
    record("chat_manager (init, add, get, clear)", True)
except Exception as e:
    record("chat_manager (init, add, get, clear)", False, str(e))

print("=" * 60)
print("TESTING LIVE LLM SERVICE MODULE (REAL API CALL)")
print("=" * 60)

try:
    assert llm_service.is_configured() is True
    api_k, b_url, mdl = llm_service.get_llm_config()
    record("llm_service.is_configured & get_llm_config", True, f"Model: {mdl}, URL: {b_url}")
except Exception as e:
    record("llm_service.is_configured & get_llm_config", False, str(e))

try:
    llm_answer = llm_service.generate_response("Please answer in 5 words: What is artificial intelligence?")
    assert len(llm_answer) > 0
    record("llm_service.generate_response (LIVE)", True, f"Output: '{llm_answer}'")
except Exception as e:
    record("llm_service.generate_response (LIVE)", False, str(e))

print("=" * 60)
print("TESTING RAG ENGINE MODULE (LIVE END-TO-END)")
print("=" * 60)

active_store = document_manager.get_vector_store()

try:
    retrieved = rag_engine.retrieve_relevant_chunks("What is MQTT architecture?", active_store, top_k=3)
    assert len(retrieved) > 0
    record("rag_engine.retrieve_relevant_chunks", True, f"Retrieved {len(retrieved)} chunks")
except Exception as e:
    record("rag_engine.retrieve_relevant_chunks", False, str(e))

try:
    context = rag_engine.build_context(retrieved)
    assert "[Source 1:" in context
    record("rag_engine.build_context", True)
except Exception as e:
    record("rag_engine.build_context", False, str(e))

try:
    p_simp = rag_engine.build_prompt("What is MQTT?", context, answer_style="Simple", exam_mode=False)
    p_exam = rag_engine.build_prompt("What is MQTT?", context, answer_style="Detailed", exam_mode=True)
    assert "easy language" in p_simp
    assert "Definition, Explanation" in p_exam
    record("rag_engine.build_prompt (Simple & Exam)", True)
except Exception as e:
    record("rag_engine.build_prompt (Simple & Exam)", False, str(e))

try:
    gen_ans = rag_engine.generate_answer(p_simp)
    assert len(gen_ans) > 10
    record("rag_engine.generate_answer (LIVE)", True, f"Sample: {gen_ans[:80]}...")
except Exception as e:
    record("rag_engine.generate_answer (LIVE)", False, str(e))

try:
    q_result = rag_engine.answer_question("Explain MQTT protocol and its architecture.", active_store, answer_style="Simple", exam_mode=False)
    assert q_result["found_context"] is True
    assert len(q_result["answer"]) > 10
    assert len(q_result["sources"]) > 0
    record("rag_engine.answer_question (LIVE)", True, f"Sources: {len(q_result['sources'])}, Answer preview: {q_result['answer'][:80]}...")
except Exception as e:
    record("rag_engine.answer_question (LIVE)", False, str(e))

try:
    exam_result = rag_engine.answer_question("Explain MQTT protocol.", active_store, answer_style="Detailed", exam_mode=True)
    assert exam_result["found_context"] is True
    assert len(exam_result["answer"]) > 10
    record("rag_engine.answer_question [Exam Mode] (LIVE)", True, f"Answer preview: {exam_result['answer'][:80]}...")
except Exception as e:
    record("rag_engine.answer_question [Exam Mode] (LIVE)", False, str(e))

try:
    summary_result = rag_engine.summarize_document("iot_doc.txt", active_store, max_chunks_per_batch=4)
    assert len(summary_result) > 20
    record("rag_engine.summarize_document (LIVE Map-Reduce)", True, f"Length: {len(summary_result)} chars")
except Exception as e:
    record("rag_engine.summarize_document (LIVE Map-Reduce)", False, str(e))

try:
    sugg = rag_engine.generate_suggested_questions("iot_doc.txt", active_store)
    assert len(sugg) >= 3
    record("rag_engine.generate_suggested_questions (LIVE)", True, f"Generated: {sugg[:2]}")
except Exception as e:
    record("rag_engine.generate_suggested_questions (LIVE)", False, str(e))

# Clean up test document from active database so the user's workspace remains clean
document_manager.clear_all_documents()

print("\n" + "=" * 60)
print(f"VERIFICATION COMPLETE: {len(passed)} PASSED, {len(failed)} FAILED")
print("=" * 60)

if failed:
    print("\nFAILURES:")
    for name, err in failed:
        print(f" - {name}: {err}")
    sys.exit(1)
else:
    print("\nALL FUNCTIONS ARE WORKING PERFECTLY!")
    sys.exit(0)
