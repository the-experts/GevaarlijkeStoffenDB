import { useState, useRef } from "react";
import ReactMarkdown from "react-markdown";
import { useLoadingDots } from "../hooks/useLoadingDots";
import { executeUnified } from "../services/api";
import "./UnifiedInterface.css";

export default function UnifiedInterface() {
    const [question, setQuestion] = useState("");
    const [file, setFile] = useState(null);
    const [response, setResponse] = useState(null);
    const [loading, setLoading] = useState(false);
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef(null);

    const buttonText = useLoadingDots(loading, "Verwerken");

    // Handle file selection from input
    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
        }
    };

    // Handle drag events
    const handleDragEnter = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile && droppedFile.type === "application/pdf") {
            setFile(droppedFile);
        } else {
            alert("Alleen PDF-bestanden zijn toegestaan.");
        }
    };

    // Handle form submission
    const handleSubmit = async (e) => {
        e.preventDefault();

        // Validate input
        if (!file && !question.trim()) {
            alert("Voer een vraag in of upload een PDF-bestand.");
            return;
        }

        setLoading(true);
        setResponse(null);

        try {
            const data = await executeUnified({
                file: file,
                question: question.trim() || null,
            });
            setResponse(data);
        } catch (err) {
            setResponse({
                success: false,
                workflow_type: "unknown",
                status: "error",
                data: {},
                error: err.message,
            });
        } finally {
            setLoading(false);
        }
    };

    // Clear file selection
    const clearFile = () => {
        setFile(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    // Render response based on workflow type
    const renderResponse = () => {
        if (!response) return null;

        // Error response
        if (!response.success || response.error) {
            return (
                <div className="response error">
                    ❌ Fout: {response.error || "Er is een fout opgetreden"}
                </div>
            );
        }

        // Query workflow response
        if (response.workflow_type === "query") {
            return (
                <div className="answer-box">
                    <div className="workflow-badge query">
                        🤖 Query Agent: {response.data.routing || "unknown"}
                    </div>
                    <strong>Antwoord:</strong>
                    {response.data.answer ? (
                        <ReactMarkdown>{response.data.answer}</ReactMarkdown>
                    ) : (
                        <p>Geen antwoord ontvangen.</p>
                    )}
                    {response.data.db_results_count > 0 && (
                        <div className="metadata">
                            📚 {response.data.db_results_count} relevante documenten gevonden
                        </div>
                    )}
                </div>
            );
        }

        // Ingest workflow response
        if (response.workflow_type === "ingest") {
            return (
                <div className="response success">
                    <div className="workflow-badge ingest">
                        📄 Document Ingest Workflow
                    </div>
                    ✅ {response.data.chunks_stored || 0} chunks succesvol opgeslagen in database
                    {response.data.source_filename && (
                        <div className="metadata">
                            Bestand: {response.data.source_filename}
                        </div>
                    )}
                </div>
            );
        }

        return null;
    };

    return (
        <form onSubmit={handleSubmit} className="unified-form">
            {/* Input area with drag-and-drop */}
            <div
                className={`input-zone ${isDragging ? "dragging" : ""}`}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
            >
                <textarea
                    placeholder="Stel een vraag over de getrainde data..."
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    rows="4"
                    disabled={loading}
                />

                {isDragging && (
                    <div className="drop-overlay">
                        <div className="drop-message">
                            📄 Sleep PDF hier naartoe
                        </div>
                    </div>
                )}
            </div>

            {/* File selection area */}
            <div className="file-area">
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="application/pdf"
                    onChange={handleFileChange}
                    style={{ display: "none" }}
                    id="file-input"
                />
                <label htmlFor="file-input" className="file-button">
                    📎 Kies PDF bestand
                </label>

                {file && (
                    <div className="file-selected">
                        <span>📄 {file.name}</span>
                        <button
                            type="button"
                            onClick={clearFile}
                            className="clear-file"
                            disabled={loading}
                        >
                            ✕
                        </button>
                    </div>
                )}
            </div>

            {/* Submit button */}
            <button type="submit" className="submit" disabled={loading}>
                {loading ? buttonText : file ? "Upload en Verwerk PDF" : "Verstuur Vraag"}
            </button>

            {/* Info message */}
            <div className="info-message">
                💡 Tip: Upload een PDF of stel een vraag. De app detecteert automatisch wat te doen!
                {file && question.trim() && (
                    <div className="priority-note">
                        ⚠️ PDF-bestand heeft prioriteit. Vraag wordt genegeerd.
                    </div>
                )}
            </div>

            {/* Response rendering */}
            {renderResponse()}
        </form>
    );
}