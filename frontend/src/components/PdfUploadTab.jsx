import { useState } from "react";
import { BASE_URL } from "../config";

export default function PdfUploadTab() {
    const [file, setFile] = useState(null);
    const [response, setResponse] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleUpload = async (e) => {
        e.preventDefault();
        if (!file) return alert("Selecteer een PDF-bestand.");

        setLoading(true);
        const formData = new FormData();
        formData.append("file", file);
        formData.append("max_length", 1000);
        formData.append("overlap", 100);

        try {
            const res = await fetch(`${BASE_URL}/process-pdf/`, {
                method: "POST",
                body: formData,
            });
            const data = await res.json();
            setResponse(data);
        } catch (err) {
            setResponse({ success: false, error: err.message });
        } finally {
            setLoading(false);
        }
    };

    return (
        <form onSubmit={handleUpload}>
            <input
                type="file"
                accept="application/pdf"
                onChange={(e) => setFile(e.target.files[0])}
            />
            <button type="submit" className="submit" disabled={loading}>
                {loading ? "Verwerken..." : "Upload PDF"}
            </button>

            {response && (
                <div
                    className={`response ${response.success ? "success" : "error"}`}
                >
                    {response.success
                        ? `✅ ${response.stored_chunks} chunks opgeslagen in database`
                        : `❌ Fout: ${response.error}`}
                </div>
            )}
        </form>
    );
}
