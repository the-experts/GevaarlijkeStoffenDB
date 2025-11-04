import { useState } from "react";

export default function QueryTab() {
    const [question, setQuestion] = useState("");
    const [response, setResponse] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleAsk = async (e) => {
        e.preventDefault();
        setLoading(true);
        setResponse(null);

        try {
            const res = await fetch("http://localhost:8000/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
            });
            const data = await res.json();
            setResponse(data);
        } catch (err) {
            setResponse({ error: err.message });
        } finally {
            setLoading(false);
        }
    };

    return (
        <form onSubmit={handleAsk}>
      <textarea
          placeholder="Stel een vraag over de getrainde data..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows="4"
      />
            <button type="submit" className="submit" disabled={loading}>
                {loading ? "Bezig..." : "Verstuur vraag"}
            </button>

            {response && (
                <div className="answer-box">
                    <strong>Antwoord:</strong>
                    <p>{response.answer || response.error || "Geen antwoord ontvangen."}</p>
                </div>
            )}
        </form>
    );
}
