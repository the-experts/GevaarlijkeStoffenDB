import { useState } from "react";
import { BASE_URL } from "../config";
import ReactMarkdown from "react-markdown";
import { useLoadingDots } from "../hooks/useLoadingDots";

export default function QueryTab() {
    const [question, setQuestion] = useState("");
    const [response, setResponse] = useState(null);
    const [loading, setLoading] = useState(false);

    const buttonText = useLoadingDots(loading, "Verwerken");

    const handleAsk = async (e) => {
        e.preventDefault();
        setLoading(true);
        setResponse(null);

        try {
            const res = await fetch(`${BASE_URL}/query`, {
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
                {loading ? buttonText : "Verstuur vraag"}
            </button>

            {response && (
                <div className="answer-box">
                    <strong>Antwoord:</strong>
                    {response.answer ? (
                        <ReactMarkdown>{response.answer}</ReactMarkdown>
                    ) : (
                        <p>{response.error || "Geen antwoord ontvangen."}</p>
                    )}
                </div>
            )}
        </form>
    );
}
