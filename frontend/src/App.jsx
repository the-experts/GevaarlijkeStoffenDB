import UnifiedInterface from "./components/UnifiedInterface";
import "./index.css";

export default function App() {
    return (
        <div className="app-container">
            <h1>☣️ GevaarlijkeStoffenDB ☣️</h1>
            <p style={{ textAlign: "center", color: "#6b7280", marginBottom: "1.5rem" }}>
                Intelligente interface met automatische workflow detectie
            </p>

            <UnifiedInterface />
        </div>
    );
}
