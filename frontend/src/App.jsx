import { useState } from "react";
import PdfUploadTab from "./components/PdfUploadTab";
import QueryTab from "./components/QueryTab";
import "./index.css";

export default function App() {
    const [activeTab, setActiveTab] = useState("upload");

    return (
        <div className="app-container">
            <h1>☣️ GevaarlijkeStoffenDB ☣️</h1>

            <div className="tabs">
                <button
                    className={`tab-button ${activeTab === "upload" ? "active" : ""}`}
                    onClick={() => setActiveTab("upload")}
                >
                    Upload PDF
                </button>
                <button
                    className={`tab-button ${activeTab === "query" ? "active" : ""}`}
                    onClick={() => setActiveTab("query")}
                >
                    Vraag stellen
                </button>
            </div>

            {activeTab === "upload" ? <PdfUploadTab /> : <QueryTab />}
        </div>
    );
}
