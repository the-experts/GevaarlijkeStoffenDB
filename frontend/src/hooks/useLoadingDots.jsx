import { useState, useEffect } from "react";

export function useLoadingDots(isLoading, baseText = "Verwerken") {
    const [text, setText] = useState(baseText);

    useEffect(() => {
        if (!isLoading) {
            setText(baseText);
            return;
        }

        let count = 0;
        const interval = setInterval(() => {
            count = (count + 1) % 4; // 0..3
            setText(baseText + " " + ".".repeat(count));
        }, 500); // elke 500ms

        return () => clearInterval(interval);
    }, [isLoading, baseText]);

    return text;
}
