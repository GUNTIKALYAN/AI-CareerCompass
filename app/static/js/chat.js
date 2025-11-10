const chatMessages = document.getElementById("chat-messages");
const userInput = document.getElementById("user-input");
const typingIndicator = document.getElementById("typing-indicator");

function addMessage(message, sender) {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", sender);

    const bubble = document.createElement("div");
    bubble.classList.add("message-bubble");
    bubble.textContent = message;

    msgDiv.appendChild(bubble);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    addMessage(message, "user");
    userInput.value = "";
    typingIndicator.style.display = "flex";

    try {
        const res = await fetch("/career/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message })
        });

        const data = await res.json();
        typingIndicator.style.display = "none";

        if (data.ai_response) {
            addMessage(data.ai_response, "bot");
        }

        if (data.assessment_complete && data.redirect_url) {
            window.location.href = data.redirect_url;
        }
    } catch (err) {
        console.error("Error sending message:", err);
        typingIndicator.style.display = "none";
    }
}

// Optional: Send on Enter key
userInput.addEventListener("keydown", function(e) {
    if (e.key === "Enter") sendMessage();
});
