// Switch between Login & Signup forms
function switchForm(form) {
  document.getElementById("login-box").classList.remove("active");
  document.getElementById("signup-box").classList.remove("active");

  if (form === "login") document.getElementById("login-box").classList.add("active");
  else document.getElementById("signup-box").classList.add("active");
}

// Show Forgot Password Modal
function showForgotPassword() {
  document.getElementById("forgotPasswordModal").style.display = "flex";
}

// Close Modal
function closeModal(id) {
  document.getElementById(id).style.display = "none";
}

// Flash close button
document.querySelectorAll(".flash-close").forEach(btn => {
  btn.addEventListener("click", () => btn.parentElement.remove());
});

// Optional: Send reset link (dummy)
function sendResetLink() {
  const email = document.getElementById("reset-email").value;
  alert("Password reset link sent to: " + email);
  closeModal("forgotPasswordModal");
}

// Initialize chat session only if on /chat page
document.addEventListener("DOMContentLoaded", () => {
  if (!window.location.pathname.startsWith("/chat")) return;

  async function initChat() {
      try {
          const res = await fetch("/chat/init_session", {
              method: "POST",
              headers: { "Content-Type": "application/json" }
          });
          const data = await res.json();
          console.log("Chat session started:", data);
          // Display welcome message and first question
          const chatContainer = document.getElementById("chat-container");
          if (chatContainer) {
              chatContainer.innerHTML = `<p>${data.welcome_message}</p><p>${data.question.question}</p>`;
          }
      } catch (err) {
          console.error("Error initializing chat:", err);
      }
  }

  initChat();
});
